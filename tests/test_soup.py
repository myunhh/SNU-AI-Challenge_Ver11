"""Checkpoint soup (same-run weight averaging) unit tests — tiny fake
adapters, CPU-only."""

import json

import pytest
import torch

pytest.importorskip("safetensors")

from snuai11.soup import resolve_checkpoint, soup_checkpoints  # noqa: E402
from snuai11.vlm import Score24Head  # noqa: E402


def make_ckpt(root, name, scale: float, cfg: dict | None = None):
    from safetensors.torch import save_file

    ckpt = root / name
    adapter = ckpt / "adapter"
    adapter.mkdir(parents=True)
    save_file(
        {
            "base.lora_A.weight": torch.full((4, 8), scale),
            "base.lora_B.weight": torch.full((8, 4), 2 * scale),
        },
        str(adapter / "adapter_model.safetensors"),
    )
    (adapter / "adapter_config.json").write_text(json.dumps(cfg or {"r": 16, "lora_alpha": 32}))
    head = Score24Head(hidden_size=8)
    with torch.no_grad():
        head.linear.weight.fill_(scale)
    head.save(ckpt / "head.pt")
    return ckpt


def test_soup_uniform_average(tmp_path):
    from safetensors.torch import load_file

    c1 = make_ckpt(tmp_path, "checkpoint-1400", scale=1.0)
    c2 = make_ckpt(tmp_path, "adapter_final", scale=3.0)
    out = soup_checkpoints([c1, c2], tmp_path / "soup")

    avg = load_file(str(out / "adapter" / "adapter_model.safetensors"))
    assert torch.allclose(avg["base.lora_A.weight"], torch.full((4, 8), 2.0))
    assert torch.allclose(avg["base.lora_B.weight"], torch.full((8, 4), 4.0))

    head = Score24Head.load(out / "head.pt")
    assert torch.allclose(head.linear.weight, torch.full((24, 8), 2.0))

    manifest = json.loads((out / "soup_manifest.json").read_text())
    assert len(manifest["sources"]) == 2
    # output layout is a valid checkpoint itself (run_pre --adapter <out>/adapter)
    adapter_dir, head_path = resolve_checkpoint(out)
    assert adapter_dir == out / "adapter" and head_path == out / "head.pt"


def test_soup_rejects_mismatched_configs(tmp_path):
    c1 = make_ckpt(tmp_path, "a", scale=1.0, cfg={"r": 16})
    c2 = make_ckpt(tmp_path, "b", scale=1.0, cfg={"r": 8})
    with pytest.raises(ValueError, match="adapter_config"):
        soup_checkpoints([c1, c2], tmp_path / "soup")


def test_soup_rejects_single_checkpoint(tmp_path):
    c1 = make_ckpt(tmp_path, "a", scale=1.0)
    with pytest.raises(ValueError, match="at least 2"):
        soup_checkpoints([c1], tmp_path / "soup")


def test_resolve_checkpoint_accepts_adapter_dir_directly(tmp_path):
    c1 = make_ckpt(tmp_path, "checkpoint-200", scale=1.0)
    adapter_dir, head_path = resolve_checkpoint(c1 / "adapter")
    assert adapter_dir == c1 / "adapter" and head_path == c1 / "head.pt"
