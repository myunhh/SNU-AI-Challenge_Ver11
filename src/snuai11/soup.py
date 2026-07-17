"""Same-trajectory checkpoint weight averaging (SWA / model-soup style).

Uniformly averages the LoRA adapter safetensors + Score24Head of several
checkpoints of ONE training run into a single adapter/head. Averaging LoRA
A/B factor matrices is a first-order approximation that is only meaningful
for nearby checkpoints of the same trajectory (same init, same basin) —
NEVER average across independent runs or different warm-starts.

The output is one adapter + one head = ONE model at inference; nothing is
combined at inference time, so this is not the rule-banned output ensemble.
Still an unusual artifact: if a souped checkpoint is ever submitted, keep
this provenance note for the reproducibility review.

Usage (post-run, optional — not wired into any runner path):
  python scripts/avg_adapters.py runs/sft32b_v11_ws8/checkpoint-1400 \
      runs/sft32b_v11_ws8/adapter_final --out runs/sft32b_v11_ws8/soup_tail2
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch


def resolve_checkpoint(ckpt: Path) -> tuple[Path, Path]:
    """(adapter_dir, head_path) for a train_sft checkpoint directory.

    Accepts either the checkpoint dir (containing adapter/ + head.pt, the
    train_sft save layout) or the adapter dir itself (head.pt in the parent).
    """
    ckpt = Path(ckpt)
    if (ckpt / "adapter" / "adapter_config.json").exists():
        adapter, head = ckpt / "adapter", ckpt / "head.pt"
    elif (ckpt / "adapter_config.json").exists():
        adapter, head = ckpt, ckpt.parent / "head.pt"
    else:
        raise FileNotFoundError(f"no adapter_config.json under {ckpt}")
    if not head.exists():
        raise FileNotFoundError(f"missing head next to adapter: {head}")
    return adapter, head


def _load_adapter_config(adapter_dir: Path) -> dict:
    return json.loads((adapter_dir / "adapter_config.json").read_text())


def average_safetensors(files: list[Path], out_file: Path) -> int:
    """Uniform mean of identically-keyed safetensors files. Returns #tensors."""
    from safetensors.torch import load_file, save_file

    dicts = [load_file(str(f)) for f in files]
    keys = set(dicts[0])
    for f, d in zip(files[1:], dicts[1:]):
        if set(d) != keys:
            raise ValueError(f"tensor key set differs: {f} — not same-run checkpoints?")
    avg = {}
    for k in dicts[0]:
        tensors = [d[k] for d in dicts]
        if any(t.shape != tensors[0].shape for t in tensors[1:]):
            raise ValueError(f"shape mismatch for {k}")
        avg[k] = torch.stack([t.float() for t in tensors]).mean(dim=0).to(tensors[0].dtype)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(avg, str(out_file))
    return len(avg)


def average_heads(paths: list[Path], out_path: Path) -> None:
    """Uniform mean of Score24Head .pt blobs (exact — the head is linear)."""
    blobs = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    hidden = blobs[0]["hidden"]
    if any(b["hidden"] != hidden for b in blobs):
        raise ValueError("head hidden sizes differ")
    keys = set(blobs[0]["state_dict"])
    if any(set(b["state_dict"]) != keys for b in blobs):
        raise ValueError("head state_dict keys differ")
    avg = {
        k: torch.stack([b["state_dict"][k].float() for b in blobs]).mean(dim=0)
        for k in keys
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": avg, "hidden": hidden}, out_path)


def soup_checkpoints(ckpts: list[Path], out: Path) -> Path:
    """Average >=2 same-run checkpoints into out/ (adapter/ + head.pt).

    The adapter configs must be identical (same base, targets, r/alpha) —
    a cheap guard against mixing runs. Writes soup_manifest.json listing
    the sources. Output layout matches a train_sft checkpoint, so run_pre
    accepts --adapter <out>/adapter directly.
    """
    if len(ckpts) < 2:
        raise ValueError("need at least 2 checkpoints to average")
    resolved = [resolve_checkpoint(c) for c in ckpts]
    adapters = [a for a, _ in resolved]
    heads = [h for _, h in resolved]

    cfg0 = _load_adapter_config(adapters[0])
    for a in adapters[1:]:
        if _load_adapter_config(a) != cfg0:
            raise ValueError(f"adapter_config.json differs: {a} vs {adapters[0]}")

    out = Path(out)
    out_adapter = out / "adapter"
    out_adapter.mkdir(parents=True, exist_ok=True)
    n = average_safetensors(
        [a / "adapter_model.safetensors" for a in adapters],
        out_adapter / "adapter_model.safetensors",
    )
    shutil.copy2(adapters[0] / "adapter_config.json", out_adapter / "adapter_config.json")
    average_heads(heads, out / "head.pt")
    (out / "soup_manifest.json").write_text(
        json.dumps({"sources": [str(c) for c in ckpts], "tensors": n, "weighting": "uniform"}, indent=2)
    )
    return out
