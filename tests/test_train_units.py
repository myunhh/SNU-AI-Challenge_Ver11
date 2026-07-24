import pytest
import torch

from snuai11 import perm
from snuai11.train_sft import (
    LORA_SUFFIXES,
    check_out_reuse,
    dist_env,
    expected_kt,
    kendall_norm_matrix,
    local_accum_for,
    margin_dpo_loss,
)


def test_dist_env_defaults_to_single_process(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert dist_env() == (0, 0, 1)


def test_dist_env_reads_torchrun_vars(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert dist_env() == (1, 1, 2)


def test_local_accum_for_single_process_unchanged():
    assert local_accum_for(4, world_size=1) == 4


def test_local_accum_for_splits_evenly_across_ranks():
    assert local_accum_for(4, world_size=2) == 2


def test_local_accum_for_rejects_indivisible_accum():
    with pytest.raises(ValueError):
        local_accum_for(5, world_size=2)


def test_margin_dpo_loss_decreases_with_gt_margin():
    label = 5
    weak = torch.zeros(1, 24)
    strong = torch.zeros(1, 24)
    strong[0, label] = 5.0
    l_weak = margin_dpo_loss(weak, label, beta=1.0, ce_weight=0.0)
    l_strong = margin_dpo_loss(strong, label, beta=1.0, ce_weight=0.0)
    assert float(l_strong) < float(l_weak)


def test_margin_dpo_loss_targets_adjacent_swaps():
    label = 0  # rank (0,1,2,3)
    negs = [perm.index_of(n) for n in perm.adjacent_swap_neighbors(perm.rank_of_index(label))]
    # raising a NON-neighbor class must not change the pure margin loss
    logits = torch.zeros(1, 24)
    base = margin_dpo_loss(logits, label, beta=1.0, ce_weight=0.0)
    far = [c for c in range(24) if c not in negs and c != label][0]
    logits2 = logits.clone()
    logits2[0, far] = 3.0
    # (log_softmax shifts all classes, so allow tiny change; neighbor raise must hurt much more)
    l_far = margin_dpo_loss(logits2, label, beta=1.0, ce_weight=0.0)
    logits3 = logits.clone()
    logits3[0, negs[0]] = 3.0
    l_neg = margin_dpo_loss(logits3, label, beta=1.0, ce_weight=0.0)
    assert float(l_neg) > float(l_far) >= float(base) - 1e-6


def test_margin_dpo_loss_grad_flows():
    logits = torch.zeros(1, 24, requires_grad=True)
    loss = margin_dpo_loss(logits, 3, beta=1.0, ce_weight=0.2)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_lora_suffixes_language_only_conventions():
    assert "q_proj" in LORA_SUFFIXES and "down_proj" in LORA_SUFFIXES
    assert "lm_head" not in LORA_SUFFIXES and "embed_tokens" not in LORA_SUFFIXES


def test_expected_kt_zero_at_onehot_gt():
    kt_norm = kendall_norm_matrix()
    logits = torch.zeros(1, 24)
    logits[0, 7] = 40.0
    assert float(expected_kt(logits, 7, kt_norm)) < 1e-6


def test_expected_kt_uniform_is_half():
    # mean KT distance from any fixed rank over S4 is 3 -> normalized 0.5
    kt_norm = kendall_norm_matrix()
    logits = torch.zeros(1, 24)
    for label in (0, 5, 23):
        assert float(expected_kt(logits, label, kt_norm)) == pytest.approx(0.5)


def test_expected_kt_prefers_near_miss_over_reversal():
    # confidently wrong on a KT=1 neighbor must cost far less than on the
    # KT=6 reversal — the partial-credit geometry the aux loss encodes
    kt_norm = kendall_norm_matrix()
    label = perm.index_of((0, 1, 2, 3))
    neighbor = perm.index_of(perm.adjacent_swap_neighbors((0, 1, 2, 3))[0])
    reversal = perm.index_of((3, 2, 1, 0))
    near = torch.zeros(1, 24)
    near[0, neighbor] = 40.0
    far = torch.zeros(1, 24)
    far[0, reversal] = 40.0
    e_near = float(expected_kt(near, label, kt_norm))
    e_far = float(expected_kt(far, label, kt_norm))
    assert e_near == pytest.approx(1 / 6, abs=1e-4)
    assert e_far == pytest.approx(1.0, abs=1e-4)
    assert e_near < e_far


def test_expected_kt_grad_flows_and_pushes_gt_up():
    kt_norm = kendall_norm_matrix()
    logits = torch.zeros(1, 24, requires_grad=True)
    loss = expected_kt(logits, 3, kt_norm)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    # minimizing the loss must INCREASE the GT logit (negative gradient)
    assert float(logits.grad[0, 3]) < 0


def _base_out_reuse_config(**overrides):
    from types import SimpleNamespace

    cfg = dict(
        allow_config_drift=False, model_id="m", phase="sft", keep_ratio=0.5,
        diversity_frac=0.2, objectness_weight=0.3, mmr_lambda=0.5,
        motion_weight=0.0, prune_prob=0.75, no_prune=False, max_pixels=1126400,
        lora_r=16, lora_alpha=32, lora_dropout=0.05,
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def test_check_out_reuse_allows_fresh_out(tmp_path):
    check_out_reuse(tmp_path / "never_created", _base_out_reuse_config())


def test_check_out_reuse_allows_matching_continuation(tmp_path, monkeypatch):
    import json

    out = tmp_path / "run"
    ckpt = out / "checkpoint-200"
    ckpt.mkdir(parents=True)
    args = _base_out_reuse_config()
    (ckpt / "train_args.json").write_text(json.dumps(vars(args)))
    # a legit continuation only changes --adapter/--steps, not the config
    # drift fields — must not raise.
    check_out_reuse(out, _base_out_reuse_config())


def test_check_out_reuse_blocks_config_drift(tmp_path):
    import json

    out = tmp_path / "run"
    ckpt = out / "checkpoint-200"
    ckpt.mkdir(parents=True)
    (ckpt / "train_args.json").write_text(json.dumps(vars(_base_out_reuse_config(motion_weight=0.3))))

    with pytest.raises(SystemExit, match="motion_weight"):
        check_out_reuse(out, _base_out_reuse_config(motion_weight=0.0))

    # --allow-config-drift bypasses the check
    check_out_reuse(out, _base_out_reuse_config(motion_weight=0.0, allow_config_drift=True))
