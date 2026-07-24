"""predict_sample unit tests — stage-2 policy + TTA remap end-to-end,
with a fake engine (no model, CPU-only)."""

import json
from types import SimpleNamespace

import pytest
import torch

from snuai11 import perm
from snuai11.fitprune import PruneConfig
from snuai11.infer import check_resume_config, predict_sample, predict_sample_resilient


class FakeSample:
    id = "fake-1"
    caption = "a storyline"
    image_paths = ("p0", "p1", "p2", "p3")
    rank = None


class ViewConsistentEngine:
    """Recovers the view sigma from the permuted image paths and answers with
    logits peaked at the VIEW-space class of a fixed original-space GT — so a
    correct pipeline must remap every view back to the same original class."""

    def __init__(self, gt_class: int, peak: float, enabled: bool = True, full_peak: float | None = None,
                 boost_frac: float = 0.0):
        self.gt_class = gt_class
        self.peak = peak
        self.full_peak = peak if full_peak is None else full_peak
        self.prune_cfg = PruneConfig(enabled=enabled, boost_frac=boost_frac)
        self.calls = {"pruned": 0, "full": 0}
        self.scored_idx_calls = 0

    def prepare(self, images, caption):
        sigma = tuple(int(p[1]) for p in images)  # "p2" -> original input 2
        return ("prep", sigma)

    def keep_mask(self, prep, caption, cfg):
        return "keep-token"

    def scored_idx(self, prep, caption, cfg):
        self.scored_idx_calls += 1
        return "boost-idx"

    def forward_prepared(self, prep, keep=None, idx=None):
        sigma = prep[1]
        c_view = perm.index_of(perm.apply_view(perm.rank_of_index(self.gt_class), sigma))
        is_full = keep is None and idx is None
        peak = self.full_peak if is_full else self.peak
        self.calls["full" if is_full else "pruned"] += 1
        logits = torch.zeros(1, 24)
        logits[0, c_view] = peak
        return logits


def test_stage2_always_runs_full_pass_and_keeps_prediction():
    eng = ViewConsistentEngine(gt_class=7, peak=20.0)
    rec = predict_sample(eng, FakeSample(), n_tta=4, tau=0.10, stage2="always")
    assert eng.calls == {"pruned": 4, "full": 4}
    assert rec["escalated"] is True
    assert rec["pred_class"] == 7
    assert tuple(rec["rank"]) == perm.rank_of_index(7)
    assert 0.0 <= rec["margin"] <= 1.0 and 0.0 <= rec["margin_final"] <= 1.0


def test_stage2_cascade_skips_full_pass_when_confident():
    eng = ViewConsistentEngine(gt_class=3, peak=20.0)  # view-consistent -> high margin
    rec = predict_sample(eng, FakeSample(), n_tta=4, tau=0.10, stage2="cascade")
    assert eng.calls == {"pruned": 4, "full": 0}
    assert rec["escalated"] is False
    assert rec["pred_class"] == 3
    assert rec["margin"] == rec["margin_final"]


def test_stage2_cascade_escalates_below_tau():
    eng = ViewConsistentEngine(gt_class=3, peak=0.0)  # uniform logits -> margin 0
    rec = predict_sample(eng, FakeSample(), n_tta=4, tau=0.10, stage2="cascade")
    assert eng.calls["full"] == 4
    assert rec["escalated"] is True


def test_stage2_off_never_escalates():
    eng = ViewConsistentEngine(gt_class=3, peak=0.0)
    rec = predict_sample(eng, FakeSample(), n_tta=4, tau=0.10, stage2="off")
    assert eng.calls == {"pruned": 4, "full": 0}
    assert rec["escalated"] is False


def test_stage2_skipped_when_pruning_disabled():
    # stage 1 already saw full tokens — a second identical pass adds nothing
    eng = ViewConsistentEngine(gt_class=3, peak=0.0, enabled=False)
    rec = predict_sample(eng, FakeSample(), n_tta=4, tau=0.10, stage2="always")
    assert eng.calls["full"] == 0
    assert rec["escalated"] is False


def test_stage2_full_evidence_can_fix_uncertain_stage1():
    # stage 1 uniform (peak 0), stage 2 confident: aggregate must follow the
    # informative full-token evidence back in ORIGINAL class space
    eng = ViewConsistentEngine(gt_class=11, peak=0.0, full_peak=20.0)
    rec = predict_sample(eng, FakeSample(), n_tta=4, tau=0.10, stage2="always")
    assert rec["pred_class"] == 11
    assert rec["margin_final"] > rec["margin"]


def test_legacy_tta3_cascade_calls_three_views():
    eng = ViewConsistentEngine(gt_class=5, peak=20.0)
    rec = predict_sample(eng, FakeSample(), n_tta=3, tau=0.10, stage2="cascade")
    assert eng.calls == {"pruned": 3, "full": 0}
    assert rec["pred_class"] == 5


def _base_resume_config(**overrides):
    cfg = dict(
        allow_config_drift=False, tta=8, stage2="always", keep_ratio=0.5,
        diversity_frac=0.2, objectness_weight=0.3, mmr_lambda=0.5,
        motion_weight=0.0, no_prune=False, max_pixels=1126400,
        adapter="runs/adapter_a", head="runs/adapter_a/head.pt",
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def test_check_resume_config_allows_fresh_out(tmp_path):
    check_resume_config(tmp_path / "never_created", _base_resume_config())


def test_check_resume_config_allows_no_progress_yet(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "config.json").write_text(json.dumps(vars(_base_resume_config(motion_weight=0.3))))
    # no progress.jsonl written yet -> nothing to protect
    check_resume_config(out, _base_resume_config(motion_weight=0.0))


def test_check_resume_config_allows_matching_resume(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    args = _base_resume_config(motion_weight=0.3)
    (out / "config.json").write_text(json.dumps(vars(args)))
    (out / "progress.jsonl").write_text('{"id": "a"}\n')
    check_resume_config(out, _base_resume_config(motion_weight=0.3))


def test_check_resume_config_blocks_config_drift(tmp_path):
    out = tmp_path / "run"
    out.mkdir()
    (out / "config.json").write_text(json.dumps(vars(_base_resume_config(motion_weight=0.3))))
    (out / "progress.jsonl").write_text('{"id": "a"}\n')

    try:
        check_resume_config(out, _base_resume_config(motion_weight=0.0))
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert "motion_weight" in str(e)

    # --allow-config-drift bypasses the check
    check_resume_config(out, _base_resume_config(motion_weight=0.0, allow_config_drift=True))


def test_check_resume_config_blocks_adapter_swap(tmp_path):
    # 2026-07-24: runs/triage_boost_0.5 silently mixed rows from two
    # different LoRA adapters because "adapter" wasn't drift-checked.
    out = tmp_path / "run"
    out.mkdir()
    (out / "config.json").write_text(json.dumps(vars(_base_resume_config(adapter="runs/adapter_a"))))
    (out / "progress.jsonl").write_text('{"id": "a"}\n')

    try:
        check_resume_config(out, _base_resume_config(adapter="runs/adapter_b"))
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert "adapter" in str(e)


class FlakyOOMEngine:
    """Wraps a real engine; raises torch.cuda.OutOfMemoryError on the first
    n_oom calls to forward_prepared (across retries), then delegates
    normally — simulates the allocator-fragmentation OOM that
    predict_sample_resilient is meant to survive."""

    def __init__(self, inner, n_oom):
        self.inner = inner
        self.n_oom = n_oom
        self.call_count = 0
        self.prune_cfg = inner.prune_cfg

    def prepare(self, images, caption):
        return self.inner.prepare(images, caption)

    def keep_mask(self, prep, caption, cfg):
        return self.inner.keep_mask(prep, caption, cfg)

    def forward_prepared(self, prep, keep):
        self.call_count += 1
        if self.call_count <= self.n_oom:
            raise torch.cuda.OutOfMemoryError("fake oom")
        return self.inner.forward_prepared(prep, keep)


def test_predict_sample_resilient_recovers_from_transient_oom(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    eng = FlakyOOMEngine(ViewConsistentEngine(gt_class=5, peak=20.0), n_oom=1)
    rec = predict_sample_resilient(eng, FakeSample(), n_tta=3, tau=0.10, stage2="cascade", max_retries=3)
    assert rec["pred_class"] == 5
    assert eng.call_count > 1  # first call(s) failed, a later retry succeeded


def test_predict_sample_resilient_reraises_after_max_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    eng = FlakyOOMEngine(ViewConsistentEngine(gt_class=5, peak=20.0), n_oom=1000)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        predict_sample_resilient(eng, FakeSample(), n_tta=3, tau=0.10, stage2="cascade", max_retries=2)


# ---- token boost (2026-07-23) ----------------------------------------------


def test_predict_sample_default_boost_frac_never_calls_scored_idx():
    # legacy callers (boost_frac=0.0, the default) must never even touch the
    # new method -- a hard guarantee that the existing keep_mask/forward_
    # prepared(keep=...) path is completely unaffected by this feature.
    eng = ViewConsistentEngine(gt_class=3, peak=20.0)
    rec = predict_sample(eng, FakeSample(), n_tta=4, tau=0.10, stage2="off")
    assert eng.scored_idx_calls == 0
    assert eng.calls == {"pruned": 4, "full": 0}
    assert rec["pred_class"] == 3


def test_predict_sample_uses_boosted_path_when_boost_frac_positive():
    eng = ViewConsistentEngine(gt_class=3, peak=20.0, boost_frac=0.4)
    rec = predict_sample(eng, FakeSample(), n_tta=4, tau=0.10, stage2="off")
    assert eng.scored_idx_calls == 4  # once per TTA view
    assert eng.calls == {"pruned": 4, "full": 0}  # boosted forward still stage-1, not stage-2
    assert rec["pred_class"] == 3
