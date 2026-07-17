"""predict_sample unit tests — stage-2 policy + TTA remap end-to-end,
with a fake engine (no model, CPU-only)."""

import torch

from snuai11 import perm
from snuai11.fitprune import PruneConfig
from snuai11.infer import predict_sample


class FakeSample:
    id = "fake-1"
    caption = "a storyline"
    image_paths = ("p0", "p1", "p2", "p3")
    rank = None


class ViewConsistentEngine:
    """Recovers the view sigma from the permuted image paths and answers with
    logits peaked at the VIEW-space class of a fixed original-space GT — so a
    correct pipeline must remap every view back to the same original class."""

    def __init__(self, gt_class: int, peak: float, enabled: bool = True, full_peak: float | None = None):
        self.gt_class = gt_class
        self.peak = peak
        self.full_peak = peak if full_peak is None else full_peak
        self.prune_cfg = PruneConfig(enabled=enabled)
        self.calls = {"pruned": 0, "full": 0}

    def prepare(self, images, caption):
        sigma = tuple(int(p[1]) for p in images)  # "p2" -> original input 2
        return ("prep", sigma)

    def keep_mask(self, prep, caption, cfg):
        return "keep-token"

    def forward_prepared(self, prep, keep=None):
        sigma = prep[1]
        c_view = perm.index_of(perm.apply_view(perm.rank_of_index(self.gt_class), sigma))
        peak = self.full_peak if keep is None else self.peak
        self.calls["full" if keep is None else "pruned"] += 1
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
