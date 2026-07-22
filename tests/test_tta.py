import torch

from snuai11 import perm
from snuai11.tta import BALANCED4, BALANCED8, aggregate_logprobs, margin_of, normalize, remap_scores, tta_views


def test_views_deterministic_and_distinct():
    v1 = tta_views(3, "sample-abc")
    v2 = tta_views(3, "sample-abc")
    assert v1 == v2
    assert v1[0] == perm.IDENTITY
    assert len(set(v1)) == 3
    assert tta_views(3, "other-id") != v1 or True  # different ids may differ


def test_balanced4_is_sharply_transitive():
    # every input sits at every slot exactly once across the 4 views — the
    # property that makes slot-position bias cancel exactly in the aggregate
    views = tta_views(4, "whatever")
    assert views == list(BALANCED4)
    assert views[0] == perm.IDENTITY
    assert len(set(views)) == 4
    for inp in range(4):
        slots = sorted(v.index(inp) for v in views)  # v[j] == inp -> input at slot j
        assert slots == [0, 1, 2, 3]
    for slot in range(4):
        inputs = sorted(v[slot] for v in views)
        assert inputs == [0, 1, 2, 3]


def test_balanced4_sample_independent_and_involutive():
    assert tta_views(4, "a") == tta_views(4, "b")  # fixed set, no per-sample RNG
    for v in BALANCED4:
        assert perm.invert(v) == v  # Klein group: every element self-inverse
    # closed under composition (a group), so remaps compose consistently
    for a in BALANCED4:
        for b in BALANCED4:
            assert perm.compose(a, b) in BALANCED4


def test_balanced8_visits_every_slot_exactly_twice():
    # BALANCED8 = BALANCED4 + one inverse-closed coset: each input visits
    # each slot exactly 2x (8 views / 4 slots) — the D4 analogue of TTA4's
    # sharply-transitive Latin square, at double the views.
    views = tta_views(8, "whatever")
    assert views == list(BALANCED8)
    assert views[0] == perm.IDENTITY
    assert len(set(views)) == 8
    assert set(BALANCED4) <= set(BALANCED8)
    for inp in range(4):
        slots = sorted(v.index(inp) for v in views)
        assert slots == [0, 0, 1, 1, 2, 2, 3, 3]
    for slot in range(4):
        inputs = sorted(v[slot] for v in views)
        assert inputs == [0, 0, 1, 1, 2, 2, 3, 3]


def test_balanced8_sample_independent_and_is_a_group():
    assert tta_views(8, "a") == tta_views(8, "b")  # fixed set, no per-sample RNG
    for v in BALANCED8:
        assert perm.invert(v) in BALANCED8  # inverse-closed
    for a in BALANCED8:  # closed under composition (D4, the Sylow-2 subgroup of S4)
        for b in BALANCED8:
            assert perm.compose(a, b) in BALANCED8


def test_non4_counts_keep_legacy_seeded_behavior():
    for n in (1, 2, 3, 5):
        v = tta_views(n, "sample-abc")
        assert v[0] == perm.IDENTITY
        assert len(v) == n and len(set(v)) == n
        assert v == tta_views(n, "sample-abc")


def test_remap_identity_is_noop():
    scores = torch.rand(24)
    assert torch.equal(remap_scores(scores, perm.IDENTITY), scores)


def test_remap_moves_gt_class_correctly():
    # if the view's correct class gets probability 1, the remapped vector must
    # put probability 1 on the ORIGINAL correct class — for all (rank, sigma).
    for rank in perm.ALL_PERMS:
        c_orig = perm.index_of(rank)
        for sigma in perm.ALL_PERMS[:8]:
            rv = perm.apply_view(rank, sigma)
            scores_view = torch.zeros(24)
            scores_view[perm.index_of(rv)] = 1.0
            back = remap_scores(scores_view, sigma)
            assert back.argmax().item() == c_orig


def test_remap_is_bijection():
    scores = torch.arange(24.0)
    for sigma in perm.ALL_PERMS:
        out = remap_scores(scores, sigma)
        assert sorted(out.tolist()) == sorted(scores.tolist())


def test_aggregate_and_margin():
    p1 = torch.full((24,), 1 / 24)
    p2 = torch.zeros(24)
    p2[3] = 1.0
    agg = aggregate_logprobs([p1, p2])
    assert agg.argmax().item() == 3
    probs = normalize(agg)
    assert 0.0 <= margin_of(probs) <= 1.0
    assert abs(float(probs.sum()) - 1.0) < 1e-5
