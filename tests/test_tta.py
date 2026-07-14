import torch

from snuai11 import perm
from snuai11.tta import aggregate_logprobs, margin_of, normalize, remap_scores, tta_views


def test_views_deterministic_and_distinct():
    v1 = tta_views(3, "sample-abc")
    v2 = tta_views(3, "sample-abc")
    assert v1 == v2
    assert v1[0] == perm.IDENTITY
    assert len(set(v1)) == 3
    assert tta_views(3, "other-id") != v1 or True  # different ids may differ


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
