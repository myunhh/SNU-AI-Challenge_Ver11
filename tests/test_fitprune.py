import torch

from snuai11.fitprune import (
    PruneConfig,
    cross_target_scores,
    keep_indices_for_image,
    select_diverse,
)


def _rand(n, d, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, d, generator=g)


def test_scores_shape_and_alignment():
    d = 32
    text = torch.zeros(3, d)
    text[:, 0] = 1.0  # events point along dim 0
    vis = torch.zeros(10, d)
    vis[:, 1] = 1.0
    vis[4, :] = 0.0
    vis[4, 0] = 1.0  # token 4 aligned with the text
    scores = cross_target_scores(vis, [text, text, text, text])
    assert scores.shape == (10,)
    assert scores.argmax().item() == 4


def test_event_max_pool_keeps_single_event_cue():
    # a token that matches ONLY event 3 must outrank a token matching nothing
    d = 16
    e_other = torch.randn(2, d)
    e3 = torch.zeros(2, d)
    e3[:, 5] = 1.0
    vis = torch.randn(8, d) * 0.01
    vis[2, 5] = 10.0  # matches event 3 only
    scores = cross_target_scores(vis, [e_other, e_other, e_other, e3])
    assert scores.argmax().item() == 2


def test_select_diverse_counts_sorted_unique():
    vis = _rand(100, 32)
    scores = torch.rand(100)
    idx = select_diverse(scores, vis, keep_ratio=0.5, diversity_frac=0.2)
    assert idx.shape[0] == 50
    assert len(set(idx.tolist())) == 50
    assert idx.tolist() == sorted(idx.tolist())


def test_top_scores_survive():
    vis = _rand(60, 16)
    scores = torch.rand(60)
    idx = set(select_diverse(scores, vis, 0.5, 0.2).tolist())
    top = torch.argsort(scores, descending=True)[:24].tolist()  # 80% of budget=30
    assert set(top).issubset(idx)


def test_diversity_prefers_dissimilar_tokens():
    # 20 near-identical high-score tokens + 1 orthogonal low-score outlier:
    # with diversity the outlier must be kept, without it must not.
    d = 8
    vis = torch.zeros(21, d)
    vis[:20, 0] = 1.0
    vis[:20] += torch.randn(20, d) * 1e-3
    vis[20, 1] = 1.0  # the outlier
    scores = torch.linspace(1.0, 0.5, 21)  # outlier has the lowest score
    kept_no_div = set(select_diverse(scores, vis, keep_ratio=0.5, diversity_frac=0.0).tolist())
    kept_div = set(select_diverse(scores, vis, keep_ratio=0.5, diversity_frac=0.3).tolist())
    assert 20 not in kept_no_div
    assert 20 in kept_div


def test_disabled_keeps_everything():
    vis = _rand(30, 8)
    events = [_rand(3, 8, seed=i) for i in range(4)]
    idx = keep_indices_for_image(vis, events, PruneConfig(enabled=False))
    assert idx.tolist() == list(range(30))
    idx2 = keep_indices_for_image(vis, events, PruneConfig(keep_ratio=1.0))
    assert idx2.tolist() == list(range(30))


def test_keep_at_least_one():
    vis = _rand(3, 8)
    events = [_rand(2, 8) for _ in range(4)]
    idx = keep_indices_for_image(vis, events, PruneConfig(keep_ratio=0.01))
    assert idx.numel() >= 1
