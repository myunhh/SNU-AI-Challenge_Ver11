import torch

from snuai11.fitprune import (
    PruneConfig,
    cross_target_scores,
    keep_indices_for_image,
    per_event_scores,
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
    e_other = _rand(2, d, seed=7)
    e_other[:, 5] = 0.0  # the cue dim belongs to event 3 alone
    e3 = torch.zeros(2, d)
    e3[:, 5] = 1.0
    vis = _rand(8, d, seed=8) * 0.01
    vis[2, 5] = 10.0  # matches event 3 only
    scores = cross_target_scores(vis, [e_other, e_other, e_other, e3])
    assert scores.argmax().item() == 2


def test_per_event_scores_shape_and_pooling():
    vis = _rand(12, 16)
    events = [_rand(3, 16, seed=i) for i in range(4)]
    per_event = per_event_scores(vis, events)
    assert per_event.shape == (4, 12)
    pooled = cross_target_scores(vis, events)
    assert torch.allclose(pooled, per_event.max(dim=0).values)


def test_cross_targeting_survives_shared_dominant_directions():
    # Rank-1 degeneracy regression (runs/prune_viz, 2026-07-16): a dominant
    # direction shared by all visual tokens plus one shared by all text
    # anchors (anisotropy / modality gap) must not collapse the 4 per-event
    # maps into one. With own-mean centering each event must still point at
    # its own visual token.
    d = 32
    bias_v = torch.zeros(d)
    bias_v[8] = 50.0
    bias_t = torch.zeros(d)
    bias_t[9] = 50.0

    events = []
    for j in range(4):
        e = bias_t.clone().unsqueeze(0)  # [1, d]
        e[0, j] += 1.0  # event j's distinctive content direction
        events.append(e)

    vis = bias_v.repeat(8, 1)
    for i in range(4):
        vis[i, i] += 1.0  # token i carries event i's cue
    vis[4:, 12] += 0.1  # filler background tokens

    per_event = per_event_scores(vis, events)
    assert per_event.argmax(dim=1).tolist() == [0, 1, 2, 3]
    # and the pooled score must rank all 4 cue tokens above the fillers
    pooled = cross_target_scores(vis, events)
    assert set(pooled.topk(4).indices.tolist()) == {0, 1, 2, 3}


def test_degenerate_repeated_anchor_caption_is_finite():
    # decompose_caption("Fire") duplicates one word into all 4 events -> all
    # anchors identical -> zero residuals; the raw-direction fallback must
    # kick in (no NaN, still a valid selection).
    d = 16
    anchor = torch.zeros(1, d)
    anchor[0, 3] = 1.0
    events = [anchor.clone() for _ in range(4)]
    vis = _rand(30, d)
    scores = cross_target_scores(vis, events)
    assert torch.isfinite(scores).all()
    idx = keep_indices_for_image(vis, events, PruneConfig(keep_ratio=0.5))
    assert idx.shape[0] == 15
    assert len(set(idx.tolist())) == 15


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
    vis[:20] += _rand(20, d, seed=3) * 1e-3
    vis[20, 1] = 1.0  # the outlier
    scores = torch.linspace(1.0, 0.5, 21)  # outlier has the lowest score
    kept_no_div = set(select_diverse(scores, vis, keep_ratio=0.5, diversity_frac=0.0).tolist())
    kept_div = set(select_diverse(scores, vis, keep_ratio=0.5, diversity_frac=0.3).tolist())
    assert 20 not in kept_no_div
    assert 20 in kept_div


def test_diversity_immune_to_shared_dominant_direction():
    # Same outlier setup drowned by a huge direction shared by ALL tokens:
    # raw cosines all sit in one narrow band (anisotropy), so only the
    # centered dissimilarity space can still find the true outlier.
    d = 8
    vis = torch.zeros(21, d)
    vis[:20, 0] = 1.0
    vis[:20] += _rand(20, d, seed=4) * 1e-3
    vis[20, 1] = 1.0
    vis[:, 7] += 100.0  # shared dominant direction
    scores = torch.linspace(1.0, 0.5, 21)
    kept = set(select_diverse(scores, vis, keep_ratio=0.5, diversity_frac=0.3).tolist())
    assert 20 in kept


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
