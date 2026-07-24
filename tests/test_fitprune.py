import pytest
import torch

from snuai11.fitprune import (
    PruneConfig,
    _minmax,
    boost_indices_for_image,
    combined_scores,
    cross_target_scores,
    keep_indices_for_image,
    merge_with_duplicates,
    motion_scores,
    objectness_scores,
    per_event_scores,
    select_diverse,
    select_mmr,
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
    idx = keep_indices_for_image([vis] * 4, 0, events, PruneConfig(keep_ratio=0.5))
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
    idx = keep_indices_for_image([vis] * 4, 0, events, PruneConfig(enabled=False))
    assert idx.tolist() == list(range(30))
    idx2 = keep_indices_for_image([vis] * 4, 0, events, PruneConfig(keep_ratio=1.0))
    assert idx2.tolist() == list(range(30))


def test_keep_at_least_one():
    vis = _rand(3, 8)
    events = [_rand(2, 8) for _ in range(4)]
    idx = keep_indices_for_image([vis] * 4, 0, events, PruneConfig(keep_ratio=0.01))
    assert idx.numel() >= 1


def _stuff_vs_things(d=16, seed=5):
    """The pruned-skier scenario from runs/prune_viz: a caption-aligned
    texture cluster ("water", 20 tokens, matches event 0's anchor), an
    off-caption background cluster ("sky", 18 tokens) and 2 small foreground
    objects far from the centroid but not caption-aligned. Visual centering
    de-means shared directions, so bg dominance must come from a cluster
    (not one global direction) to survive centering — as in real images."""
    e = torch.eye(d)
    g = torch.Generator().manual_seed(seed)
    water = e[3].repeat(20, 1) + torch.randn(20, d, generator=g) * 1e-2
    sky = e[8].repeat(18, 1) + torch.randn(18, d, generator=g) * 1e-2
    fg = torch.stack([6.0 * e[1], 6.0 * e[2]])  # tokens 38, 39
    vis = torch.cat([water, sky, fg])
    events = [e[3 + j].unsqueeze(0) for j in range(4)]  # event 0 == "water"
    return vis, events


def test_objectness_scores_rank_foreground_first():
    vis, _ = _stuff_vs_things()
    obj = objectness_scores(vis)
    assert obj.shape == (40,)
    assert set(obj.topk(2).indices.tolist()) == {38, 39}


def test_objectness_blend_rescues_pruned_foreground():
    # stuff-over-things regression: at 50% (budget 20) pure cosine fills the
    # whole budget with the 20 caption-aligned water tokens and cuts both
    # foreground objects; the objectness blend + MMR must keep them.
    vis, events = _stuff_vs_things()
    old = PruneConfig(keep_ratio=0.5, objectness_weight=0.0, mmr_lambda=0.0, diversity_frac=0.0)
    new = PruneConfig(keep_ratio=0.5)
    kept_old = set(keep_indices_for_image([vis] * 4, 0, events, old).tolist())
    kept_new = set(keep_indices_for_image([vis] * 4, 0, events, new).tolist())
    assert not {38, 39} & kept_old
    assert {38, 39} <= kept_new
    assert len(kept_new) == len(kept_old) == 20


def test_combined_scores_weight_zero_is_pure_cosine_ranking():
    vis = _rand(30, 16)
    events = [_rand(3, 16, seed=i) for i in range(4)]
    cfg0 = PruneConfig(objectness_weight=0.0)
    blended = combined_scores([vis] * 4, 0, events, cfg0)
    raw = cross_target_scores(vis, events, cfg0)
    assert torch.equal(torch.argsort(blended, stable=True), torch.argsort(raw, stable=True))


def test_select_mmr_counts_sorted_unique_and_budget():
    vis = _rand(100, 32)
    scores = torch.rand(100)
    idx = select_mmr(scores, vis, keep_ratio=0.5, mmr_lambda=0.3)
    assert idx.shape[0] == 50
    assert len(set(idx.tolist())) == 50
    assert idx.tolist() == sorted(idx.tolist())
    # keep everything when the budget covers all tokens
    assert select_mmr(scores, vis, 1.0, 0.3).tolist() == list(range(100))


def test_select_mmr_compresses_redundant_high_scorers():
    # 20 near-identical high scorers + 10 distinct medium scorers, budget 15:
    # top-k would take 15 duplicates; MMR must trade some for novel tokens.
    d = 8
    vis = torch.zeros(30, d)
    vis[:20, 0] = 1.0
    vis[:20] += _rand(20, d, seed=6) * 1e-3
    for i in range(10):
        vis[20 + i, i % (d - 1) + 1] = 1.0
    scores = torch.cat([torch.full((20,), 0.9), torch.full((10,), 0.75)])
    kept = select_mmr(scores, vis, keep_ratio=0.5, mmr_lambda=0.3)
    n_novel = sum(1 for i in kept.tolist() if i >= 20)
    assert n_novel >= 5
    # and the single best-scored token always survives
    assert int(torch.argmax(scores)) in set(kept.tolist())


def test_select_mmr_deterministic():
    vis = _rand(64, 16, seed=9)
    scores = torch.rand(64)
    a = select_mmr(scores, vis, 0.5, 0.3).tolist()
    b = select_mmr(scores, vis, 0.5, 0.3).tolist()
    assert a == b


def test_legacy_path_via_zero_flags_matches_old_selection():
    # objectness_weight=0 + mmr_lambda=0 must reproduce the previous
    # pipeline exactly (min-max on scores is monotonic -> same argsort).
    vis = _rand(50, 16, seed=11)
    events = [_rand(3, 16, seed=i) for i in range(4)]
    cfg = PruneConfig(keep_ratio=0.5, diversity_frac=0.2, objectness_weight=0.0, mmr_lambda=0.0)
    got = keep_indices_for_image([vis] * 4, 0, events, cfg)
    old = select_diverse(cross_target_scores(vis, events, cfg), vis, 0.5, 0.2)
    assert got.tolist() == old.tolist()


# ---- motion blend (2026-07-22) --------------------------------------------


def _four_images(n=40, d=16, seed=21):
    return [_rand(n, d, seed=seed + i) for i in range(4)]


def test_motion_weight_zero_is_bitwise_legacy_blend():
    # THE regression gate: motion_weight=0 (the default) must reproduce the
    # pre-motion combined_scores output bit-for-bit — siblings ignored.
    pe = _four_images()
    events = [_rand(3, 16, seed=i) for i in range(4)]
    for w_obj in (0.0, 0.3):
        cfg = PruneConfig(objectness_weight=w_obj)  # motion_weight defaults to 0.0
        got = combined_scores(pe, 1, events, cfg)
        vis = pe[1]
        cos = _minmax(cross_target_scores(vis, events, cfg))
        if w_obj == 0.0:
            expected = cos
        else:
            expected = (1.0 - w_obj) * cos + w_obj * _minmax(objectness_scores(vis))
        assert torch.equal(got, expected)


def test_motion_scores_matches_measure_script():
    # fitprune.motion_scores must equal the validated reference implementation
    # in scripts/measure_motion_signal.py on aligned grids.
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "measure_motion_signal",
        Path(__file__).resolve().parents[1] / "scripts" / "measure_motion_signal.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pe = _four_images()
    for i in range(4):
        assert torch.equal(motion_scores(pe, i), mod.motion_scores(pe, i))


def test_motion_scores_zero_for_static_high_for_moving():
    d = 16
    base = _rand(10, d, seed=31)
    pe = [base.clone() for _ in range(4)]
    for j in range(4):
        pe[j][7] = base[7] + torch.full((d,), float(j))  # token 7 drifts per frame
    mot = motion_scores(pe, 0)
    assert mot is not None and mot.shape == (10,)
    static = torch.cat([mot[:7], mot[8:]])
    assert torch.all(static < 1e-6)
    assert mot[7] > 1.0


def test_motion_grid_mismatch_falls_back_to_legacy():
    pe = _four_images()
    pe[2] = _rand(33, 16, seed=99)  # one image with a different token count
    events = [_rand(3, 16, seed=i) for i in range(4)]
    assert motion_scores(pe, 0) is None
    with_motion = PruneConfig(objectness_weight=0.3, motion_weight=0.5)
    without = PruneConfig(objectness_weight=0.3, motion_weight=0.0)
    assert torch.equal(
        combined_scores(pe, 0, events, with_motion),
        combined_scores(pe, 0, events, without),
    )
    kept_m = keep_indices_for_image(pe, 0, events, with_motion)
    kept_0 = keep_indices_for_image(pe, 0, events, without)
    assert kept_m.tolist() == kept_0.tolist()


def test_motion_weight_sum_validation():
    pe = _four_images()
    events = [_rand(3, 16, seed=i) for i in range(4)]
    for w_obj, w_mot in ((0.3, 0.3), (0.5, 0.5)):  # sum <= 1 is fine (1.0 -> w_cos = 0)
        out = combined_scores(pe, 0, events, PruneConfig(objectness_weight=w_obj, motion_weight=w_mot))
        assert torch.isfinite(out).all()
    with pytest.raises(ValueError):
        combined_scores(pe, 0, events, PruneConfig(objectness_weight=0.7, motion_weight=0.5))


def test_keep_mask_with_motion_weight(monkeypatch):
    # Integration: Engine.keep_mask over a synthetic 4-image Prepared with
    # motion_weight > 0 — correct shape, all text kept, 50% of each image.
    from snuai11.vlm import Engine, Prepared

    n_per, d, n_text = 12, 16, 5
    pe = _four_images(n=n_per, d=d)
    L = n_text + 4 * n_per
    positions = [torch.arange(n_text + i * n_per, n_text + (i + 1) * n_per) for i in range(4)]
    vmask = torch.zeros(1, L, dtype=torch.bool)
    for p in positions:
        vmask[0, p] = True
    prep = Prepared(
        inputs_embeds=torch.zeros(1, L, d),
        position_ids=torch.zeros(3, 1, L, dtype=torch.long),
        attention_mask=torch.ones(1, L, dtype=torch.long),
        visual_pos_masks=vmask,
        deepstack=[torch.zeros(4 * n_per, d) for _ in range(3)],
        per_image_embeds=pe,
        image_positions=positions,
    )
    events = [_rand(3, d, seed=i) for i in range(4)]
    monkeypatch.setattr(Engine, "device", torch.device("cpu"), raising=False)
    eng = object.__new__(Engine)
    eng.event_embeds = lambda caption: events
    cfg = PruneConfig(keep_ratio=0.5, motion_weight=0.3)
    keep = eng.keep_mask(prep, "cap", cfg)
    assert keep.shape == (1, L) and keep.dtype == torch.bool
    assert keep[0, :n_text].all()  # text tokens always kept
    kept_vis = int(keep[0][vmask[0]].sum())
    assert kept_vis == 4 * round(n_per * 0.5)  # 50% of each image's tokens


# ---- token boost (2026-07-23) ----------------------------------------------


def test_boost_frac_zero_is_empty():
    vis, events = _stuff_vs_things()
    dup = boost_indices_for_image([vis] * 4, 0, events, PruneConfig(keep_ratio=0.5))
    assert dup.numel() == 0


def test_boost_picks_highest_scoring_members_of_kept_set():
    # boost must never resurrect a pruned token, and must rank strictly by
    # the SAME combined_scores used for selection.
    vis, events = _stuff_vs_things()
    cfg = PruneConfig(keep_ratio=0.5, boost_frac=0.5)  # top half of the kept 20 -> 10
    kept = set(keep_indices_for_image([vis] * 4, 0, events, cfg).tolist())
    dup = boost_indices_for_image([vis] * 4, 0, events, cfg)
    assert dup.numel() == 10
    assert set(dup.tolist()) <= kept
    scores = combined_scores([vis] * 4, 0, events, cfg)
    ranked_kept = sorted(kept, key=lambda i: -scores[i].item())
    assert set(dup.tolist()) == set(ranked_kept[:10])


def test_boost_copies_multiplies_extra_occurrences():
    vis, events = _stuff_vs_things()
    cfg1 = PruneConfig(keep_ratio=0.5, boost_frac=0.1, boost_copies=1)
    cfg3 = PruneConfig(keep_ratio=0.5, boost_frac=0.1, boost_copies=3)
    dup1 = boost_indices_for_image([vis] * 4, 0, events, cfg1)
    dup3 = boost_indices_for_image([vis] * 4, 0, events, cfg3)
    assert dup3.numel() == 3 * dup1.numel()
    assert set(dup3.tolist()) == set(dup1.tolist())
    dup0 = boost_indices_for_image([vis] * 4, 0, events, PruneConfig(keep_ratio=0.5, boost_frac=0.1, boost_copies=0))
    assert dup0.numel() == 0  # 0 copies -> no boost even though boost_frac > 0


def test_boost_frac_rounds_up_to_at_least_one_and_caps_to_kept_size():
    vis = _rand(3, 8)
    events = [_rand(2, 8) for _ in range(4)]
    dup = boost_indices_for_image([vis] * 4, 0, events, PruneConfig(keep_ratio=0.5, boost_frac=0.01))
    assert dup.numel() >= 1
    cfg_full = PruneConfig(keep_ratio=0.5, boost_frac=1.0)
    kept = keep_indices_for_image([vis] * 4, 0, events, cfg_full)
    dup_full = boost_indices_for_image([vis] * 4, 0, events, cfg_full)
    assert dup_full.numel() == kept.numel()  # capped, never exceeds the kept set


def test_merge_with_duplicates_noop_when_extra_empty():
    base = torch.tensor([2, 5, 9, 20])
    out = merge_with_duplicates(base, base[:0])
    assert torch.equal(out, base)


def test_merge_with_duplicates_inserts_adjacent_and_preserves_order():
    base = torch.tensor([2, 5, 9, 20])
    extra = torch.tensor([5, 9, 5])  # position 5 duplicated twice, 9 once
    out = merge_with_duplicates(base, extra)
    assert out.tolist() == [2, 5, 5, 5, 9, 9, 20]


def test_select_for_forward_matches_legacy_keep_path():
    # Independent cross-check (legacy math re-derived here, not called from
    # production code): forward_prepared's boolean-keep branch and
    # vlm._select_for_forward(prep, keep[0].nonzero()) must select IDENTICAL
    # embeds/position_ids/attention_mask/visual_pos_masks/deepstack.
    from snuai11.vlm import Prepared, _select_for_forward

    L, d, V = 10, 4, 6
    inputs_embeds = torch.arange(L * d, dtype=torch.float32).reshape(1, L, d)
    position_ids = torch.arange(3 * L, dtype=torch.long).reshape(3, 1, L)
    attention_mask = torch.ones(1, L, dtype=torch.long)
    visual_pos_masks = torch.zeros(1, L, dtype=torch.bool)
    visual_pos_masks[0, [1, 2, 4, 5, 7, 8]] = True
    deepstack = [torch.arange(V * d, dtype=torch.float32).reshape(V, d) + k * 1000 for k in range(3)]
    prep = Prepared(
        inputs_embeds=inputs_embeds, position_ids=position_ids,
        attention_mask=attention_mask, visual_pos_masks=visual_pos_masks,
        deepstack=deepstack, per_image_embeds=[], image_positions=[],
    )

    keep = torch.zeros(1, L, dtype=torch.bool)
    keep[0, [0, 2, 4, 6, 8, 9]] = True  # keeps 3 of the 6 visual positions (2, 4, 8)
    idx = keep[0].nonzero().squeeze(1)

    embeds_old = prep.inputs_embeds[:, idx]
    pos_old = prep.position_ids[:, :, idx]
    attn_old = prep.attention_mask[:, idx]
    vmask_old = prep.visual_pos_masks[:, idx]
    kept_rows = keep[prep.visual_pos_masks]
    ds_old = [t[kept_rows] for t in prep.deepstack]

    embeds_new, pos_new, attn_new, vmask_new, ds_new = _select_for_forward(prep, idx)

    assert torch.equal(embeds_old, embeds_new)
    assert torch.equal(pos_old, pos_new)
    assert torch.equal(attn_old, attn_new)
    assert torch.equal(vmask_old, vmask_new)
    for a, b in zip(ds_old, ds_new):
        assert torch.equal(a, b)


def test_select_for_forward_duplicates_embeds_position_and_deepstack_together():
    from snuai11.vlm import Prepared, _select_for_forward

    L, d, V = 10, 4, 6
    inputs_embeds = torch.arange(L * d, dtype=torch.float32).reshape(1, L, d)
    position_ids = torch.arange(3 * L, dtype=torch.long).reshape(3, 1, L)
    attention_mask = torch.ones(1, L, dtype=torch.long)
    visual_pos_masks = torch.zeros(1, L, dtype=torch.bool)
    visual_pos_masks[0, [1, 2, 4, 5, 7, 8]] = True
    deepstack = [torch.arange(V * d, dtype=torch.float32).reshape(V, d)]
    prep = Prepared(
        inputs_embeds=inputs_embeds, position_ids=position_ids,
        attention_mask=attention_mask, visual_pos_masks=visual_pos_masks,
        deepstack=deepstack, per_image_embeds=[], image_positions=[],
    )
    # duplicate sequence position 4 (the 3rd visual token overall -> deepstack row 2)
    idx = torch.tensor([0, 2, 4, 4, 8, 9])
    embeds, pos, attn, vmask, ds = _select_for_forward(prep, idx)

    assert embeds.shape[1] == 6
    assert torch.equal(embeds[:, 2], embeds[:, 3])
    assert torch.equal(embeds[0, 2], inputs_embeds[0, 4])
    assert torch.equal(pos[:, :, 2], pos[:, :, 3])  # SAME m-rope coordinate, not a new one
    assert torch.equal(pos[:, :, 2], position_ids[:, :, 4])
    assert attn.shape == (1, 6)
    assert vmask[0].tolist() == [False, True, True, True, True, False]
    assert ds[0].shape[0] == 4  # one deepstack row per True vmask entry
    assert torch.equal(ds[0][1], deepstack[0][2])
    assert torch.equal(ds[0][2], deepstack[0][2])  # duplicated row


def test_scored_idx_matches_keep_mask_when_boost_frac_zero(monkeypatch):
    from snuai11.vlm import Engine, Prepared

    n_per, d, n_text = 12, 16, 5
    pe = _four_images(n=n_per, d=d)
    L = n_text + 4 * n_per
    positions = [torch.arange(n_text + i * n_per, n_text + (i + 1) * n_per) for i in range(4)]
    vmask = torch.zeros(1, L, dtype=torch.bool)
    for p in positions:
        vmask[0, p] = True
    prep = Prepared(
        inputs_embeds=torch.zeros(1, L, d),
        position_ids=torch.zeros(3, 1, L, dtype=torch.long),
        attention_mask=torch.ones(1, L, dtype=torch.long),
        visual_pos_masks=vmask,
        deepstack=[torch.zeros(4 * n_per, d) for _ in range(3)],
        per_image_embeds=pe,
        image_positions=positions,
    )
    events = [_rand(3, d, seed=i) for i in range(4)]
    monkeypatch.setattr(Engine, "device", torch.device("cpu"), raising=False)
    eng = object.__new__(Engine)
    eng.event_embeds = lambda caption: events
    cfg = PruneConfig(keep_ratio=0.5)  # boost_frac defaults to 0.0
    keep = eng.keep_mask(prep, "cap", cfg)
    got = eng.scored_idx(prep, "cap", cfg)
    assert torch.equal(got, keep[0].nonzero().squeeze(1))


def test_scored_idx_boost_frac_positive_duplicates_top_kept_tokens_per_image(monkeypatch):
    from snuai11.vlm import Engine, Prepared

    n_per, d, n_text = 12, 16, 5
    pe = _four_images(n=n_per, d=d)
    L = n_text + 4 * n_per
    positions = [torch.arange(n_text + i * n_per, n_text + (i + 1) * n_per) for i in range(4)]
    vmask = torch.zeros(1, L, dtype=torch.bool)
    for p in positions:
        vmask[0, p] = True
    prep = Prepared(
        inputs_embeds=torch.zeros(1, L, d),
        position_ids=torch.zeros(3, 1, L, dtype=torch.long),
        attention_mask=torch.ones(1, L, dtype=torch.long),
        visual_pos_masks=vmask,
        deepstack=[torch.zeros(4 * n_per, d) for _ in range(3)],
        per_image_embeds=pe,
        image_positions=positions,
    )
    events = [_rand(3, d, seed=i) for i in range(4)]
    monkeypatch.setattr(Engine, "device", torch.device("cpu"), raising=False)
    eng = object.__new__(Engine)
    eng.event_embeds = lambda caption: events
    cfg = PruneConfig(keep_ratio=0.5, boost_frac=0.5)  # half of each image's 6 kept -> 3 extra
    keep = eng.keep_mask(prep, "cap", cfg)
    base = keep[0].nonzero().squeeze(1)
    got = eng.scored_idx(prep, "cap", cfg)
    assert got.numel() == base.numel() + 4 * 3
    assert set(base.tolist()) <= set(got.tolist())  # every base position still present
    assert got[:n_text].tolist() == list(range(n_text))  # text prefix untouched
