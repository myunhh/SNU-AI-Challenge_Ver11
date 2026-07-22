"""Cross-Targeted FitPrune — caption-conditioned visual token selection.

Pure tensor logic (no model dependency) so it is CPU-testable.

Pipeline per image:
  1. The caption is decomposed into 4 events C1..C4 (rule-based).
  2. Each of the image's N merged visual tokens (already projected into the
     LLM embedding space) is scored against ALL 4 events — the 4x4
     cross-target matrix over the sample's 4 images.
  3. Per-token importance = max over events (an image may match any event
     because inputs are shuffled; max-pooling keeps a token that matters to
     ANY event — FitPrune's distribution-preservation objective applied
     conservatively).
  4. Keep top (keep_ratio - diversity_frac*keep_ratio) tokens by importance,
     then fill the remaining diversity budget with tokens most DISSIMILAR to
     the kept set (LearnPruner's diversity tokens) so that background /
     off-caption cues survive.

Scoring uses cosine similarity between visual tokens and text-token
embeddings of the event's content words — a training-free surrogate for
FitPrune's attention statistics that needs no extra forward pass (the
visual embeddings are computed by the model anyway).

Selection objective (2026-07-16, post prune_viz forensics): pure
caption-cosine top-k has a stuff-over-things failure mode observed on real
samples — scene nouns in the caption (pool/ice/snow) match every patch of a
large homogeneous region, so hundreds of near-identical background tokens
fill the top-k while the small objects that actually discriminate temporal
order (ball, cap, hoop, skier) are cut. Two counter-terms, both on by
default and each ablatable to exactly the previous behavior:

  * objectness (cfg.objectness_weight): blend in the norm of each token's
    residual from the image centroid — foreground "things" deviate far from
    the image mean, homogeneous "stuff" sits near it. This is precisely the
    magnitude that _center_unit's normalization discards. Blended after
    per-image min-max so both terms share a [0, 1] scale; weight 0 restores
    the pure-cosine ranking (min-max is monotonic).
  * MMR selection (cfg.mmr_lambda): replace top-k + diversity-fill with one
    greedy maximal-marginal-relevance pass — pick argmax(score - lambda *
    relu(max cos-sim to already-kept)) until the budget is full. Redundant
    background patches collapse to a few representatives and the freed
    budget flows to lower-scored but novel tokens. relu: dissimilarity is
    not rewarded (that was the old farthest-point diversity fill's job,
    which MMR subsumes; diversity_frac is ignored when mmr_lambda > 0).
    lambda 0 falls back to the legacy top-k + diversity-fill path.

  * motion (cfg.motion_weight, 2026-07-22): blend in each token's mean raw
    residual norm against the SAME grid position of the sample's other 3
    frames — static background scores near 0, moving/changing foreground
    high. Raw norms, not cosine: a token that looks identical across frames
    must score ~0 regardless of direction. Validated before implementation
    (scripts/measure_motion_signal.py): spearman +0.693 against the
    prune_viz-verified objectness proxy but +0.011 against the caption
    cosine — foreground signal that the caption channel does not carry.
    Only defined when all 4 images share one token grid (~78% of train);
    otherwise the term is silently dropped for that image (exact legacy
    2-term blend — never resize/interpolate to force alignment). Weight 0
    reproduces the pre-motion pipeline bit-for-bit.

Default weights (objectness 0.3, MMR lambda 0.5) are structural choices,
not tuned values (Ver11 has no local holdout). The lambda floor is derived,
not arbitrary: both terms live on [0, 1], so in the worst case (foreground
at the cosine minimum -> blended score = w; a fully redundant background
cluster at the maximum -> deduped score = (1 - w) - lambda) rescue requires
lambda > 1 - 2w = 0.4; 0.5 adds margin for near-1 intra-cluster sims.
Ablate via the flags in a paired train-subset A/B (S3) if a measurement
slot opens. Keep-sets differ from all selections made before this change:
no A/B against pre-change runs.

Similarity geometry (2026-07-16): every cosine is computed after removing
each side's OWN population mean ("all-but-the-top"-style mean removal):
visual tokens are centered by the image's token centroid, text anchors by
the caption-level anchor mean mu_T. A shared dominant direction on either
side makes sim(v_i, t_j) approximately (per-token offset) + (per-event
scale) * const, so after pooling all 4 event maps collapse into one generic
saliency map (rank-1 degeneracy — observed on real samples in
runs/prune_viz, where the pre-fix E1..E4 heatmaps were visually identical).
Own-mean centering is the unique constant shift that zeroes the shared
component exactly; centering text by the VISUAL centroid (the first fix
attempt) leaves the modality-gap vector mu_T - mu_V inside every anchor and
merely relocates the degeneracy. Measured on the real Qwen3-VL-32B embed
table (400 train captions): mean pairwise cosine of the 4 event mean
directions goes +0.20 raw -> -0.33 after mu_T centering (~ the -1/3 maximum
contrast for 4 mean-zero vectors), i.e. events become mutually contrastive
anchors instead of near-parallel ones. The same centering is used for the
diversity dissimilarity space (farthest-point picks are meaningless in the
raw anisotropic space where all pairwise cosines sit in one narrow band).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PruneConfig:
    keep_ratio: float = 0.5  # fraction of visual tokens kept per image
    diversity_frac: float = 0.2  # legacy path only (mmr_lambda == 0)
    text_pool: str = "max"  # pool over text tokens of one event: max|mean
    event_pool: str = "max"  # pool over the 4 events: max|mean
    objectness_weight: float = 0.3  # blend weight of centroid-residual norm (0 = pure cosine)
    mmr_lambda: float = 0.5  # MMR redundancy penalty (0 = legacy top-k + diversity fill)
    motion_weight: float = 0.0  # cross-frame residual-norm blend weight (0 = pre-motion behavior)
    enabled: bool = True


def _pool(x: torch.Tensor, how: str, dim: int) -> torch.Tensor:
    if how == "max":
        return x.max(dim=dim).values
    if how == "mean":
        return x.mean(dim=dim)
    raise ValueError(f"unknown pool {how!r}")


def _center_unit(x: torch.Tensor) -> torch.Tensor:
    """Unit vectors of x after removing its own population mean."""
    return F.normalize(x - x.mean(dim=0, keepdim=True), dim=-1)


def _minmax(x: torch.Tensor) -> torch.Tensor:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-8)


@torch.no_grad()
def per_event_scores(
    visual: torch.Tensor,  # [N, D] merged visual tokens (LLM embed space)
    event_embeds: list[torch.Tensor],  # E x [T_j, D] text-token embeddings
    cfg: PruneConfig = PruneConfig(),
) -> torch.Tensor:
    """Per-event importance per visual token. [E, N]

    mu_T is the mean over ALL events' anchors, so per-event rows are only
    faithful to the selection path when the full event list is passed
    (visualizers must slice rows of this, not re-score one event alone).
    """
    v = _center_unit(visual.float())

    all_t = torch.cat([e.float() for e in event_embeds], dim=0)
    mu_t = all_t.mean(dim=0, keepdim=True)

    per_event = []
    for emb in event_embeds:
        t_f = emb.float()
        res = t_f - mu_t
        # Degenerate captions (one word repeated into all events) have zero
        # residuals — fall back to the raw anchor direction per row.
        ok = res.norm(dim=-1, keepdim=True) > 1e-6 * t_f.norm(dim=-1, keepdim=True)
        t = F.normalize(torch.where(ok, res, t_f), dim=-1)
        sim = v @ t.T  # [N, T_j]
        per_event.append(_pool(sim, cfg.text_pool, dim=1))  # [N]
    return torch.stack(per_event, dim=0)  # [E, N]


@torch.no_grad()
def cross_target_scores(
    visual: torch.Tensor,  # [N, D] merged visual tokens (LLM embed space)
    event_embeds: list[torch.Tensor],  # 4 x [T_j, D] text-token embeddings
    cfg: PruneConfig = PruneConfig(),
) -> torch.Tensor:
    """Importance score per visual token, pooled over the 4 events. [N]"""
    return _pool(per_event_scores(visual, event_embeds, cfg), cfg.event_pool, dim=0)


@torch.no_grad()
def objectness_scores(visual: torch.Tensor) -> torch.Tensor:
    """Norm of each token's residual from the image centroid. [N]

    Foreground objects deviate far from the image mean; homogeneous
    background sits near it. Raw scale — callers min-max before blending.
    """
    v_f = visual.float()
    return (v_f - v_f.mean(dim=0, keepdim=True)).norm(dim=-1)


@torch.no_grad()
def motion_scores(per_image_embeds: list[torch.Tensor], img_i: int) -> torch.Tensor | None:
    """Mean cross-frame residual norm per token of image img_i. [N] or None.

    motion[p] = mean_{j != i} ||v_i[p] - v_j[p]|| — a token whose grid
    position looks the same in the sample's other 3 frames (static
    background) scores near 0, moving/changing foreground scores high. Raw
    residual norms, NOT cosine: visually identical tokens must score ~0
    regardless of direction (scripts/measure_motion_signal.py is the
    validated reference this mirrors).

    Returns None when the 4 images' token counts differ (different native
    resolutions, ~22% of train): positional alignment is meaningless there.
    Callers must then silently drop the motion term (legacy cos+objectness
    blend) — never crash, never resize/interpolate to force alignment.
    """
    n = per_image_embeds[img_i].shape[0]
    if any(t.shape[0] != n for t in per_image_embeds):
        return None
    v_i = per_image_embeds[img_i].float()
    others = [per_image_embeds[j].float() for j in range(len(per_image_embeds)) if j != img_i]
    diffs = torch.stack([(v_i - o).norm(dim=-1) for o in others], dim=0)  # [3, N]
    return diffs.mean(dim=0)  # [N]


@torch.no_grad()
def combined_scores(
    per_image_embeds: list[torch.Tensor],
    img_i: int,
    event_embeds: list[torch.Tensor],
    cfg: PruneConfig = PruneConfig(),
) -> torch.Tensor:
    """Selection score for image img_i: caption-cosine blended with
    objectness and cross-frame motion, each per-image min-maxed to [0, 1].
    objectness_weight/motion_weight are the shares taken from the cosine
    term (w_cos = 1 - w_obj - w_mot); each weight at 0 removes its term
    exactly, so every ablation is exact — motion_weight=0 reproduces the
    pre-motion output bit-for-bit, and a grid-mismatched sample under
    motion_weight>0 takes the very same legacy code path. [N]"""
    w_obj, w_mot = cfg.objectness_weight, cfg.motion_weight
    if w_obj + w_mot > 1.0:
        raise ValueError(f"objectness_weight + motion_weight = {w_obj + w_mot:g} > 1")
    visual = per_image_embeds[img_i]
    cos = _minmax(cross_target_scores(visual, event_embeds, cfg))
    mot = motion_scores(per_image_embeds, img_i) if w_mot > 0.0 else None
    if mot is None:  # motion off — or unequal token grids: exact legacy blend
        if w_obj <= 0.0:
            return cos
        return (1.0 - w_obj) * cos + w_obj * _minmax(objectness_scores(visual))
    out = (1.0 - w_obj - w_mot) * cos + w_mot * _minmax(mot)
    if w_obj > 0.0:
        out = out + w_obj * _minmax(objectness_scores(visual))
    return out


@torch.no_grad()
def select_mmr(
    scores: torch.Tensor,  # [N]
    visual: torch.Tensor,  # [N, D]
    keep_ratio: float,
    mmr_lambda: float,
) -> torch.Tensor:
    """Indices (sorted ascending — spatial order preserved) of kept tokens.

    Greedy maximal-marginal-relevance over the WHOLE keep budget: pick
    argmax(score - lambda * relu(max cos-sim to kept)) each round.
    Similarity lives in the centered space (see module docstring); relu so
    dissimilarity is never rewarded, only redundancy penalized.
    """
    n = scores.shape[0]
    keep = max(1, min(n, round(n * keep_ratio)))
    if keep >= n:
        return torch.arange(n, device=scores.device)

    v = _center_unit(visual.float())
    first = torch.argmax(scores)  # argmax → first occurrence: deterministic
    kept = first.view(1)
    taken = torch.zeros(n, dtype=torch.bool, device=scores.device)
    taken[first] = True
    max_sim = v @ v[first]  # [N]
    for _ in range(keep - 1):
        adj = scores - mmr_lambda * max_sim.clamp_min(0.0)
        adj = adj.masked_fill(taken, float("-inf"))
        pick = torch.argmax(adj)
        kept = torch.cat([kept, pick.view(1)])
        taken[pick] = True
        max_sim = torch.maximum(max_sim, v @ v[pick])
    return torch.sort(kept).values


@torch.no_grad()
def select_diverse(
    scores: torch.Tensor,  # [N]
    visual: torch.Tensor,  # [N, D]
    keep_ratio: float,
    diversity_frac: float,
) -> torch.Tensor:
    """Indices (sorted ascending — spatial order preserved) of kept tokens.

    top-k by score for (1 - diversity_frac) of the budget, then greedily add
    the token with the LOWEST max-cosine-similarity to the kept set until the
    budget is full (farthest-point style, LearnPruner diversity tokens).
    Dissimilarity lives in the centered space (see module docstring).
    """
    n = scores.shape[0]
    keep = max(1, min(n, round(n * keep_ratio)))
    if keep >= n:
        return torch.arange(n, device=scores.device)
    n_div = int(round(keep * diversity_frac))
    n_top = keep - n_div

    order = torch.argsort(scores, descending=True, stable=True)
    kept = order[:n_top] if n_top > 0 else order[:0]

    if n_div > 0:
        v = _center_unit(visual.float())
        remaining = order[n_top:]
        if kept.numel() == 0:
            # degenerate: seed with the single best-scored token
            kept = order[:1]
            remaining = order[1:]
            n_div_left = n_div - 1
        else:
            n_div_left = n_div
        # max cosine similarity of each remaining token to the kept set
        max_sim = (v[remaining] @ v[kept].T).max(dim=1).values
        for _ in range(n_div_left):
            if remaining.numel() == 0:
                break
            pick = torch.argmin(max_sim)  # tensor index — no host sync
            chosen = remaining[pick]
            kept = torch.cat([kept, chosen.view(1)])
            mask = torch.ones_like(remaining, dtype=torch.bool)
            mask[pick] = False
            remaining = remaining[mask]
            if remaining.numel():
                sim_new = (v[remaining] @ v[chosen].view(1, -1).T).squeeze(1)
                max_sim = torch.maximum(max_sim[mask], sim_new)

    return torch.sort(kept).values


@torch.no_grad()
def keep_indices_for_image(
    per_image_embeds: list[torch.Tensor],
    img_i: int,
    event_embeds: list[torch.Tensor],
    cfg: PruneConfig = PruneConfig(),
) -> torch.Tensor:
    """End-to-end selection for image img_i — sibling embeds are the motion
    context (only read when motion_weight > 0). Returns ascending indices."""
    visual = per_image_embeds[img_i]
    if not cfg.enabled or cfg.keep_ratio >= 1.0:
        return torch.arange(visual.shape[0], device=visual.device)
    scores = combined_scores(per_image_embeds, img_i, event_embeds, cfg)
    if cfg.mmr_lambda > 0.0:
        return select_mmr(scores, visual, cfg.keep_ratio, cfg.mmr_lambda)
    return select_diverse(scores, visual, cfg.keep_ratio, cfg.diversity_frac)
