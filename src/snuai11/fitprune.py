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
    diversity_frac: float = 0.2  # fraction of the KEPT budget from diversity
    text_pool: str = "max"  # pool over text tokens of one event: max|mean
    event_pool: str = "max"  # pool over the 4 events: max|mean
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
    visual: torch.Tensor,
    event_embeds: list[torch.Tensor],
    cfg: PruneConfig = PruneConfig(),
) -> torch.Tensor:
    """End-to-end per-image selection. Returns ascending token indices."""
    if not cfg.enabled or cfg.keep_ratio >= 1.0:
        return torch.arange(visual.shape[0], device=visual.device)
    scores = cross_target_scores(visual, event_embeds, cfg)
    return select_diverse(scores, visual, cfg.keep_ratio, cfg.diversity_frac)
