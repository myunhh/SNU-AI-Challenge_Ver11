"""TTA3 — score a sample under shuffled views and aggregate in the original
class space (the technique gated at +3.81pp in earlier versions).

View v is a permutation sigma_v; the model scores 24 classes in view space;
perm.view_class_map(sigma_v) remaps those scores back to original space;
views are aggregated by mean log-probability (Laplace-smoothed).
"""

from __future__ import annotations

import math
import random

import torch

from . import perm


# Klein four-group — a sharply transitive set: across the 4 views every
# original input sits at every slot EXACTLY once (a Latin square over
# positions), so slot-position bias cancels exactly in the aggregate, not
# just in expectation like random extra views. All non-identity elements are
# fixed-point-free involutions and the set is closed under composition.
BALANCED4: tuple[perm.Perm, ...] = ((0, 1, 2, 3), (1, 0, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0))

# Balanced 8-view set (ported from Ver8, 2026-07-22 — same champion serving
# recipe: Ver8 DPO checkpoint-600 + TTA8 balanced, ~0.93 LB). BALANCED4 plus
# one inverse-closed coset (a 4-cycle pair + two involutions): every input
# visits every slot exactly TWICE across the 8 views, so BALANCED4's exact
# position-bias cancellation is preserved at double the views (unlike a
# generic "identity + 7 random shuffles" TTA8, which only cancels bias in
# expectation). The full set is the dihedral group D4 — the Sylow-2 subgroup
# of S4 (|S4|=24=8x3) — i.e. this is the next rung of the same algebraic
# ladder as BALANCED4 (Klein four-group -> D4); the ladder's only remaining
# rung is all of S4 (24 views). Ver8 measured TTA4->TTA8 balanced at
# +1.22pp real LB (../PROJECT_SUMMARY.md champion history).
BALANCED8: tuple[perm.Perm, ...] = BALANCED4 + (
    (1, 2, 3, 0), (0, 3, 2, 1), (3, 0, 1, 2), (2, 1, 0, 3))


def tta_views(n_views: int, sample_id: str) -> list[perm.Perm]:
    """Deterministic views for one sample.

    n_views == 4 -> the balanced Klein set (identity included), identical for
    every sample: exact position-bias cancellation (2026-07-17 default).
    n_views == 8 -> the balanced D4 set (BALANCED8 above, 2026-07-22): same
    exact-cancellation property at 2x the views, matching the Ver8 champion
    serving recipe (TTA8 balanced).
    Any other n -> identity + (n-1) per-sample seeded shuffles — byte-exact
    legacy behavior, kept so earlier pipelines (e.g. TTA3) stay reproducible.
    """
    if n_views == 4:
        return list(BALANCED4)
    if n_views == 8:
        return list(BALANCED8)
    views: list[perm.Perm] = [perm.IDENTITY]
    rng = random.Random(f"tta:{sample_id}")
    pool = [p for p in perm.ALL_PERMS if p != perm.IDENTITY]
    rng.shuffle(pool)
    views.extend(pool[: max(0, n_views - 1)])
    return views


def remap_scores(scores_view: torch.Tensor, sigma: perm.Perm) -> torch.Tensor:
    """scores_view[c_view] -> scores_orig[m[c_view]] (bijective)."""
    m = perm.view_class_map(sigma)
    out = torch.empty_like(scores_view)
    out[torch.tensor(m, device=scores_view.device)] = scores_view
    return out


def aggregate_logprobs(per_view_probs: list[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    """Mean of log(p + eps) across views -> aggregated 24-vector (log space)."""
    logs = [torch.log(p + eps) for p in per_view_probs]
    return torch.stack(logs, dim=0).mean(dim=0)


def margin_of(probs: torch.Tensor) -> float:
    top2 = torch.topk(probs, 2).values
    return float(top2[0] - top2[1])


def normalize(logits_or_logprobs: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits_or_logprobs.float(), dim=-1)


def entropy_of(probs: torch.Tensor) -> float:
    p = probs.clamp_min(1e-12)
    return float(-(p * p.log()).sum() / math.log(24.0))
