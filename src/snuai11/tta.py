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


def tta_views(n_views: int, sample_id: str) -> list[perm.Perm]:
    """Deterministic per-sample views: identity + (n-1) seeded shuffles."""
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
