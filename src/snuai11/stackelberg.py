"""Stackelberg two-time-scale optimization (ICML 2026, Zeng et al.).

The network splits into a body M (vision tower + language backbone via LoRA
adapters — the leader) and a head w (Score24Head — the follower). The
follower tracks its best response w*(M) by running on a FASTER time scale:
beta_k (head LR) >> alpha_k (body LR). This sharpens the local curvature of
the reduced objective Phi(M) = f(M, w*(M)) and accelerates early-phase
convergence.

Implementation: AdamW with two parameter groups + either a shared cosine
schedule (constant beta/alpha ratio — the paper's practical variant) or the
theoretical polynomial decays alpha_k ~ k^-a, beta_k ~ k^-b with a > b.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class StackelbergConfig:
    body_lr: float = 2e-4  # alpha_0 — LoRA adapters (leader, slow)
    head_lr: float = 1e-3  # beta_0  — Score24Head (follower, 5x fast; paper band 3-10x)
    body_weight_decay: float = 0.0
    # l2 on the HEAD is load-bearing: Assumption 3.1 (f strongly convex in w)
    # does not hold for plain softmax-CE — the penalty makes w*(M) unique and
    # the Stackelberg reduction valid (paper demos use 0.1).
    head_weight_decay: float = 0.1
    schedule: str = "cosine"  # cosine | poly | constant
    poly_body_exp: float = 0.6  # alpha_k = alpha_0/(k+1)^{3/5} (Thm 3.6, leader)
    poly_head_exp: float = 0.4  # beta_k  = beta_0 /(k+1)^{2/5} (follower decays slower)
    warmup_steps: int = 20


def build_param_groups(
    body_params: list[nn.Parameter],
    head_params: list[nn.Parameter],
    cfg: StackelbergConfig,
) -> list[dict]:
    body = [p for p in body_params if p.requires_grad]
    head = [p for p in head_params if p.requires_grad]
    if not head:
        raise ValueError("head has no trainable parameters")
    if not body:
        raise ValueError("body has no trainable parameters (LoRA not attached?)")
    overlap = {id(p) for p in body} & {id(p) for p in head}
    if overlap:
        raise ValueError("body/head parameter groups overlap")
    if cfg.head_lr < cfg.body_lr:
        raise ValueError("two-time-scale invariant violated: head_lr must be >= body_lr")
    return [
        {"params": body, "lr": cfg.body_lr, "weight_decay": cfg.body_weight_decay, "name": "body"},
        {"params": head, "lr": cfg.head_lr, "weight_decay": cfg.head_weight_decay, "name": "head"},
    ]


def build_optimizer(groups: list[dict], cfg: StackelbergConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(groups)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: StackelbergConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Per-group schedule; group order must match build_param_groups."""
    import math

    def warmup(step: int) -> float:
        if cfg.warmup_steps <= 0:
            return 1.0
        return min(1.0, (step + 1) / cfg.warmup_steps)

    def factor(step: int, exp: float) -> float:
        if cfg.schedule == "constant":
            return warmup(step)
        if cfg.schedule == "poly":
            return warmup(step) / float(step + 1) ** exp
        if cfg.schedule == "cosine":
            t = min(step, total_steps) / max(1, total_steps)
            return warmup(step) * 0.5 * (1.0 + math.cos(math.pi * t))
        raise ValueError(f"unknown schedule {cfg.schedule!r}")

    lambdas = [
        lambda s: factor(s, cfg.poly_body_exp),  # body (leader)
        lambda s: factor(s, cfg.poly_head_exp),  # head (follower)
    ]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)
