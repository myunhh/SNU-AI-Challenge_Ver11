import pytest
import torch
from torch import nn

from snuai11.stackelberg import (
    StackelbergConfig,
    build_optimizer,
    build_param_groups,
    build_scheduler,
)


def _setup(schedule="cosine"):
    cfg = StackelbergConfig(schedule=schedule, warmup_steps=2)
    body = nn.Linear(4, 4)
    head = nn.Linear(4, 24)
    groups = build_param_groups(list(body.parameters()), list(head.parameters()), cfg)
    opt = build_optimizer(groups, cfg)
    return cfg, opt


def test_groups_have_asymmetric_lrs():
    cfg, opt = _setup()
    lrs = {g["name"]: g["lr"] for g in opt.param_groups}
    assert lrs["head"] == pytest.approx(cfg.head_lr)
    assert lrs["body"] == pytest.approx(cfg.body_lr)
    assert lrs["head"] / lrs["body"] == pytest.approx(5.0)
    wds = {g["name"]: g["weight_decay"] for g in opt.param_groups}
    assert wds["head"] > 0.0  # strong-convexity regularizer on the follower
    assert wds["body"] == pytest.approx(cfg.body_weight_decay)


def test_inverted_lrs_rejected():
    from snuai11.stackelberg import StackelbergConfig, build_param_groups

    cfg = StackelbergConfig(body_lr=1e-3, head_lr=1e-4)
    body, head = nn.Linear(4, 4), nn.Linear(4, 24)
    with pytest.raises(ValueError):
        build_param_groups(list(body.parameters()), list(head.parameters()), cfg)


def test_overlap_rejected():
    cfg = StackelbergConfig()
    layer = nn.Linear(4, 4)
    with pytest.raises(ValueError):
        build_param_groups(list(layer.parameters()), list(layer.parameters()), cfg)


def test_frozen_body_rejected():
    cfg = StackelbergConfig()
    body = nn.Linear(4, 4)
    for p in body.parameters():
        p.requires_grad_(False)
    head = nn.Linear(4, 24)
    with pytest.raises(ValueError):
        build_param_groups(list(body.parameters()), list(head.parameters()), cfg)


@pytest.mark.parametrize("schedule", ["cosine", "poly", "constant"])
def test_schedules_keep_ratio_or_widen(schedule):
    cfg, opt = _setup(schedule)
    sched = build_scheduler(opt, cfg, total_steps=100)
    ratios = []
    for _ in range(50):
        opt.step()
        sched.step()
        lrs = {g["name"]: g["lr"] for g in opt.param_groups}
        assert lrs["head"] > lrs["body"] > 0
        ratios.append(lrs["head"] / lrs["body"])
    if schedule in ("cosine", "constant"):
        assert all(abs(r - 5.0) < 1e-6 for r in ratios)
    else:  # poly: body decays FASTER than head -> ratio grows (two time scales)
        assert ratios[-1] > ratios[0]


def test_poly_decays():
    cfg, opt = _setup("poly")
    sched = build_scheduler(opt, cfg, total_steps=100)
    first = None
    for step in range(30):
        opt.step()
        sched.step()
        if step == 5:
            first = [g["lr"] for g in opt.param_groups]
    last = [g["lr"] for g in opt.param_groups]
    assert all(l < f for l, f in zip(last, first))
