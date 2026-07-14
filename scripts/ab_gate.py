#!/usr/bin/env python
"""Paired-bootstrap A/B gate over two predict runs (progress.jsonl with truth).

Adoption rule (project convention): dEM >= +2pp AND 95% CI lower bound > 0.
Pairwise (1-KT/6) is reported alongside as the LB proxy.

  python scripts/ab_gate.py runs/cal_A runs/cal_B --name fitprune --out runs/gate_fitprune.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snuai11 import perm  # noqa: E402


def load_run(path: Path) -> dict[str, dict]:
    f = path / "progress.jsonl" if path.is_dir() else path
    out = {}
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if "truth" in r:
            out[r["id"]] = r
    return out


def metrics(rec: dict) -> tuple[float, float]:
    pred, truth = tuple(rec["rank"]), tuple(rec["truth"])
    return float(pred == truth), perm.pairwise_score(pred, truth)


def bootstrap(deltas: list[float], iters: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(iters):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a", type=Path, help="baseline")
    ap.add_argument("run_b", type=Path, help="candidate")
    ap.add_argument("--name", default="gate")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    a, b = load_run(args.run_a), load_run(args.run_b)
    ids = sorted(set(a) & set(b))
    if len(ids) < 100:
        raise SystemExit(f"only {len(ids)} paired labeled samples — not enough")

    report = {"name": args.name, "n": len(ids), "baseline": str(args.run_a), "candidate": str(args.run_b)}
    for metric_i, metric_name in ((0, "em"), (1, "pairwise")):
        va = [metrics(a[i])[metric_i] for i in ids]
        vb = [metrics(b[i])[metric_i] for i in ids]
        deltas = [y - x for x, y in zip(va, vb)]
        lo, hi = bootstrap(deltas)
        d = sum(deltas) / len(deltas)
        report[metric_name] = {
            "baseline": round(sum(va) / len(va), 4),
            "candidate": round(sum(vb) / len(vb), 4),
            "delta_pp": round(100 * d, 2),
            "ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
            "adopt": bool(d >= 0.02 and lo > 0) if metric_name == "em" else bool(lo > 0),
        }
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
