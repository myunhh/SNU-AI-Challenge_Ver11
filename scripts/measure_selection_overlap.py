#!/usr/bin/env python
"""Zero-additional-compute pre-check for "selection-marginalized TTA"
(average stage-1 logits across a few different prune-CONFIG choices,
instead of searching for one best config) — before spending any real
train-subset eval on it, check whether the candidate configs' keep-sets
actually differ enough to matter. Reuses embeds_cache.pt (pure CPU).
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from snuai11.fitprune import PruneConfig, keep_indices_for_image  # noqa: E402

CONFIGS = {
    "production (w0.3/λ0.5)": PruneConfig(),
    "legacy (w0/λ0, top-k+div0.2)": PruneConfig(objectness_weight=0.0, mmr_lambda=0.0, diversity_frac=0.2),
    "high-MMR (w0.3/λ0.9)": PruneConfig(objectness_weight=0.3, mmr_lambda=0.9),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()
    records = torch.load(REPO / "runs/frame_contrast/embeds_cache.pt", weights_only=False)
    if args.n:
        records = records[: args.n]
    names = list(CONFIGS.keys())
    overlaps = {(a, b): [] for a, b in combinations(names, 2)}

    for r in tqdm(records, desc="overlap", unit="샘플", mininterval=2.0):
        for img_i in range(4):
            visual = r["per_image_embeds"][img_i]
            keep = {}
            for name, cfg in CONFIGS.items():
                idx = keep_indices_for_image(visual, r["event_embeds"], cfg)
                keep[name] = set(idx.tolist())
            n = visual.shape[0]
            for a, b in combinations(names, 2):
                inter = len(keep[a] & keep[b])
                union = len(keep[a] | keep[b])
                overlaps[(a, b)].append(inter / union if union else 1.0)

    print(f"\n=== keep-set 중복률(Jaccard) — {len(records)}건 x 4이미지 ===")
    for (a, b), xs in overlaps.items():
        m = sum(xs) / len(xs)
        print(f"  {a:28s} vs {b:28s}  mean Jaccard={m:.3f}  (min={min(xs):.3f}, max={max(xs):.3f})")


if __name__ == "__main__":
    main()
