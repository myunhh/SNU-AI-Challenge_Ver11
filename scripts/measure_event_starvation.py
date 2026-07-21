#!/usr/bin/env python
"""Test the "split caption into units, prune per-unit with a guaranteed
quota, then merge/union" idea against the CURRENT shipped selection
("merge scores across units via max-pool, then select once globally").

Reuses the embeds_cache.pt from measure_frame_contrast.py — zero extra
GPU/vision-tower cost, pure CPU tensor ops.

Question: does global top-k over the max-pooled cross-event score ever
DISCARD an event's own single best-matching token because other events'
tokens crowd the budget? If that "starvation" rate is high, a per-event
quota/union scheme has a real problem to fix; if it's already ~0%, the
extra mechanism would just reproduce current behavior at added complexity.

Usage: python scripts/measure_event_starvation.py
  (expects runs/frame_contrast/embeds_cache.pt to already exist)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from snuai11.fitprune import PruneConfig, keep_indices_for_image, per_event_scores  # noqa: E402


def best_token_survival(records, cfg: PruneConfig, desc: str):
    discarded = total = starved_asym = 0
    per_event_discard = [0, 0, 0, 0]
    per_event_total = [0, 0, 0, 0]
    hist = [0, 0, 0, 0, 0]  # count of images with 0..4 events' best token discarded
    for r in tqdm(records, desc=desc, unit="샘플", mininterval=2.0):
        for img_i in range(4):
            visual = r["per_image_embeds"][img_i]
            pe = per_event_scores(visual, r["event_embeds"], cfg)  # [4, N]
            keep = set(keep_indices_for_image(visual, r["event_embeds"], cfg).tolist())
            discards_here = 0
            for e in range(4):
                best = int(pe[e].argmax())
                total += 1
                per_event_total[e] += 1
                if best not in keep:
                    discarded += 1
                    per_event_discard[e] += 1
                    discards_here += 1
            hist[discards_here] += 1
            if 0 < discards_here < 4:
                starved_asym += 1
    return discarded, total, per_event_discard, per_event_total, starved_asym, hist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="레코드 수 제한(기본: 캐시 전체)")
    args = ap.parse_args()

    cache_path = REPO / "runs/frame_contrast/embeds_cache.pt"
    t0 = time.monotonic()
    records = torch.load(cache_path, weights_only=False)
    print(f"[starvation] 캐시 로드 {time.monotonic()-t0:.1f}s ({len(records)}건)")
    if args.n:
        records = records[: args.n]
    n_images = len(records) * 4
    print(f"[starvation] {len(records)}건 샘플 x 4이미지 = {n_images}건 이미지 검사\n")

    configs = [
        ("pure-cosine (objectness=0, MMR=0, div=0) — 07-16 수정 이전 legacy",
         PruneConfig(objectness_weight=0.0, mmr_lambda=0.0, diversity_frac=0.0)),
        ("production (objectness=0.3, MMR λ=0.5) — 실제 배선 값", PruneConfig()),
    ]
    for name, cfg in configs:
        t1 = time.monotonic()
        d, t, ped, pet, starved, hist = best_token_survival(records, cfg, desc=name[:20])
        print(f"=== {name}  [{time.monotonic()-t1:.1f}s] ===")
        print(f"  이벤트의 '자기 최고토큰'이 keep-set에서 잘린 비율: {d}/{t} = {d/t:.1%}")
        print(f"  이벤트별: " + ", ".join(f"E{e+1} {ped[e]}/{pet[e]}={ped[e]/pet[e]:.1%}" for e in range(4)))
        print(f"  이미지당 잘린 이벤트 수 분포(0~4개): {hist}  (합={sum(hist)})")
        print(f"  편중 사례(4개 중 일부만 잘림 — 특정 이벤트만 밀려난 신호): {starved}/{n_images} = {starved/n_images:.1%}\n")


if __name__ == "__main__":
    main()
