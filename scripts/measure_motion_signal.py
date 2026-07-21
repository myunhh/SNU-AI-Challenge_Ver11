#!/usr/bin/env python
"""Cheap, zero-extra-GPU-cost validation of the "frame-difference / motion
prior" pruning idea (Agent A #2, proposed 2026-07-21, never actually
tested) — before committing to a Ver11 FitPrune code change + overnight
retrain, check whether the signal exists at all.

Reuses runs/frame_contrast/embeds_cache.pt (already-extracted vision-tower
embeddings, no model load needed here). For each sample where all 4 images
share the same token grid (same native resolution -> direct positional
alignment valid, ~78% of the cache), per token position p:

  motion[img_i, p] = mean_j!=i || v_i[p] - v_j[p] ||   (raw residual norm,
                      NOT cosine -- a token that's visually identical across
                      frames should score near 0 regardless of direction)

Compared against:
  - objectness_scores (07-16 fix's residual-from-centroid norm) -- already
    validated via prune_viz as a real foreground/background discriminator,
    used here as a cheap ground-truth PROXY (no re-rendering needed).
  - cross_target_scores (the existing caption-cosine surrogate) -- to see
    whether motion is redundant with it or catches something different.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from snuai11.fitprune import PruneConfig, cross_target_scores, objectness_scores  # noqa: E402


def motion_scores(per_image_embeds: list[torch.Tensor], img_i: int) -> torch.Tensor:
    v_i = per_image_embeds[img_i].float()
    others = [per_image_embeds[j].float() for j in range(4) if j != img_i]
    diffs = torch.stack([(v_i - o).norm(dim=-1) for o in others], dim=0)  # [3, N]
    return diffs.mean(dim=0)  # [N]


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    ar = a.argsort().argsort().float()
    br = b.argsort().argsort().float()
    return torch.corrcoef(torch.stack([ar, br]))[0, 1].item()


def main() -> None:
    records = torch.load(REPO / "runs/frame_contrast/embeds_cache.pt", weights_only=False)
    cfg = PruneConfig()

    aligned = [r for r in records if len({t.shape[0] for t in r["per_image_embeds"]}) == 1]
    print(f"[motion] {len(aligned)}/{len(records)}건 4장 토큰수 일치(정렬 가능) 표본으로 진행\n")

    sp_obj, sp_cos = [], []
    for r in aligned:
        pe = r["per_image_embeds"]
        event_embeds = r["event_embeds"]
        for img_i in range(4):
            mot = motion_scores(pe, img_i)
            obj = objectness_scores(pe[img_i])
            cos = cross_target_scores(pe[img_i], event_embeds, cfg)
            sp_obj.append(spearman(mot, obj))
            sp_cos.append(spearman(mot, cos))

    def report(name: str, xs: list[float]) -> None:
        xs_sorted = sorted(xs)
        n = len(xs)
        print(f"  {name:32s} n={n:4d}  mean={sum(xs)/n:+.4f}  "
              f"p25={xs_sorted[n//4]:+.4f}  p50={xs_sorted[n//2]:+.4f}  p75={xs_sorted[3*n//4]:+.4f}")

    print("=== motion score의 스피어만 상관 ===")
    report("motion vs objectness(07-16 검증된 전경 프록시)", sp_obj)
    report("motion vs cos(기존 캡션 서로게이트)", sp_cos)

    print("\n=== 해석 기준 ===")
    print("  motion~objectness 상관이 뚜렷이 양(+)이면: 모션이 진짜 전경 물체를 잡는다는 방증.")
    print("  motion~cos 상관이 낮으면(0 근처): 모션이 캡션 코사인과 다른(=상보적) 정보라는 뜻")
    print("  → 낮은 게 오히려 좋은 신호(캡션이 못 잡는 걸 모션이 보완한다는 가설과 부합).")


if __name__ == "__main__":
    main()
