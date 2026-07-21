#!/usr/bin/env python
"""Gauge cross-frame contrast magnitude — training-free check for whether
per-event visual-text matching differs meaningfully ACROSS a sample's 4
images, before investing in the keep_mask/fitprune refactor that would be
needed for frame-differentiated pruning (docs/fitprune_fix_2026-07-16.md
§6 item 6, "프레임 간 대비 점수").

Two phases, split so the expensive GPU part is cached and every re-analysis
(different pooling, different margin definition) is free CPU-only reruns:

  Phase 1 (GPU, needs the model): for each of N sampled train items, decompose
    the caption into 4 events and run ONLY the vision tower (Engine.prepare —
    no LLM forward, no adapter) to get per-image visual tokens + event text
    embeddings. Cached to --out/embeds_cache.pt; skipped if it already exists.

  Phase 2 (CPU, from cache): per sample, per event e, compute the 4 images'
    max-pooled match score for e, and margin_e = top1 image score - top2
    (mirrors the project's existing score24 "margin" vocabulary). This is the
    REAL signal a frame-contrast redesign would exploit.

    Null control: for the same sample's caption/events, swap in 4 images
    drawn from OTHER random samples (reusing cached embeddings — zero extra
    GPU cost) and recompute the same margin. If real margins aren't bigger
    than this cross-sample null, the apparent differentiation is embedding
    noise, not "which frame depicts this event" -- the whole premise for a
    frame-contrast redesign would be unsupported.

Usage:
  python scripts/measure_frame_contrast.py --n 300 --out runs/frame_contrast
  (add --four-bit --model-id ... only if testing on an unquantized checkpoint)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import torch  # noqa: E402


def extract(args) -> Path:
    cache_path = Path(args.out) / "embeds_cache.pt"
    if cache_path.exists():
        print(f"[extract] 캐시 존재 — 재사용: {cache_path}")
        return cache_path

    from run_common import ensure_data, resolve_model_id
    from snuai11.data import load_samples
    from snuai11.decompose import decompose_caption, split_clauses
    from snuai11.fsm import letter_token_ids
    from snuai11.vlm import DEFAULT_MAX_PIXELS, Engine, PruneConfig, Score24Head, load_model_and_processor
    from tqdm import tqdm

    model_id = args.model_id or resolve_model_id()
    data_root = ensure_data("data")
    all_samples = load_samples(data_root, "train")
    rng = random.Random(args.seed)
    samples = rng.sample(all_samples, min(args.n, len(all_samples)))
    print(f"[extract] train {len(all_samples)}건 중 {len(samples)}건 표본(seed={args.seed})")

    print(f"[extract] loading {model_id} (four_bit={args.four_bit})")
    model, processor = load_model_and_processor(model_id, four_bit=args.four_bit)
    letter_ids = letter_token_ids(processor.tokenizer)
    head = Score24Head.init_from_lm_head(model, letter_ids).to(model.lm_head.weight.device).eval()
    engine = Engine(model, processor, head, PruneConfig(), max_pixels=args.max_pixels or DEFAULT_MAX_PIXELS)

    records = []
    for s in tqdm(samples, desc="extract", unit="샘플", mininterval=2.0):
        events = decompose_caption(s.caption)
        n_clauses = len(split_clauses(s.caption))
        event_embeds = engine.event_embeds(s.caption)
        with torch.no_grad():
            prep = engine.prepare(s.image_paths, s.caption)
        records.append({
            "id": s.id,
            "caption": s.caption,
            "n_clauses": n_clauses,
            "events": events,
            "event_embeds": [e.cpu() for e in event_embeds],
            "per_image_embeds": [t.cpu() for t in prep.per_image_embeds],
        })

    Path(args.out).mkdir(parents=True, exist_ok=True)
    torch.save(records, cache_path)
    print(f"[extract] {len(records)}건 저장 -> {cache_path}")
    return cache_path


def analyze(cache_path: Path, args) -> None:
    from snuai11.fitprune import PruneConfig, per_event_scores

    records = torch.load(cache_path, weights_only=False)
    n = len(records)
    cfg = PruneConfig()  # only text_pool matters here (default "max")
    rng = random.Random(args.seed + 1)

    def img_event_matrix(event_embeds, per_image_embeds) -> torch.Tensor:
        """[4 images, 4 events] max-pooled match score."""
        rows = []
        for img_i in range(4):
            pe = per_event_scores(per_image_embeds[img_i], event_embeds, cfg)  # [E, N]
            rows.append(pe.max(dim=1).values)
        return torch.stack(rows, dim=0)  # [4, 4]

    def margins(mat: torch.Tensor) -> list[float]:
        """top1 - top2 across images, per event. len 4."""
        top2 = torch.topk(mat, k=2, dim=0).values  # [2, E]
        return (top2[0] - top2[1]).tolist()

    real_margin, real_clauses, real_id = [], [], []
    for r in records:
        mat = img_event_matrix(r["event_embeds"], r["per_image_embeds"])
        for m in margins(mat):
            real_margin.append(m)
            real_clauses.append(r["n_clauses"])
            real_id.append(r["id"])

    K = args.null_draws
    null_margin, null_clauses = [], []
    pool = list(range(n))
    for i, r in enumerate(records):
        others = [j for j in pool if j != i]
        for _ in range(K):
            picks = rng.sample(others, 4)
            null_embeds = [records[p]["per_image_embeds"][rng.randrange(4)] for p in picks]
            mat = img_event_matrix(r["event_embeds"], null_embeds)
            for m in margins(mat):
                null_margin.append(m)
                null_clauses.append(r["n_clauses"])

    def pctl(xs: list[float], p: float) -> float:
        xs = sorted(xs)
        return xs[int(round(p * (len(xs) - 1)))]

    def report(name: str, xs: list[float]) -> None:
        print(f"  {name:28s} n={len(xs):6d}  mean={sum(xs)/len(xs):.4f}"
              f"  p50={pctl(xs,.50):.4f}  p75={pctl(xs,.75):.4f}"
              f"  p90={pctl(xs,.90):.4f}  p95={pctl(xs,.95):.4f}")

    print(f"\n=== 대비(margin) 분포 — 샘플 {n}건, 이벤트당 {len(real_margin)}개 실측값, "
          f"null draw K={K} ({len(null_margin)}개) ===\n")
    print("[전체]")
    report("real (같은 샘플 4장)", real_margin)
    report("null (무관 샘플 4장)", null_margin)

    print("\n[절 개수(n_clauses)별 — real]")
    for bucket, lo, hi in [("1절", 1, 1), ("2절", 2, 2), ("3+절", 3, 99)]:
        xs = [m for m, c in zip(real_margin, real_clauses) if lo <= c <= hi]
        if xs:
            report(f"{bucket} (n_caption≈{len(xs)//4})", xs)

    print("\n[절 개수별 — null]")
    for bucket, lo, hi in [("1절", 1, 1), ("2절", 2, 2), ("3+절", 3, 99)]:
        xs = [m for m, c in zip(null_margin, null_clauses) if lo <= c <= hi]
        if xs:
            report(f"{bucket}", xs)

    # per-sample permutation-style signal check: real max-margin vs that
    # sample's own null draws' max-margin distribution.
    null_by_id: dict[str, list[float]] = {}
    idx = 0
    for r in records:
        vals = []
        for _ in range(K):
            vals.append(max(null_margin[idx:idx + 4]))
            idx += 4
        null_by_id[r["id"]] = vals

    beats_null = 0
    for r in records:
        mat = img_event_matrix(r["event_embeds"], r["per_image_embeds"])
        real_max = max(margins(mat))
        null_vals = null_by_id[r["id"]]
        if real_max > max(null_vals):
            beats_null += 1
    print(f"\n[샘플별 신호 체크] real max-margin이 자기 null 분포(K={K}개) 전부를 능가한 샘플: "
          f"{beats_null}/{n} ({beats_null/n:.1%})")

    out = {
        "n_samples": n, "null_draws": K,
        "real_margin_mean": sum(real_margin) / len(real_margin),
        "null_margin_mean": sum(null_margin) / len(null_margin),
        "beats_null_own_fraction": beats_null / n,
    }
    summary_path = cache_path.parent / "summary.json"
    summary_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n[analyze] 요약 저장 -> {summary_path}")

    # 3 qualitative examples: highest real max-margin, for a caption+winner sanity read
    scored = []
    for r in records:
        mat = img_event_matrix(r["event_embeds"], r["per_image_embeds"])
        scored.append((max(margins(mat)), r))
    scored.sort(key=lambda t: -t[0])
    print("\n[정성 확인용 상위 3건 — 캡션 vs 이벤트별 승자 이미지]")
    for m, r in scored[:3]:
        mat = img_event_matrix(r["event_embeds"], r["per_image_embeds"])
        winners = [int(mat[:, e].argmax()) + 1 for e in range(4)]
        print(f"  {r['id']}  max_margin={m:.4f}  n_clauses={r['n_clauses']}")
        print(f"    caption: {r['caption']!r}")
        for e, ev in enumerate(r["events"]):
            print(f"    E{e+1} (winner=img{winners[e]}): {ev!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--null-draws", type=int, default=3, dest="null_draws")
    ap.add_argument("--max-pixels", type=int, default=None)
    ap.add_argument("--out", default="runs/frame_contrast")
    args = ap.parse_args()

    cache_path = extract(args)
    analyze(cache_path, args)


if __name__ == "__main__":
    main()
