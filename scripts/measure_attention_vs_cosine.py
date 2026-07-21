#!/usr/bin/env python
"""Attention-harvesting pruning — feasibility/qualitative check (Agent-A idea #1).

fitprune.py's own docstring calls its cosine scoring a "training-free surrogate
for FitPrune's attention statistics". This script replaces the surrogate with
the model's OWN real, contextualized attention (via attn_implementation="eager"
+ forward hooks on a couple of middle decoder layers) and compares it to the
existing cosine-based cross_target_scores map on the SAME 3 forensic samples
used to validate the 07-16 objectness+MMR fix (docs/fitprune_fix_2026-07-16.md):
diZi5g (skier), u7w0lr (pool), 2vqGOF (lens).

Scope deliberately kept tight (reduced max_pixels, 2 middle layers, no
generation, single forward per sample, immediate CPU-move + cache clear after
each hook fire) — this machine has a thermal constraint, budget is a single
short run, not an exploration session.

Usage: python scripts/measure_attention_vs_cosine.py --out runs/attn_vs_cosine
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

FORENSIC_IDS = ["diZi5g", "u7w0lr", "2vqGOF"]


def find_caption_token_range(tokenizer, input_ids: list[int], caption: str) -> tuple[int, int]:
    """Locate the caption's tokens as a subsequence of the REAL (post
    image-placeholder-expansion) input_ids — NOT a fresh text-only
    tokenization, whose indices do not correspond to the actual multimodal
    sequence (image placeholders expand to hundreds of tokens each inside
    the processor's own call, which a standalone tokenizer() call never
    sees). Trims outer tokens on retry to dodge BPE boundary-merge
    differences between standalone and in-context tokenization."""
    needle = f'"{caption.strip()}"'
    cap_ids = tokenizer(needle, add_special_tokens=False).input_ids
    for trim in range(0, 3):
        probe = cap_ids[trim: len(cap_ids) - trim] if trim else cap_ids
        if len(probe) < 3:
            break
        for i in range(len(input_ids) - len(probe) + 1):
            if input_ids[i: i + len(probe)] == probe:
                return i, i + len(probe)
    raise ValueError(f"caption token subsequence not found (tried trims 0-2): {caption!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--max-pixels", type=int, default=401408, help="축소 해상도 — 어텐션 O(seq^2) 메모리 제어")
    ap.add_argument("--out", default="runs/attn_vs_cosine")
    args = ap.parse_args()

    from run_common import ensure_data, resolve_model_id
    from snuai11.data import load_samples
    from snuai11.fitprune import PruneConfig, cross_target_scores
    from snuai11.fsm import letter_token_ids
    from snuai11.vlm import Engine, Score24Head, load_model_and_processor

    model_id = args.model_id or resolve_model_id()
    data_root = ensure_data("data")
    all_samples = {s.id: s for s in load_samples(data_root, "train")}
    samples = [all_samples[sid] for sid in FORENSIC_IDS if sid in all_samples]
    print(f"[attn] {len(samples)}/{len(FORENSIC_IDS)}개 포렌식 샘플 확보: {[s.id for s in samples]}")

    print(f"[attn] loading {model_id} (attn_implementation=eager, max_pixels={args.max_pixels})")
    model, processor = load_model_and_processor(model_id, attn_implementation="eager")
    letter_ids = letter_token_ids(processor.tokenizer)
    head = Score24Head.init_from_lm_head(model, letter_ids).to(model.lm_head.weight.device).eval()
    cfg = PruneConfig()
    engine = Engine(model, processor, head, cfg, max_pixels=args.max_pixels)

    n_layers = len(engine.lm.layers)
    targets = sorted({n_layers // 3, 2 * n_layers // 3})
    print(f"[attn] {n_layers}층 중 {targets}층에서 어텐션 수확")

    captured: dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def hook(_module, _inputs, output):
            aw = output[1]  # [1, heads, seq, seq]
            captured[idx] = aw[0].mean(dim=0).detach().to("cpu", dtype=torch.float32)  # [seq, seq]
        return hook

    handles = [engine.lm.layers[i].self_attn.register_forward_hook(make_hook(i)) for i in targets]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []

    try:
        for s in samples:
            print(f"\n[attn] === {s.id}: {s.caption!r} ===")
            enc = engine.encode(s.image_paths, s.caption)
            real_input_ids = enc["input_ids"][0].cpu().tolist()
            cap_lo, cap_hi = find_caption_token_range(processor.tokenizer, real_input_ids, s.caption)
            print(f"  caption 토큰 범위(실 input_ids 기준): [{cap_lo}, {cap_hi})  (전체 길이 {len(real_input_ids)})")

            event_embeds = engine.event_embeds(s.caption)
            captured.clear()
            with torch.no_grad():
                prep = engine.prepare(s.image_paths, s.caption)
                _ = engine.forward_prepared(prep, keep=None)
            for img_i in range(4):
                lo, hi = prep.image_positions[img_i].min().item(), prep.image_positions[img_i].max().item()
                print(f"    img{img_i+1} 실제 비주얼 위치 범위 [{lo},{hi}]  "
                      f"(캡션[{cap_lo},{cap_hi}) 뒤에 있어야 정상 — causal mask상 캡션이 봐야 함)")

            if len(targets) == 2:
                a, b = captured[targets[0]], captured[targets[1]]
                same = torch.equal(a, b)
                maxdiff = (a - b).abs().max().item()
                print(f"  [debug] layer{targets[0]} vs layer{targets[1]}: identical={same} "
                      f"maxdiff={maxdiff:.6f} shapeA={tuple(a.shape)} shapeB={tuple(b.shape)} "
                      f"sumA={a.sum().item():.4f} sumB={b.sum().item():.4f} "
                      f"same_object={captured[targets[0]] is captured[targets[1]]}")

            for layer_idx, aw in captured.items():
                cap_attn = aw[cap_lo:cap_hi, :].mean(dim=0)  # [seq] — real attention FROM caption tokens
                sink_mass = cap_attn[0].item()
                cap_attn_desunk = cap_attn.clone()
                cap_attn_desunk[0] = 0.0
                cap_attn_desunk = cap_attn_desunk / cap_attn_desunk.sum().clamp_min(1e-8)  # renormalize sans sink
                top5_idx = cap_attn.topk(5).indices.tolist()
                print(f"    [debug] layer{layer_idx}: sink(pos0)={sink_mass:.4f} "
                      f"top5_positions={top5_idx} top5_vals={[round(v,4) for v in cap_attn.topk(5).values.tolist()]}")
                for img_i in range(4):
                    positions = prep.image_positions[img_i].cpu()
                    visual = prep.per_image_embeds[img_i]
                    real_map = cap_attn_desunk[positions]  # de-sunk, renormalized real attention
                    cos_map = cross_target_scores(visual, event_embeds, cfg).cpu()  # [N] existing surrogate

                    real_r = real_map.argsort(descending=True).argsort().float()
                    cos_r = cos_map.argsort(descending=True).argsort().float()
                    spearman = torch.corrcoef(torch.stack([real_r, cos_r]))[0, 1].item()

                    k = max(1, round(0.5 * len(real_map)))
                    real_top = set(real_map.topk(k).indices.tolist())
                    cos_top = set(cos_map.topk(k).indices.tolist())
                    jacc = len(real_top & cos_top) / len(real_top | cos_top)

                    rec = {"id": s.id, "layer": layer_idx, "img": img_i + 1,
                           "attn_mass_desunk": round(real_map.sum().item(), 4),
                           "spearman": round(spearman, 4), "top50pct_jaccard": round(jacc, 4)}
                    results.append(rec)
                    print(f"      img{img_i+1}: mass(desunk)={real_map.sum().item():.4f}  "
                          f"spearman={spearman:+.3f}  top50%-jaccard={jacc:.3f}")
            del prep
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        for h in handles:
            h.remove()

    import json
    (out_dir / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    n = len(results)
    mean_sp = sum(r["spearman"] for r in results) / n
    mean_j = sum(r["top50pct_jaccard"] for r in results) / n
    print(f"\n[attn] 전체 평균 (n={n}): spearman={mean_sp:+.3f}  top50%-jaccard={mean_j:.3f}")
    print(f"[attn] 저장 -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
