"""Inference CLI — one-pass score24 + TTA + margin cascade.

Stage 1 (fast): Cross-Targeted FitPrune tokens (~50%) -> score24 head ->
24-class probs per TTA view -> remap to original space -> aggregate.
Stage 2 (escalation, margin < tau): FULL-token forwards on the SAME prepared
tensors (vision tower is not recomputed) -> aggregate stages together.

Unlike Ver5's rejected "re-ask the same model the same information" cascade,
stage 2 adds information (the pruned-away tokens) rather than re-phrasing.

Resumable: progress.jsonl is append-only; done ids are skipped on restart.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from . import perm
from .data import load_samples, split_train_holdout
from .fitprune import PruneConfig
from .submission import write_submission
from .tta import aggregate_logprobs, margin_of, normalize, remap_scores, tta_views
from .vlm import DEFAULT_MAX_PIXELS, Engine, Score24Head, load_model_and_processor


def load_engine(args) -> Engine:
    model, processor = load_model_and_processor(args.model_id, four_bit=args.four_bit)
    if args.adapter:
        from peft import PeftModel

        PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
        print(f"[engine] adapter attached: {args.adapter}")
    if args.head:
        head_path = Path(args.head)
    elif args.adapter and (Path(args.adapter).parent / "head.pt").exists():
        head_path = Path(args.adapter).parent / "head.pt"
    else:
        head_path = None
    from .fsm import letter_token_ids

    letter_ids = letter_token_ids(processor.tokenizer)
    if head_path:
        head = Score24Head.load(head_path, device="cuda")
        print(f"[engine] head loaded: {head_path}")
    else:
        head = Score24Head.init_from_lm_head(model, letter_ids).to("cuda")
        print("[engine] zero-shot head (lm_head letter rows)")
    head.eval()
    prune_cfg = PruneConfig(
        keep_ratio=args.keep_ratio,
        diversity_frac=args.diversity_frac,
        enabled=not args.no_prune,
    )
    return Engine(model, processor, head, prune_cfg, max_pixels=args.max_pixels)


@torch.no_grad()
def predict_sample(engine: Engine, sample, n_tta: int, tau: float) -> dict:
    """Returns dict with pred rank, probs, margin, escalation flag."""
    views = tta_views(n_tta, sample.id)
    preps, view_probs = [], []
    for sigma in views:
        images = [sample.image_paths[sigma[j]] for j in range(4)]
        prep = engine.prepare(images, sample.caption)
        preps.append((sigma, prep))
        keep = engine.keep_mask(prep, sample.caption, engine.prune_cfg)
        logits = engine.forward_prepared(prep, keep)[0]
        view_probs.append(remap_scores(normalize(logits), sigma))

    agg = aggregate_logprobs(view_probs)
    probs = normalize(agg)
    margin = margin_of(probs)
    escalated = False

    if engine.prune_cfg.enabled and margin < tau:
        escalated = True
        for sigma, prep in preps:
            logits = engine.forward_prepared(prep, keep=None)[0]  # full tokens
            view_probs.append(remap_scores(normalize(logits), sigma))
        agg = aggregate_logprobs(view_probs)
        probs = normalize(agg)

    pred_class = int(probs.argmax())
    return {
        "pred_class": pred_class,
        "rank": list(perm.rank_of_index(pred_class)),
        "probs": [round(float(p), 6) for p in probs],
        "margin": round(margin, 6),
        "escalated": escalated,
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--holdout-val", action="store_true", help="evaluate on the 945-sample holdout")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--head", default=None)
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tta", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--keep-ratio", type=float, default=0.5)
    ap.add_argument("--diversity-frac", type=float, default=0.2)
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eval", action="store_true")
    args = ap.parse_args(argv)

    if args.model_id is None:
        from run_common import resolve_model_id

        args.model_id = resolve_model_id()
    out = Path(args.out or ("runs/holdout_v11" if args.holdout_val else f"runs/{args.split}_v11"))
    out.mkdir(parents=True, exist_ok=True)

    if args.holdout_val:
        alls = load_samples(args.data_root, "train")
        _, samples = split_train_holdout(alls)
        (out / "split.json").write_text(json.dumps({"holdout_ids": sorted(s.id for s in samples)}))
    else:
        samples = load_samples(args.data_root, args.split)
    if args.limit:
        samples = samples[: args.limit]

    progress_path = out / "progress.jsonl"
    done: dict[str, dict] = {}
    if progress_path.exists():
        for line in progress_path.read_text().splitlines():
            rec = json.loads(line)
            done[rec["id"]] = rec
        print(f"[resume] {len(done)} samples already done")

    engine = load_engine(args)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    n_esc, t_start = 0, time.time()
    with open(progress_path, "a") as f:
        for i, sample in enumerate(samples):
            if sample.id in done:
                continue
            t0 = time.time()
            rec = predict_sample(engine, sample, args.tta, args.tau)
            rec["id"] = sample.id
            rec["elapsed_s"] = round(time.time() - t0, 2)
            if sample.rank is not None:
                rec["truth"] = list(sample.rank)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[sample.id] = rec
            n_esc += int(rec["escalated"])
            if (i + 1) % 25 == 0:
                rate = (time.time() - t_start) / max(1, len(done))
                print(f"[{len(done)}/{len(samples)}] {rec['elapsed_s']:.1f}s/sample, escalated {n_esc}")

    records = [done[s.id] for s in samples if s.id in done]

    if args.eval or args.holdout_val:
        labeled = [(r, s) for r, s in zip(records, samples) if s.rank is not None]
        em = sum(tuple(r["rank"]) == s.rank for r, s in labeled) / max(1, len(labeled))
        pw = sum(perm.pairwise_score(tuple(r["rank"]), s.rank) for r, s in labeled) / max(1, len(labeled))
        esc = sum(r["escalated"] for r, _ in labeled)
        report = {"n": len(labeled), "em": round(em, 4), "pairwise": round(pw, 4), "escalated": esc}
        (out / "eval.json").write_text(json.dumps(report, indent=2))
        print(f"[eval] EM {em:.4f} | pairwise {pw:.4f} | escalated {esc}/{len(labeled)}")

    if args.split == "test" and not args.holdout_val:
        rows = [(r["id"], tuple(r["rank"])) for r in records]
        sub = write_submission(rows, out / "submission.csv", Path(args.data_root) / "sample_submission.csv")
        print(f"[submission] {sub} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
