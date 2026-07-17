"""Inference CLI — one-pass score24 + TTA + full-token stage 2.

Stage 1: Cross-Targeted FitPrune tokens (~50%) -> score24 head -> 24-class
probs per TTA view -> remap to original space -> aggregate.
Stage 2: FULL-token forwards on the SAME prepared tensors (vision tower is
not recomputed) -> aggregate stages together. Policy via --stage2:
  always  (default) every sample — under the 24h budget the extra forwards
          are nearly free, and the full pass adds the pruned-away evidence
          for everyone, not just low-margin samples;
  cascade only when the stage-1 margin < tau (the validated efficiency
          path — byte-compatible with the pre-2026-07-17 pipeline);
  off     never (stage-1 only).

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
from tqdm import tqdm

from . import perm
from .data import load_samples
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
        objectness_weight=args.objectness_weight,
        mmr_lambda=args.mmr_lambda,
        enabled=not args.no_prune,
    )
    return Engine(model, processor, head, prune_cfg, max_pixels=args.max_pixels)


@torch.no_grad()
def predict_sample(engine: Engine, sample, n_tta: int, tau: float, stage2: str = "always") -> dict:
    """Returns dict with pred rank, probs, margins, escalation flag.

    stage2 in {"always", "cascade", "off"} — see module docstring. Stage 2 is
    skipped whenever pruning is disabled (stage 1 already saw full tokens, a
    second identical pass would only duplicate the same evidence).
    """
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
    margin1 = margin_of(probs)  # stage-1 margin — tau 포렌식용 (semantics 불변)

    escalated = engine.prune_cfg.enabled and stage2 != "off" and (
        stage2 == "always" or margin1 < tau
    )
    if escalated:
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
        "margin": round(margin1, 6),
        "margin_final": round(margin_of(probs), 6),
        "escalated": bool(escalated),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--split", choices=["test", "train"], default="test")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--head", default=None)
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tta", type=int, default=4,
                    help="4 = balanced Klein set (exact position-bias cancellation); "
                         "other n = identity + (n-1) seeded shuffles (legacy, e.g. 3)")
    ap.add_argument("--stage2", choices=["always", "cascade", "off"], default="always",
                    help="full-token stage-2 policy (cascade reproduces the pre-2026-07-17 pipeline)")
    ap.add_argument("--tau", type=float, default=0.10, help="cascade escalation margin (--stage2 cascade)")
    ap.add_argument("--keep-ratio", type=float, default=0.5)
    ap.add_argument("--diversity-frac", type=float, default=0.2)
    ap.add_argument("--objectness-weight", type=float, default=0.3)
    ap.add_argument("--mmr-lambda", type=float, default=0.5)
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eval", action="store_true",
                     help="score against truth (train split only; in-sample since Ver11 trains on 100%)")
    args = ap.parse_args(argv)

    if args.model_id is None:
        from run_common import resolve_model_id

        args.model_id = resolve_model_id()
    out = Path(args.out or f"runs/{args.split}_v11")
    out.mkdir(parents=True, exist_ok=True)

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

    n_esc = 0
    # 이미 완료된 샘플은 스킵만 하므로 재개 시 진행바가 그 지점까지 빨리감기된다.
    pbar = tqdm(samples, desc=f"infer:{args.split}", dynamic_ncols=True, mininterval=5.0)
    with open(progress_path, "a") as f:
        for sample in pbar:
            if sample.id in done:
                continue
            t0 = time.time()
            rec = predict_sample(engine, sample, args.tta, args.tau, stage2=args.stage2)
            rec["id"] = sample.id
            rec["elapsed_s"] = round(time.time() - t0, 2)
            if sample.rank is not None:
                rec["truth"] = list(sample.rank)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[sample.id] = rec
            n_esc += int(rec["escalated"])
            pbar.set_postfix(esc=n_esc, s_per=f"{rec['elapsed_s']:.1f}", refresh=False)

    records = [done[s.id] for s in samples if s.id in done]

    if args.eval:
        labeled = [(r, s) for r, s in zip(records, samples) if s.rank is not None]
        em = sum(tuple(r["rank"]) == s.rank for r, s in labeled) / max(1, len(labeled))
        pw = sum(perm.pairwise_score(tuple(r["rank"]), s.rank) for r, s in labeled) / max(1, len(labeled))
        esc = sum(r["escalated"] for r, _ in labeled)
        report = {"n": len(labeled), "em": round(em, 4), "pairwise": round(pw, 4), "escalated": esc}
        (out / "eval.json").write_text(json.dumps(report, indent=2))
        print(f"[eval] EM {em:.4f} | pairwise {pw:.4f} | escalated {esc}/{len(labeled)}")

    if args.split == "test":
        rows = [(r["id"], tuple(r["rank"])) for r in records]
        sub = write_submission(rows, out / "submission.csv", Path(args.data_root) / "sample_submission.csv")
        print(f"[submission] {sub} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
