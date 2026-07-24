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
import gc
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


# Fields that change what a prediction in progress.jsonl actually means.
# --limit/--eval/--split etc. don't affect per-sample scoring and are
# deliberately excluded.
#
# 2026-07-24: "adapter"/"head" were missing here, which let a --out dir get
# resumed under a DIFFERENT LoRA adapter than produced its existing rows —
# runs/triage_boost_0.5 silently mixed 216 rows scored with the DPO-on-
# ckpt1200 adapter with rows scored with the SFT-only ckpt1200 adapter into
# one progress.jsonl/submission.csv, caught only by manually diffing ids
# against a backup dir. Unlike train_sft.check_out_reuse (where --adapter
# legitimately differs across a warm-start continuation), here a changed
# --adapter/--head IS a scoring-config change, not a legitimate resume.
_RESUME_DRIFT_FIELDS = (
    "tta", "stage2", "keep_ratio", "diversity_frac", "objectness_weight",
    "mmr_lambda", "motion_weight", "boost_frac", "boost_copies", "no_prune", "max_pixels",
    "adapter", "head",
)


def check_resume_config(out: Path, args: argparse.Namespace) -> None:
    """Refuse to append to an existing progress.jsonl under a different
    scoring config than produced those records, unless --allow-config-drift.

    2026-07-23: same class of bug as train_sft.check_out_reuse — progress.jsonl
    resumes purely by sample id (see module docstring) with no config check,
    so a crash-and-resume (scripts/run_pre_supervised.sh) or a manually
    reused --out with different flags would silently mix predictions scored
    under different configs (e.g. motion_weight 0.0 vs 0.3) into one
    submission.csv, with nothing recording which rows came from which config.
    """
    if getattr(args, "allow_config_drift", False):
        return
    prior_path = out / "config.json"
    progress_path = out / "progress.jsonl"
    if not prior_path.exists() or not progress_path.exists() or not progress_path.read_text().strip():
        return
    prior = json.loads(prior_path.read_text())
    drift = {
        f: (prior[f], getattr(args, f))
        for f in _RESUME_DRIFT_FIELDS
        if f in prior and prior[f] != getattr(args, f)
    }
    if drift:
        raise SystemExit(
            f"[resume-drift] {out} has progress.jsonl written under a different "
            f"scoring config: {drift}. Resuming would mix predictions from "
            f"different configs into one submission.csv. Use a new --out, or "
            f"pass --allow-config-drift if this is intentional."
        )


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
        motion_weight=args.motion_weight,
        boost_frac=args.boost_frac,
        boost_copies=args.boost_copies,
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
        if engine.prune_cfg.boost_frac > 0.0:
            idx = engine.scored_idx(prep, sample.caption, engine.prune_cfg)
            logits = engine.forward_prepared(prep, idx=idx)[0]
        else:
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


def predict_sample_resilient(
    engine: Engine, sample, n_tta: int, tau: float, stage2: str = "always", max_retries: int = 1,
) -> dict:
    """predict_sample, self-healing transient CUDA OOM in-process.

    2026-07-23: on 24GB cards, several hundred TTA8+stage2=always samples in
    accumulate allocator fragmentation until a <500MB allocation fails with
    ~20+GB already in use and a few hundred MB sitting reserved-but-unallocated
    in the cache. The known fix so far was an external supervisor
    (scripts/run_pre_supervised.sh) killing and relaunching the whole process
    on crash, which works but costs a full ~15s model reload per crash and
    depends on that wrapper being what actually invokes inference. This
    catches the same error one level up: empty_cache()+gc.collect() then
    retries the same sample in-process, so a genuinely transient blip never
    has to cost a process exit.

    2026-07-23 (same day, later): live 819-sample runs show this is NOT pure
    per-call fragmentation — the crash-time `used` VRAM creeps up over hours
    (23.15GB -> 23.23GB+) and "reserved but unallocated" stays several hundred
    MB even after repeated empty_cache() calls, i.e. a process-lifetime growth
    that only a real process restart (fresh CUDA context) reclaims. Measured
    over 37 retry sequences with the original max_retries=5 + exponential
    backoff: 36 fully exhausted all 5 attempts before crashing anyway (only 1
    recovered) — most of that 25s of backoff per doomed sample was pure waste
    that made the external supervisor's crash-restart cycle slower, not
    rarer. max_retries=1 with a short flat sleep keeps the cheap shot at a
    genuine transient blip while minimizing wasted time on the (now-dominant)
    case where only a full restart will actually free the memory.
    """
    for attempt in range(max_retries + 1):
        try:
            return predict_sample(engine, sample, n_tta, tau, stage2=stage2)
        except torch.cuda.OutOfMemoryError:
            if attempt == max_retries:
                raise
            print(f"[oom-retry] {sample.id}: attempt {attempt + 1}/{max_retries} — "
                  f"emptying cache and retrying in-process", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(1)


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
                    help="4 = balanced Klein set; 8 = balanced D4 set (2x views, same "
                         "exact cancellation, Ver8 champion recipe); "
                         "other n = identity + (n-1) seeded shuffles (legacy, e.g. 3)")
    ap.add_argument("--stage2", choices=["always", "cascade", "off"], default="always",
                    help="full-token stage-2 policy (cascade reproduces the pre-2026-07-17 pipeline)")
    ap.add_argument("--tau", type=float, default=0.10, help="cascade escalation margin (--stage2 cascade)")
    ap.add_argument("--keep-ratio", type=float, default=0.5)
    ap.add_argument("--diversity-frac", type=float, default=0.2)
    ap.add_argument("--objectness-weight", type=float, default=0.3)
    ap.add_argument("--mmr-lambda", type=float, default=0.5)
    ap.add_argument("--motion-weight", type=float, default=0.0,
                    help="cross-frame residual-norm blend weight (0 = pre-motion behavior)")
    ap.add_argument("--boost-frac", type=float, default=0.0,
                    help="duplicate this fraction of each image's TOP-scoring kept visual "
                         "tokens once more in the LLM input sequence, at the same m-rope "
                         "position as the original (0 = off, exact legacy sequence; "
                         "2026-07-23 token-boost track, unvalidated, see fitprune.py)")
    ap.add_argument("--boost-copies", type=int, default=1,
                    help="extra copies appended per boosted token (ignored when --boost-frac 0)")
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eval", action="store_true",
                     help="score against truth (train split only; in-sample since Ver11 trains on 100%)")
    ap.add_argument("--allow-config-drift", action="store_true",
                    help="allow resuming an --out whose progress.jsonl was written under a "
                         "different scoring config (see check_resume_config)")
    args = ap.parse_args(argv)

    if args.model_id is None:
        from run_common import resolve_model_id

        args.model_id = resolve_model_id()
    out = Path(args.out or f"runs/{args.split}_v11")
    out.mkdir(parents=True, exist_ok=True)
    check_resume_config(out, args)

    samples = load_samples(args.data_root, args.split)
    if args.limit:
        samples = samples[: args.limit]

    progress_path = out / "progress.jsonl"
    done: dict[str, dict] = {}
    if progress_path.exists():
        n_torn = 0
        for line in progress_path.read_text().splitlines():
            if not line.strip() or "\x00" in line:
                n_torn += 1
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # a mid-write crash (e.g. power loss) can leave the last
                # record truncated on disk; drop it and let it be redone.
                n_torn += 1
                continue
            done[rec["id"]] = rec
        if n_torn:
            print(f"[resume] dropped {n_torn} torn/incomplete trailing record(s)")
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
            rec = predict_sample_resilient(engine, sample, args.tta, args.tau, stage2=args.stage2)
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
