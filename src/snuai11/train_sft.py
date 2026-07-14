"""Training CLI — score24 SFT (phase 1) and adjacent-swap margin DPO (phase 2),
both under Stackelberg two-time-scale optimization.

Body M (leader, slow lr): LoRA adapters on the language layers only.
Head w (follower, 5x lr + l2): Score24Head, init from lm_head letter rows.
Single-loop simultaneous updates (one head step per body step, same batch) —
exactly the scheme analyzed in the Stackelberg paper.

Pruning is ACTIVE during training (prob --prune-prob) so the model adapts to
the pruned token distribution (FitPrune's shallow-pruning penalty mitigation);
the remaining fraction trains on full tokens so the cascade's escalation path
stays in-distribution.

Usage (A100):  python run_fit.py                  # SFT with defaults
               python run_fit.py --phase dpo --adapter runs/sft32b_v11/adapter_final
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import perm
from .data import load_samples, split_train_holdout, uniform_augment
from .fitprune import PruneConfig
from .fsm import letter_token_ids
from .stackelberg import StackelbergConfig, build_optimizer, build_param_groups, build_scheduler
from .vlm import (
    DEFAULT_MAX_PIXELS,
    Engine,
    Score24Head,
    load_model_and_processor,
    verify_lora_only_on_language,
    write_env_report,
)

LORA_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def lora_target_modules(model) -> list[str]:
    """Full-path LoRA targets: language decoder layers only (never vision,
    never lm_head/embed_tokens)."""
    targets = [
        name
        for name, module in model.named_modules()
        if ".language_model.layers." in name
        and name.endswith(LORA_SUFFIXES)
        and hasattr(module, "weight")
    ]
    if not targets:
        raise RuntimeError("no LoRA targets found — model layout changed?")
    return targets


def attach_lora(model, r: int, alpha: int, dropout: float, adapter: str | None):
    from peft import LoraConfig, PeftModel, get_peft_model

    if adapter:
        peft_model = PeftModel.from_pretrained(model, adapter, is_trainable=True)
    else:
        cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=lora_target_modules(model),
            task_type=None,
        )
        peft_model = get_peft_model(model, cfg)
    verify_lora_only_on_language(model)
    return peft_model


def margin_dpo_loss(logits24: torch.Tensor, label: int, beta: float, ce_weight: float) -> torch.Tensor:
    """Reference-free class-margin DPO: push the GT class log-prob above its 3
    adjacent-swap (Kendall-distance-1) hard negatives — the dominant error
    mode. A small CE anchor keeps the absolute scale from drifting."""
    logp = F.log_softmax(logits24, dim=-1)[0]
    gt_rank = perm.rank_of_index(label)
    negs = [perm.index_of(n) for n in perm.adjacent_swap_neighbors(gt_rank)]
    margins = torch.stack([logp[label] - logp[n] for n in negs])
    loss = -F.logsigmoid(beta * margins).mean()
    if ce_weight > 0:
        loss = loss + ce_weight * F.cross_entropy(logits24, torch.tensor([label], device=logits24.device))
    return loss


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--phase", choices=["sft", "dpo"], default="sft")
    ap.add_argument("--adapter", default=None, help="resume/init adapter (required for dpo)")
    ap.add_argument("--head", default=None, help="head .pt to resume (defaults to <adapter>/../head.pt)")
    ap.add_argument("--steps", type=int, default=None, help="optimizer steps (default: sft 2000 / dpo 400)")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=17)
    # Stackelberg
    ap.add_argument("--body-lr", type=float, default=None, help="default: sft 2e-4 / dpo 5e-5")
    ap.add_argument("--lr-ratio", type=float, default=5.0, help="head_lr = ratio * body_lr (paper band 3-10x)")
    ap.add_argument("--head-wd", type=float, default=0.1)
    ap.add_argument("--schedule", choices=["cosine", "poly", "constant"], default="cosine")
    ap.add_argument("--uniform-lr", action="store_true", help="ablation arm: head_lr = body_lr")
    # FitPrune
    ap.add_argument("--keep-ratio", type=float, default=0.5)
    ap.add_argument("--diversity-frac", type=float, default=0.2, help="fraction of KEEP budget from diversity tokens")
    ap.add_argument("--prune-prob", type=float, default=0.75, help="per-sample prob of training with pruning on")
    ap.add_argument("--no-prune", action="store_true")
    ap.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    # LoRA
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--four-bit", action="store_true", help="on-the-fly 4bit for UNquantized ckpts (8B smoke)")
    # DPO
    ap.add_argument("--dpo-beta", type=float, default=1.0)
    ap.add_argument("--dpo-ce-weight", type=float, default=0.2)
    ap.add_argument("--holdout-size", type=int, default=945)
    args = ap.parse_args(argv)

    if args.model_id is None:
        from run_common import resolve_model_id

        args.model_id = resolve_model_id()
    if args.phase == "dpo" and not args.adapter:
        ap.error("--phase dpo requires --adapter (the SFT output)")
    if args.steps is None:
        args.steps = 2000 if args.phase == "sft" else 400
    if args.body_lr is None:
        args.body_lr = 2e-4 if args.phase == "sft" else 5e-5
    out = Path(args.out or f"runs/{'sft' if args.phase == 'sft' else 'dpo'}32b_v11")
    out.mkdir(parents=True, exist_ok=True)
    write_env_report(out)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- data ----------------------------------------------------------
    all_samples = load_samples(args.data_root, "train")
    train_samples, holdout = split_train_holdout(all_samples, args.holdout_size)
    (out / "split.json").write_text(json.dumps({"holdout_ids": sorted(s.id for s in holdout)}))
    print(f"[data] train {len(train_samples)} / holdout {len(holdout)} (never trained on)")

    # ---- model ---------------------------------------------------------
    model, processor = load_model_and_processor(args.model_id, four_bit=args.four_bit)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    peft_model = attach_lora(model, args.lora_r, args.lora_alpha, args.lora_dropout, args.adapter)

    letter_ids = letter_token_ids(processor.tokenizer)
    if args.head:
        head = Score24Head.load(args.head, device="cuda")
    elif args.adapter and (Path(args.adapter).parent / "head.pt").exists():
        head = Score24Head.load(Path(args.adapter).parent / "head.pt", device="cuda")
    else:
        head = Score24Head.init_from_lm_head(model, letter_ids).to("cuda")
    head.train().requires_grad_(True)

    prune_cfg = PruneConfig(
        keep_ratio=args.keep_ratio,
        diversity_frac=args.diversity_frac,
        enabled=not args.no_prune,
    )
    engine = Engine(model, processor, head, prune_cfg, max_pixels=args.max_pixels)

    # ---- Stackelberg optimizer ------------------------------------------
    body_params = [p for n, p in peft_model.named_parameters() if "lora_" in n and p.requires_grad]
    head_lr = args.body_lr * (1.0 if args.uniform_lr else args.lr_ratio)
    scfg = StackelbergConfig(
        body_lr=args.body_lr,
        head_lr=head_lr,
        head_weight_decay=args.head_wd,
        schedule=args.schedule,
    )
    groups = build_param_groups(body_params, list(head.parameters()), scfg)
    optimizer = build_optimizer(groups, scfg)
    scheduler = build_scheduler(optimizer, scfg, total_steps=args.steps)
    n_body = sum(p.numel() for p in body_params)
    print(f"[stackelberg] body {n_body/1e6:.1f}M @ {scfg.body_lr:g} | head {sum(p.numel() for p in head.parameters())} @ {scfg.head_lr:g} (wd {scfg.head_weight_decay})")

    # ---- loop ------------------------------------------------------------
    peft_model.train()
    rng = random.Random(args.seed)
    log_path = out / "train_log.jsonl"
    pool: list[int] = []
    running: list[float] = []
    hits: list[float] = []
    t0 = time.time()

    def save(tag: str) -> None:
        ckpt = out / tag
        peft_model.save_pretrained(str(ckpt / "adapter"))
        head.save(ckpt / "head.pt")
        (ckpt / "train_args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.accum):
            if not pool:
                pool = list(range(len(train_samples)))
                rng.shuffle(pool)
            sample = uniform_augment(train_samples[pool.pop()], rng)
            prep = engine.prepare(sample.image_paths, sample.caption)
            use_prune = prune_cfg.enabled and rng.random() < args.prune_prob
            keep = engine.keep_mask(prep, sample.caption, prune_cfg) if use_prune else None
            logits = engine.forward_prepared(prep, keep)
            if args.phase == "sft":
                loss = F.cross_entropy(logits, torch.tensor([sample.label], device=logits.device))
            else:
                loss = margin_dpo_loss(logits, sample.label, args.dpo_beta, args.dpo_ce_weight)
            (loss / args.accum).backward()
            step_loss += float(loss.detach()) / args.accum
            hits.append(float(logits.argmax(dim=-1).item() == sample.label))
        torch.nn.utils.clip_grad_norm_(body_params + list(head.parameters()), 1.0)
        optimizer.step()
        scheduler.step()
        running.append(step_loss)

        if step % 10 == 0 or step == 1:
            lrs = {g["name"]: g["lr"] for g in optimizer.param_groups}
            rec = {
                "step": step,
                "loss": sum(running) / len(running),
                "acc": sum(hits) / max(1, len(hits)),
                "lr_body": lrs["body"],
                "lr_head": lrs["head"],
                "elapsed_s": round(time.time() - t0, 1),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{args.phase} {step}/{args.steps}] loss {rec['loss']:.4f} acc {rec['acc']:.3f} "
                  f"lr {rec['lr_body']:.2e}/{rec['lr_head']:.2e} ({rec['elapsed_s']:.0f}s)")
            running.clear()
            hits.clear()
        if step % args.save_steps == 0:
            save(f"checkpoint-{step}")

    peft_model.eval()
    save("adapter_final")
    head.save(out / "head.pt")
    print(f"[done] {args.phase} -> {out}")


if __name__ == "__main__":
    main()
