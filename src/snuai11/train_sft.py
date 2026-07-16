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
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from . import perm
from .data import load_samples, uniform_augment
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


# ===========================================================================
# DDP 헬퍼 (Ver10/grpo.py와 같은 패턴) — torchrun의 RANK/LOCAL_RANK/WORLD_SIZE 기준
# ===========================================================================

def dist_env() -> tuple[int, int, int]:
    """(rank, local_rank, world_size). torchrun 없이 단일 프로세스면 (0, 0, 1)."""
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws <= 1:
        return 0, 0, 1
    return int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), ws


def init_distributed() -> tuple[int, int, int]:
    """world_size>1이면 NCCL 프로세스그룹 초기화 + 이 프로세스의 GPU 고정.

    timeout 30분: 첫 스텝 전 32B 모델 로딩(캐시 미스 시 다운로드 포함)이 rank마다
    편차가 있어 먼저 끝난 rank가 첫 collective에서 대기할 수 있다. 기본 10분보다
    넉넉히 잡되, 진짜 교착은 30분이면 드러난다."""
    rank, local_rank, world_size = dist_env()
    if world_size > 1:
        import torch.distributed as dist
        from datetime import timedelta
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    return rank, local_rank, world_size


def sync_grads_sum(params, world_size: int) -> None:
    """스텝당 1회, opt.step() 직전 all-reduce(SUM, 나누기 없음).

    (loss/args.accum)로 이미 전체(global) accum 기준 스케일이라, rank들의 부분합을
    그냥 더하면 단일GPU 직렬실행과 동일한 그래디언트가 된다(world_size로 다시
    나누면 과소평가). DDP wrapper를 안 쓰는 건 Ver10/grpo.py와 같은 이유는 아니고
    (여기는 backward 호출횟수가 매 스텝 항상 local_accum으로 고정, 조건부 스킵
    없음) 단순히 두 트레이너의 DDP 방식을 통일해 관리 부담을 줄이기 위함.
    """
    if world_size <= 1:
        return
    import torch.distributed as dist
    for p in params:
        if p.requires_grad and p.grad is not None:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)


def local_accum_for(accum: int, world_size: int) -> int:
    """--accum을 world_size로 나눈 rank당 몫. 나누어떨어지지 않으면 즉시 에러."""
    if accum % world_size != 0:
        raise ValueError(f"--accum({accum})은 world_size({world_size})로 나누어떨어져야 함 "
                         "(rank마다 동일한 수의 샘플을 처리)")
    return accum // world_size


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
    args = ap.parse_args(argv)

    rank, local_rank, world_size = init_distributed()
    is_main = rank == 0

    if args.model_id is None:
        from run_common import resolve_model_id

        args.model_id = resolve_model_id()
    if args.phase == "dpo" and not args.adapter:
        ap.error("--phase dpo requires --adapter (the SFT output)")
    if args.steps is None:
        args.steps = 2000 if args.phase == "sft" else 400
    if args.body_lr is None:
        args.body_lr = 2e-4 if args.phase == "sft" else 5e-5
    try:
        local_accum = local_accum_for(args.accum, world_size)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    out = Path(args.out or f"runs/{'sft' if args.phase == 'sft' else 'dpo'}32b_v11")
    out.mkdir(parents=True, exist_ok=True)
    if is_main:
        write_env_report(out)

    # body/head 파라미터 초기화(fresh LoRA 시 랜덤)가 rank 간 동일하도록 전역 시드 고정.
    # --adapter로 로드하는 경우는 파일이 이미 결정적이라 무해.
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- data ----------------------------------------------------------
    # No local holdout carve-out: tuning decisions were already validated
    # against the Ver4/Ver10 track; Ver11 trains on the full train set and
    # the design is judged via LB slots (see CLAUDE.md).
    train_samples = load_samples(args.data_root, "train")
    if is_main:
        print(f"[data] train {len(train_samples)} (100%, no local holdout)")
        if world_size > 1:
            print(f"[dist] world_size={world_size} — accum={args.accum}는 rank당 {local_accum}개로 분할")

    # ---- model ---------------------------------------------------------
    model, processor = load_model_and_processor(
        args.model_id, four_bit=args.four_bit, device_map={"": local_rank})
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
    if is_main:
        print(f"[stackelberg] body {n_body/1e6:.1f}M @ {scfg.body_lr:g} | head {sum(p.numel() for p in head.parameters())} @ {scfg.head_lr:g} (wd {scfg.head_weight_decay})")

    # ---- loop ------------------------------------------------------------
    peft_model.train()
    # rank마다 다른 데이터를 보게 시드를 갈라친다(전역 random.seed와는 별개 인스턴스 —
    # 위쪽 model/LoRA 초기화 동기화용 전역 시드는 이미 다 쓰고 지나온 뒤라 여기서 갈라도 무해).
    rng = random.Random(args.seed + rank)
    log_path = out / "train_log.jsonl"
    pool: list[int] = []
    running: list[float] = []
    hits: list[float] = []
    t0 = time.time()
    all_params = body_params + list(head.parameters())

    def save(tag: str) -> None:
        if is_main:
            ckpt = out / tag
            peft_model.save_pretrained(str(ckpt / "adapter"))
            head.save(ckpt / "head.pt")
            (ckpt / "train_args.json").write_text(json.dumps(vars(args), indent=2, default=str))
        if world_size > 1:
            import torch.distributed as dist
            dist.barrier()   # rank0 저장 끝날 때까지 다른 rank 대기

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(local_accum):
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
        sync_grads_sum(all_params, world_size)
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        scheduler.step()
        running.append(step_loss)

        log_boundary = step % 10 == 0 or step == 1
        if log_boundary and world_size > 1:
            # gather는 로그 시점(윈도우당 1회)에만 — 매 스텝 부르면 rank0의 running이
            # 매번 다른 rank의 "그 시점까지 누적된 로컬 리스트" 전체를 다시 흡수해
            # O(n^2)로 부풀며 초반 값이 중복 반영되는 버그가 있었다(2026-07-16 발견).
            import torch.distributed as dist
            gathered_running: list[object] = [None] * world_size
            dist.all_gather_object(gathered_running, running)
            gathered_hits: list[object] = [None] * world_size
            dist.all_gather_object(gathered_hits, hits)
            if is_main:
                running = [x for part in gathered_running for x in part]
                hits = [x for part in gathered_hits for x in part]

        if log_boundary and is_main:
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
        if log_boundary:
            # rank0가 아니어도 로컬 윈도우를 비워야 all_gather 크기가 매 스텝 재사용됨
            # (rank0와 같은 10스텝 윈도우 경계에서 함께 리셋).
            running.clear()
            hits.clear()
        if step % args.save_steps == 0:
            save(f"checkpoint-{step}")

    peft_model.eval()
    save("adapter_final")
    if is_main:
        head.save(out / "head.pt")
        print(f"[done] {args.phase} -> {out}")


if __name__ == "__main__":
    main()
