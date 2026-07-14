#!/usr/bin/env python
"""GPU smoke — validates the surgical pruning path before any long run.

Checks:
  1. 24 letter answer tokens are distinct single tokens
  2. PARITY: stock Qwen3VLModel.forward vs our surgical path (keep=None)
     produce identical last_hidden_state (max|diff| == 0 expected)
  3. keep=all-True equals keep=None on score24 logits
  4. pruned forward (50% keep): shorter sequence, finite logits
  5. --train: 3 Stackelberg optimizer steps — loss finite, back>0 on both
     the LoRA body and the score24 head
  6. per-forward timing

Local 4090:  python scripts/smoke_gpu.py --model-id /home/yhmin/model/hub/Qwen3-VL-8B-Instruct --four-bit --train
A100 32B :   python scripts/smoke_gpu.py --train
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--keep-ratio", type=float, default=0.5)
    ap.add_argument("--max-pixels", type=int, default=None)
    args = ap.parse_args()

    from run_common import ensure_data, resolve_model_id

    from snuai11.data import load_samples, uniform_augment
    from snuai11.fitprune import PruneConfig
    from snuai11.fsm import letter_token_ids
    from snuai11.vlm import DEFAULT_MAX_PIXELS, Engine, Score24Head, load_model_and_processor

    model_id = args.model_id or resolve_model_id()
    data_root = ensure_data("data")
    samples = load_samples(data_root, "train")[:3]

    print(f"[smoke] loading {model_id} (four_bit={args.four_bit})")
    model, processor = load_model_and_processor(model_id, four_bit=args.four_bit)

    # 1. letter tokens
    letter_ids = letter_token_ids(processor.tokenizer)
    assert len(set(letter_ids)) == 24
    print(f"[1/6 OK] 24 distinct single-token letters, e.g. A -> {letter_ids[0]}")

    head = Score24Head.init_from_lm_head(model, letter_ids).to(model.lm_head.weight.device).eval()
    prune_cfg = PruneConfig(keep_ratio=args.keep_ratio)
    engine = Engine(model, processor, head, prune_cfg,
                    max_pixels=args.max_pixels or DEFAULT_MAX_PIXELS)

    s = samples[0]
    with torch.no_grad():
        # 2. parity stock vs surgical
        enc = engine.encode(s.image_paths, s.caption)
        stock = engine.mm(
            input_ids=enc["input_ids"],
            pixel_values=enc["pixel_values"],
            image_grid_thw=enc["image_grid_thw"],
            mm_token_type_ids=enc["mm_token_type_ids"],
            attention_mask=enc["attention_mask"],
        ).last_hidden_state

        prep = engine.prepare(s.image_paths, s.caption)
        out_full = engine.lm(
            inputs_embeds=prep.inputs_embeds,
            position_ids=prep.position_ids,
            attention_mask=prep.attention_mask,
            visual_pos_masks=prep.visual_pos_masks,
            deepstack_visual_embeds=prep.deepstack,
            use_cache=False,
        ).last_hidden_state
        diff = (stock - out_full).abs().max().item()
        print(f"[2/6] parity stock-vs-surgical max|diff| = {diff:.3e}")
        assert diff < 1e-3, "surgical path diverges from stock forward"

        # 3. keep=None == keep=all
        keep_all = torch.ones_like(prep.visual_pos_masks, dtype=torch.bool)
        l_none = engine.forward_prepared(prep, None)[0]
        l_all = engine.forward_prepared(prep, keep_all)[0]
        d2 = (l_none - l_all).abs().max().item()
        print(f"[3/6] keep=None vs keep=all logits max|diff| = {d2:.3e}")
        assert d2 < 1e-3

        # 4. pruned forward
        L_full = prep.inputs_embeds.shape[1]
        keep = engine.keep_mask(prep, s.caption, prune_cfg)
        L_kept = int(keep.sum())
        t0 = time.time()
        l_pruned = engine.forward_prepared(prep, keep)[0]
        t_pruned = time.time() - t0
        n_vis = int(prep.visual_pos_masks.sum())
        n_vis_kept = int(keep[prep.visual_pos_masks].sum())
        assert torch.isfinite(l_pruned).all()
        print(f"[4/6] seq {L_full} -> {L_kept} (visual {n_vis} -> {n_vis_kept}, "
              f"{n_vis_kept/n_vis:.1%}); pruned forward {t_pruned:.2f}s; "
              f"top1={int(l_pruned.argmax())} margin_logit={float(l_pruned.topk(2).values.diff().abs()):.3f}")

        # timing full vs pruned
        t0 = time.time()
        engine.forward_prepared(prep, None)
        t_full = time.time() - t0
        print(f"[6/6] full forward {t_full:.2f}s vs pruned {t_pruned:.2f}s")

    if args.train:
        import random

        import torch.nn.functional as F

        from snuai11.stackelberg import StackelbergConfig, build_optimizer, build_param_groups
        from snuai11.train_sft import attach_lora

        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        peft_model = attach_lora(model, r=8, alpha=16, dropout=0.0, adapter=None)
        head.train().requires_grad_(True)
        body = [p for n, p in peft_model.named_parameters() if "lora_" in n and p.requires_grad]
        scfg = StackelbergConfig()
        opt = build_optimizer(build_param_groups(body, list(head.parameters()), scfg), scfg)
        rng = random.Random(0)
        peft_model.train()
        for step in range(3):
            sample = uniform_augment(samples[step % len(samples)], rng)
            prep = engine.prepare(sample.image_paths, sample.caption)
            keep = engine.keep_mask(prep, sample.caption, prune_cfg)
            logits = engine.forward_prepared(prep, keep)
            loss = F.cross_entropy(logits, torch.tensor([sample.label], device=logits.device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gb = sum(float(p.grad.abs().sum()) for p in body if p.grad is not None)
            gh = sum(float(p.grad.abs().sum()) for p in head.parameters() if p.grad is not None)
            opt.step()
            print(f"[train {step}] loss={float(loss):.4f} grad_body={gb:.3e} grad_head={gh:.3e}")
            assert torch.isfinite(loss) and gb > 0 and gh > 0, "no training signal (back=0)"
        print("[train OK] back>0 on body and head")

    if torch.cuda.is_available():
        print(f"[vram] peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
    print("SMOKE PASSED")


if __name__ == "__main__":
    main()
