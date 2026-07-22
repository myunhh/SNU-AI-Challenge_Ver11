"""Qwen3-VL engine — loading, Cross-Targeted FitPrune forward, Score24Head.

Verified against transformers 5.12.1 modeling_qwen3_vl.py:
- Qwen3VLModel.forward does NOT accept visual_pos_masks/deepstack_visual_embeds,
  so the pruned pass calls model.model.language_model (Qwen3VLTextModel) directly.
- get_rope_index returns (3, B, L) SEMANTIC m-rope coordinates — index-selecting
  a subset of positions keeps them valid (they are grid coordinates, not
  sequence indices).
- masked_scatter must happen with the FULL placeholder set first
  (get_placeholder_mask asserts token count == feature count); pruning is a
  pure index-select afterwards.
- deepstack features (3 tensors, consumed by decoder layers 0..2) are
  row-major over (B, L) visual positions: the kept-row filter is
  keep[visual_pos_masks].
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from . import perm
from .decompose import content_words, decompose_caption
from .fitprune import PruneConfig, keep_indices_for_image
from .prompting import build_score24_messages

DEFAULT_MAX_PIXELS = 1_126_400  # ~1100 merged visual tokens per image
DEFAULT_MIN_PIXELS = 65_536

# unsloth prequantized checkpoints list bare module names, but transformers
# 5.12.1 should_convert_module matches prefix-dot / exact / endswith — bare
# "visual" does not match "model.visual.blocks.0.attn.qkv", so the vision
# tower would be silently re-quantized to 4bit. Prefixed entries fix that.
_SKIP_PREFIX_FIX = ["model.visual", "model.language_model.embed_tokens", "lm_head"]


def is_prequantized(model_id: str) -> bool:
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_id)
        return getattr(cfg, "quantization_config", None) is not None
    except Exception:
        return False


def _patch_skip_modules(quant_dict: dict) -> None:
    """Mutate a checkpoint's quantization_config dict in place. For
    prequantized checkpoints a user-passed BitsAndBytesConfig is IGNORED by
    AutoHfQuantizer.merge_quantization_configs (bnb has no loading
    attributes), so the fix must go into config.quantization_config itself."""
    skip = list(quant_dict.get("llm_int8_skip_modules") or [])
    for extra in _SKIP_PREFIX_FIX:
        if extra not in skip:
            skip.append(extra)
    quant_dict["llm_int8_skip_modules"] = skip


def verify_vision_not_quantized(model) -> None:
    bad = [
        n
        for n, m in model.model.visual.named_modules()
        if type(m).__name__ in ("Linear4bit", "Linear8bitLt")
    ]
    if bad:
        raise RuntimeError(f"vision tower got quantized (skip-module patch failed): {bad[:5]}")
    w = model.lm_head.weight
    if type(w).__name__ == "Params4bit":
        raise RuntimeError("lm_head got quantized — letter-row head init would be garbage")


def verify_lora_only_on_language(model) -> None:
    lora_sites = [n for n, m in model.named_modules() if hasattr(m, "lora_A")]
    if not lora_sites:
        raise RuntimeError("no LoRA modules found")
    offenders = [n for n in lora_sites if ".language_model.layers." not in n]
    if offenders:
        raise RuntimeError(f"LoRA attached outside language layers: {offenders[:5]}")


def load_model_and_processor(
    model_id: str,
    four_bit: bool = False,
    attn_implementation: str = "sdpa",
    device_map: str | None = "auto",
):
    """Load Qwen3-VL (+processor). Prequantized checkpoints are auto-detected
    and their skip-module list is prefix-patched; `four_bit` quantizes an
    UNQUANTIZED checkpoint on the fly with the same skip conventions."""
    from transformers import AutoConfig, AutoProcessor, Qwen3VLForConditionalGeneration

    cfg = AutoConfig.from_pretrained(model_id)
    quant = getattr(cfg, "quantization_config", None)
    kwargs: dict = {"dtype": torch.bfloat16, "attn_implementation": attn_implementation}
    quantized = quant is not None
    if quantized:
        if four_bit:
            raise ValueError("--four-bit is for unquantized checkpoints only")
        if not isinstance(quant, dict):
            raise TypeError(f"unexpected quantization_config type: {type(quant)}")
        _patch_skip_modules(quant)
        kwargs["config"] = cfg
        kwargs["device_map"] = device_map
    elif four_bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["embed_tokens", "lm_head", "visual", "merger"] + _SKIP_PREFIX_FIX,
        )
        kwargs["device_map"] = device_map

    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    if quantized or four_bit:
        verify_vision_not_quantized(model)
    else:
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


class Score24Head(nn.Module):
    """Follower w of the Stackelberg game: Linear(hidden -> 24), fp32,
    initialized from the (untied, unquantized bf16) lm_head rows of the 24
    letter tokens A..X — at init this reproduces the constrained letter-logit
    scorer, then trains on the fast time scale."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, perm.N_CLASSES, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # h: [B, hidden]
        return self.linear(h.float())

    @classmethod
    def init_from_lm_head(cls, model, letter_ids: list[int]) -> "Score24Head":
        head = cls(model.config.text_config.hidden_size)
        with torch.no_grad():
            rows = model.lm_head.weight[torch.tensor(letter_ids)].float()
            head.linear.weight.copy_(rows)
        return head

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "hidden": self.linear.in_features}, path)

    @classmethod
    def load(cls, path: Path | str, device="cpu") -> "Score24Head":
        blob = torch.load(path, map_location=device, weights_only=True)
        head = cls(blob["hidden"])
        head.load_state_dict(blob["state_dict"])
        return head.to(device)


@dataclass
class Prepared:
    """Everything needed for LLM forwards of one (sample, view) — vision
    tower and m-rope already done, so pruned and full passes share it."""

    inputs_embeds: torch.Tensor  # [1, L, D] (full placeholders scattered)
    position_ids: torch.Tensor  # [3, 1, L]
    attention_mask: torch.Tensor  # [1, L]
    visual_pos_masks: torch.Tensor  # [1, L] bool
    deepstack: list[torch.Tensor]  # 3 x [V, D]
    per_image_embeds: list[torch.Tensor]  # 4 x [tokens_i, D] (scoring input)
    image_positions: list[torch.Tensor]  # 4 x [tokens_i] sequence indices


class Engine:
    """One-pass score24 with optional Cross-Targeted FitPrune."""

    def __init__(self, model, processor, head: Score24Head, prune_cfg: PruneConfig = PruneConfig(),
                 max_pixels: int = DEFAULT_MAX_PIXELS, min_pixels: int = DEFAULT_MIN_PIXELS):
        self.model = model
        self.processor = processor
        self.head = head
        self.prune_cfg = prune_cfg
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self._event_cache: dict[str, list[torch.Tensor]] = {}

    # ---- plumbing -------------------------------------------------------
    @property
    def mm(self):  # Qwen3VLModel
        return self.model.model

    @property
    def lm(self):  # Qwen3VLTextModel
        return self.model.model.language_model

    @property
    def device(self):
        return self.lm.embed_tokens.weight.device

    def encode(self, images, caption: str):
        from PIL import Image

        pil = [Image.open(p).convert("RGB") if not hasattr(p, "mode") else p for p in images]
        messages = build_score24_messages(caption)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.processor(
            text=[text],
            images=pil,
            return_tensors="pt",
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        return {k: v.to(self.device) if hasattr(v, "to") else v for k, v in enc.items()}

    @torch.no_grad()
    def prepare(self, images, caption: str) -> Prepared:
        """Vision tower + scatter + m-rope. no_grad is safe: vision tower and
        embed_tokens are frozen (LoRA lives only in the language layers)."""
        enc = self.encode(images, caption)
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        grid = enc["image_grid_thw"]

        vis = self.mm.get_image_features(enc["pixel_values"], grid, return_dict=True)
        per_image = [t.to(self.device) for t in vis.pooler_output]
        deepstack = [d.to(self.device) for d in vis.deepstack_features]

        inputs_embeds = self.mm.get_input_embeddings()(input_ids)
        image_embeds = torch.cat(per_image, dim=0).to(inputs_embeds.dtype)
        image_mask, _ = self.mm.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        visual_pos_masks = image_mask[..., 0]

        position_ids, _ = self.mm.get_rope_index(
            input_ids,
            enc["mm_token_type_ids"],
            image_grid_thw=grid,
            attention_mask=attention_mask,
        )

        counts = [int(t.shape[0]) for t in per_image]
        vis_pos = visual_pos_masks[0].nonzero().squeeze(1)
        assert vis_pos.numel() == sum(counts), "placeholder/feature count mismatch"
        image_positions = list(vis_pos.split(counts))

        return Prepared(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            visual_pos_masks=visual_pos_masks,
            deepstack=deepstack,
            per_image_embeds=per_image,
            image_positions=image_positions,
        )

    def event_embeds(self, caption: str) -> list[torch.Tensor]:
        """Text-token embeddings (LLM embed space) of the 4 rule-based events,
        stopword-filtered — the cross-target scoring anchors."""
        if caption in self._event_cache:
            return self._event_cache[caption]
        tok = self.processor.tokenizer
        embed = self.lm.embed_tokens
        out = []
        with torch.no_grad():
            for event in decompose_caption(caption):
                words = " ".join(content_words(event))
                ids = tok(words, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)
                out.append(embed(ids)[0].float())
        if len(self._event_cache) > 4096:
            self._event_cache.clear()
        self._event_cache[caption] = out
        return out

    def keep_mask(self, prep: Prepared, caption: str, cfg: PruneConfig) -> torch.Tensor:
        """[1, L] bool — True for all text tokens + selected visual tokens
        (4x4 cross-target scoring, max-pool over events, +diversity)."""
        L = prep.inputs_embeds.shape[1]
        keep = torch.ones(1, L, dtype=torch.bool, device=self.device)
        if not cfg.enabled or cfg.keep_ratio >= 1.0:
            return keep
        events = self.event_embeds(caption)
        for img_i in range(len(prep.per_image_embeds)):
            kept_local = keep_indices_for_image(prep.per_image_embeds, img_i, events, cfg)
            positions = prep.image_positions[img_i]
            keep[0, positions] = False
            keep[0, positions[kept_local]] = True
        return keep

    def forward_prepared(self, prep: Prepared, keep: torch.Tensor | None = None) -> torch.Tensor:
        """LLM forward -> score24 logits [1, 24]. keep=None means full tokens."""
        if keep is None:
            embeds, pos, attn = prep.inputs_embeds, prep.position_ids, prep.attention_mask
            vmask, ds = prep.visual_pos_masks, prep.deepstack
        else:
            idx = keep[0].nonzero().squeeze(1)
            embeds = prep.inputs_embeds[:, idx]
            pos = prep.position_ids[:, :, idx]
            attn = prep.attention_mask[:, idx]
            vmask = prep.visual_pos_masks[:, idx]
            kept_rows = keep[prep.visual_pos_masks]  # row-major over V
            ds = [d[kept_rows] for d in prep.deepstack]

        out = self.lm(
            inputs_embeds=embeds,
            position_ids=pos,
            attention_mask=attn,
            visual_pos_masks=vmask,
            deepstack_visual_embeds=ds,
            use_cache=False,
        )
        h = out.last_hidden_state  # [1, L', D] post final RMSNorm
        last = attn.sum(dim=1) - 1
        pooled = h[torch.arange(h.shape[0], device=h.device), last]
        return self.head(pooled)

    def score24(self, images, caption: str, prune: bool = True) -> torch.Tensor:
        """Convenience one-shot: [24] logits."""
        prep = self.prepare(images, caption)
        keep = self.keep_mask(prep, caption, self.prune_cfg) if prune else None
        return self.forward_prepared(prep, keep)[0]


def letter_ids_for(tokenizer) -> list[int]:
    from .fsm import letter_token_ids

    return letter_token_ids(tokenizer)


def write_env_report(out_dir: Path | str) -> None:
    import platform

    import transformers

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    try:
        import bitsandbytes

        info["bitsandbytes"] = bitsandbytes.__version__
    except Exception:
        pass
    try:
        import peft

        info["peft"] = peft.__version__
    except Exception:
        pass
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "env.json").write_text(json.dumps(info, indent=2))
