#!/usr/bin/env python
"""Visualize Cross-Targeted FitPrune — the 4 (images) x 4 (caption events)
score heatmaps plus the actual kept/pruned patch mask fed to the LLM.

Merged-token spatial mapping: the Qwen2VL/Qwen3VL image processor already
groups every 2x2 (merge_size) patch block contiguously in `pixel_values`
(image_processing_qwen2_vl.py: patches.permute(...).reshape(grid_h*grid_w, ...)
groups by merge-block, sub-row, sub-col), and Qwen3VLVisionPatchMerger just
does a contiguous view(-1, hidden*merge**2) — no window-index permutation in
this model (unlike Qwen2.5-VL). So merged token k is exactly row-major over
the (grid_h/2, grid_w/2) merge-block grid: k = row*(grid_w/2) + col. This was
verified by reading transformers==5.12.1 source, not assumed.

Usage (needs the GPU free — loads the full model to run the vision tower):
  Local 4090:  python scripts/visualize_prune.py \
                 --model-id /home/yhmin/model/hub/Qwen3-VL-8B-Instruct --four-bit
  A100 32B  :  python scripts/visualize_prune.py
Flags: --n 3 (samples) --start 0 --keep-ratio 0.5 --diversity-frac 0.2
       --out runs/prune_viz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

PATCH_SIZE = 16  # Qwen3-VL preprocessor_config.json: patch_size
MERGE = 2  # merge_size

# Presentation palette (dataviz skill reference palette — validated, CVD-safe).
# Sequential magnitude = single blue hue, alpha-ramped (not the paper's jet
# colormap: jet is not colorblind-safe and reads as "louder" than the data).
HEAT_HUE = (42, 120, 214)  # #2a78d6 — sequential blue, categorical slot 1
ACCENT = (42, 120, 214)  # kept-cell border, same blue for one consistent meaning
FADE_GRAY = (137, 135, 129)  # #898781 — muted ink; pruned cells fade toward this
GRID_LINE = (225, 224, 217)  # #e1e0d9 — hairline, recedes into the surface
INK = (11, 11, 11)  # #0b0b0b — primary text

_CJK_FONT_CANDIDATES = [
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 1),  # KR subface
    ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 0),
]


def load_font(size: int = 16):
    """PIL's bitmap default font is ASCII-only and renders Korean labels as
    tofu boxes — load a real CJK-capable TTF/TTC so presentation text is
    legible. Falls back to the bitmap default (Latin-only) if none found."""
    from PIL import ImageFont

    for path, index in _CJK_FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size, index=index)
            except Exception:
                continue
    return ImageFont.load_default()


def event_score_grids(visual: torch.Tensor, event_embeds, cfg, h2: int, w2: int) -> np.ndarray:
    """Per-event importance maps on the (h2, w2) merged-token grid, computed
    in ONE call with the full event list so the text-anchor mean mu_T matches
    the selection path exactly (re-scoring a single event alone would center
    by that event's own mean — a different geometry than what selection saw).
    [E, h2, w2] numpy, each event min-max normalized to [0, 1]."""
    from snuai11.fitprune import per_event_scores

    s = per_event_scores(visual, event_embeds, cfg).cpu().numpy()  # [E, N]
    s = s.reshape(s.shape[0], h2, w2)
    lo = s.min(axis=(1, 2), keepdims=True)
    hi = s.max(axis=(1, 2), keepdims=True)
    return (s - lo) / (hi - lo + 1e-8)


def overlay(resized_img: Image.Image, grid01: np.ndarray, kept_mask: np.ndarray | None,
            cell: int = MERGE * PATCH_SIZE) -> Image.Image:
    """Upsample the (h2, w2) score grid to pixel resolution.

    kept_mask=None: sequential-magnitude mode — single-hue blue wash, alpha
    ramped by score (transparent = low importance, saturated blue = high).
    kept_mask given: status mode — kept cells keep full original color, pruned
    cells fade toward a neutral gray (context stays legible, nothing goes
    black or blank) — this is the "what the model actually sees" panel."""
    base = resized_img.convert("RGB")
    W, H = base.size
    arr = np.array(base).astype(np.float32)

    up = np.kron(grid01, np.ones((cell, cell)))[:H, :W]
    if kept_mask is None:
        heat = np.tile(np.array(HEAT_HUE, dtype=np.float32), (*up.shape, 1))
        alpha = (up * 0.7)[..., None]
        out = arr * (1 - alpha) + heat * alpha
    else:
        keep_up = np.kron(kept_mask.astype(np.float32), np.ones((cell, cell)))[:H, :W]
        gray = arr.mean(axis=-1, keepdims=True)
        faded = 0.35 * gray + 0.65 * np.array(FADE_GRAY, dtype=np.float32)
        out = arr * keep_up[..., None] + faded * (1 - keep_up[..., None])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def draw_grid_lines(img: Image.Image, h2: int, w2: int, kept_mask: np.ndarray | None = None,
                     cell: int = MERGE * PATCH_SIZE) -> Image.Image:
    """Faint hairlines everywhere; if kept_mask given, a 2px accent-blue box
    around every kept cell so the retained region pops at a glance/distance."""
    from PIL import ImageDraw

    img = img.copy()
    d = ImageDraw.Draw(img)
    W, H = img.size
    for r in range(h2 + 1):
        d.line([(0, r * cell), (W, r * cell)], fill=GRID_LINE, width=1)
    for c in range(w2 + 1):
        d.line([(c * cell, 0), (c * cell, H)], fill=GRID_LINE, width=1)
    if kept_mask is not None:
        for r in range(h2):
            for c in range(w2):
                if kept_mask[r, c]:
                    x0, y0, x1, y1 = c * cell, r * cell, (c + 1) * cell, (r + 1) * cell
                    d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=ACCENT, width=2)
    return img


def truncate_to_width(text: str, font, max_width: int, draw) -> str:
    """Shrink `text` with a trailing ellipsis until it fits max_width pixels,
    measured with the actual font (character width varies a lot between
    Latin and Hangul, so a fixed char-count cutoff either clips or wastes
    space depending on script)."""
    if draw.textlength(text, font=font) <= max_width:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if draw.textlength(text[:mid] + "…", font=font) <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + "…" if lo > 0 else "…"


def make_panel(title: str, img: Image.Image, font=None) -> Image.Image:
    from PIL import ImageDraw

    pad = 28
    canvas = Image.new("RGB", (img.width, img.height + pad), (255, 255, 255))
    canvas.paste(img, (0, pad))
    d = ImageDraw.Draw(canvas)
    title = truncate_to_width(title, font, img.width - 8, d)
    d.text((4, 6), title, fill=INK, font=font)
    return canvas


def make_caption(events: list[str], font, width: int) -> Image.Image:
    """One-time header listing the 4 full event texts (E1..E4) at natural
    reading width — avoids repeating (and clipping) the same event text in
    every 260px-wide panel across all 4 image rows."""
    from PIL import ImageDraw

    line_h = 22
    img = Image.new("RGB", (width, line_h * len(events) + 6), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for i, ev in enumerate(events):
        text = truncate_to_width(f"E{i+1}: {ev}", font, width - 8, d)
        d.text((4, 4 + i * line_h), text, fill=INK, font=font)
    return img


def make_legend(font=None, width: int = 900) -> Image.Image:
    """One legend strip per sample: what blue-wash / blue-box / gray mean.
    Laid out left-to-right by measured text width so it never clips,
    regardless of how many panels the row above ends up being."""
    from PIL import ImageDraw

    img = Image.new("RGB", (width, 28), (255, 255, 255))
    d = ImageDraw.Draw(img)
    x = 0
    swatch, gap_after_swatch, gap_after_label = 16, 4, 28
    entries = [
        (tuple(int(0.3 * 255 + 0.7 * v) for v in HEAT_HUE), None, "= 이벤트 중요도 (진할수록 높음)"),
        (None, ACCENT, "= kept 패치"),
        (FADE_GRAY, None, "= pruned 패치 (페이드)"),
    ]
    for fill, outline, label in entries:
        box = [x, 8, x + swatch, 24]
        if fill is not None:
            d.rectangle(box, fill=fill)
        else:
            d.rectangle(box, outline=outline, width=2)
        x += swatch + gap_after_swatch
        d.text((x, 6), label, fill=INK, font=font)
        x += d.textlength(label, font=font) + gap_after_label
    return img


def hstack(panels: list[Image.Image], gap: int = 6) -> Image.Image:
    h = max(p.height for p in panels)
    w = sum(p.width for p in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + gap
    return canvas


def vstack(panels: list[Image.Image], gap: int = 10) -> Image.Image:
    w = max(p.width for p in panels)
    h = sum(p.height for p in panels) + gap * (len(panels) - 1)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    y = 0
    for p in panels:
        canvas.paste(p, (0, y))
        y += p.height + gap
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--four-bit", action="store_true")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--keep-ratio", type=float, default=0.5)
    ap.add_argument("--diversity-frac", type=float, default=0.2)
    ap.add_argument("--objectness-weight", type=float, default=0.3)
    ap.add_argument("--mmr-lambda", type=float, default=0.5)
    ap.add_argument("--motion-weight", type=float, default=0.0,
                    help="cross-frame residual-norm blend weight (0 = pre-motion behavior)")
    ap.add_argument("--max-pixels", type=int, default=None)
    ap.add_argument("--out", default="runs/prune_viz")
    args = ap.parse_args()

    from run_common import ensure_data, resolve_model_id

    from snuai11.data import load_samples
    from snuai11.decompose import decompose_caption
    from snuai11.fitprune import PruneConfig
    from snuai11.fsm import letter_token_ids
    from snuai11.vlm import DEFAULT_MAX_PIXELS, Engine, Score24Head, load_model_and_processor

    model_id = args.model_id or resolve_model_id()
    data_root = ensure_data("data")
    samples = load_samples(data_root, "train")[args.start : args.start + args.n]

    print(f"[viz] loading {model_id} (four_bit={args.four_bit})")
    model, processor = load_model_and_processor(model_id, four_bit=args.four_bit)
    letter_ids = letter_token_ids(processor.tokenizer)
    head = Score24Head.init_from_lm_head(model, letter_ids).to(model.lm_head.weight.device).eval()
    cfg = PruneConfig(keep_ratio=args.keep_ratio, diversity_frac=args.diversity_frac,
                      objectness_weight=args.objectness_weight, mmr_lambda=args.mmr_lambda,
                      motion_weight=args.motion_weight)
    engine = Engine(model, processor, head, cfg, max_pixels=args.max_pixels or DEFAULT_MAX_PIXELS)

    font = load_font(16)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for s in samples:
        print(f"[viz] sample {s.id}: {s.caption!r}")
        events = decompose_caption(s.caption)
        event_embeds = engine.event_embeds(s.caption)

        enc = engine.encode(s.image_paths, s.caption)
        grid_thw = enc["image_grid_thw"].cpu()  # [4, 3] (t, h, w) in patch units

        with torch.no_grad():
            prep = engine.prepare(s.image_paths, s.caption)
            keep_mask_full = engine.keep_mask(prep, s.caption, cfg)[0]  # [L] bool

        rows = []
        for img_i, path in enumerate(s.image_paths):
            t, h, w = (int(x) for x in grid_thw[img_i])
            assert t == 1, f"video-mode grid_t={t} unexpected for still image"
            h2, w2 = h // MERGE, w // MERGE
            resized = Image.open(path).convert("RGB").resize((w * PATCH_SIZE, h * PATCH_SIZE), Image.BICUBIC)

            visual = prep.per_image_embeds[img_i]  # [h2*w2, D]
            positions = prep.image_positions[img_i]
            kept_local = keep_mask_full[positions].cpu().numpy().astype(bool)  # [h2*w2] row-major
            kept_grid = kept_local.reshape(h2, w2)
            n_kept, n_tot = int(kept_local.sum()), kept_local.size
            assert n_tot == h2 * w2

            grids = event_score_grids(visual, event_embeds, cfg, h2, w2)
            panels = []
            for ev_i in range(len(events)):
                ov = draw_grid_lines(overlay(resized, grids[ev_i], None), h2, w2)
                ov.thumbnail((260, 260))
                panels.append(make_panel(f"img{img_i+1} · E{ev_i+1}", ov, font))

            from snuai11.fitprune import objectness_scores
            obj = objectness_scores(visual).cpu().numpy().reshape(h2, w2)
            obj = (obj - obj.min()) / (obj.max() - obj.min() + 1e-8)
            ov = draw_grid_lines(overlay(resized, obj, None), h2, w2)
            ov.thumbnail((260, 260))
            panels.append(make_panel(f"img{img_i+1} OBJ (w={cfg.objectness_weight})", ov, font))

            if cfg.motion_weight > 0.0:
                from snuai11.fitprune import motion_scores

                mot = motion_scores(prep.per_image_embeds, img_i)
                if mot is not None:
                    mot = mot.cpu().numpy().reshape(h2, w2)
                    mot = (mot - mot.min()) / (mot.max() - mot.min() + 1e-8)
                    ov = draw_grid_lines(overlay(resized, mot, None), h2, w2)
                    ov.thumbnail((260, 260))
                    panels.append(make_panel(f"img{img_i+1} MOTION (w={cfg.motion_weight})", ov, font))
                else:
                    blank = Image.new("RGB", (resized.width, resized.height), FADE_GRAY)
                    blank.thumbnail((260, 260))
                    panels.append(make_panel(f"img{img_i+1} MOTION (grid 불일치, 폴백)", blank, font))

            final = draw_grid_lines(overlay(resized, np.ones((h2, w2)), kept_grid), h2, w2, kept_mask=kept_grid)
            final.thumbnail((260, 260))
            panels.append(make_panel(f"img{img_i+1} KEPT {n_kept}/{n_tot} ({n_kept/n_tot:.0%})", final, font))

            rows.append(hstack(panels))

        row_width = max(r.width for r in rows)
        legend = make_legend(font, width=row_width)
        caption = make_caption(events, font, width=row_width)
        grid_img = vstack([legend, caption] + rows)
        dst = out_dir / f"{s.id}.png"
        grid_img.save(dst)
        print(f"[viz]   -> {dst}  ({grid_img.width}x{grid_img.height})")

    print(f"[viz] done: {len(samples)} sample(s) in {out_dir}/")


if __name__ == "__main__":
    main()
