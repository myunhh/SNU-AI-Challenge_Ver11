"""Prompt SSOT — training and inference must build byte-identical prompts.

One-pass score24: the model reads 4 labeled images + the caption + a legend
mapping the 24 letters A..X to rank tuples, and the answer position is the
single next token after the assistant generation prompt. The Score24Head
reads the last hidden state at that position.
"""

from __future__ import annotations

from . import perm

SYSTEM_PROMPT = (
    "You are an expert at temporal reasoning over video frames. "
    "Given shuffled frames and the storyline, you determine each frame's "
    "chronological position."
)


def score24_legend() -> str:
    """Legend text derived from perm SSOT (never hand-written)."""
    lines = []
    for c in range(perm.N_CLASSES):
        r = perm.rank_of_index(c)
        shown = "[" + ", ".join(str(x + 1) for x in r) + "]"
        lines.append(f"{perm.letter_of_index(c)} = {shown}")
    return "\n".join(lines)


def instruction_text(caption: str, legend: bool = True) -> str:
    parts = [
        "The four images above are video frames in shuffled order.",
        f'Storyline: "{caption.strip()}"',
        (
            "Determine each image's chronological position. The answer is a "
            "letter whose legend entry [r1, r2, r3, r4] means Image i is the "
            "ri-th frame in time."
        ),
    ]
    if legend:
        parts.append("Legend:\n" + score24_legend())
    parts.append("Answer with exactly one letter (A-X).")
    return "\n\n".join(parts)


def build_score24_messages(caption: str, legend: bool = True) -> list[dict]:
    """Chat messages with 4 image placeholders, ready for
    processor.apply_chat_template(..., add_generation_prompt=True)."""
    content: list[dict] = []
    for i in range(4):
        content.append({"type": "text", "text": f"Image {i + 1}:"})
        content.append({"type": "image"})
    content.append({"type": "text", "text": instruction_text(caption, legend=legend)})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
