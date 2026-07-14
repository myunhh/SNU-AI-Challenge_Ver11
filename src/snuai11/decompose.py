"""Rule-based caption decomposition into exactly 4 event clauses.

Competition rule: model-generated text must not enter training data, so the
decomposition used for training-time pruning is strictly rule-based. The same
deterministic function is used at inference (train = serve consistency).

For Cross-Targeted FitPrune the 4 events only need to COVER the caption
(importance is max-pooled over events), so exact event ordering is
irrelevant here — coverage and separation are what matter.
"""

from __future__ import annotations

import re

# Temporal / sequencing connectives that mark an event boundary. Matched
# case-insensitively. Comma-led variants are handled by the regex below.
_CONNECTIVES = [
    r"and then",
    r"then",
    r"before",
    r"after(?:ward|wards)?",
    r"followed by",
    r"next",
    r"finally",
    r"subsequently",
    r"meanwhile",
    r"eventually",
    r"later",
]

_BOUNDARY_RE = re.compile(
    r"(?:[.;]\s+)|(?:,?\s+(?:" + "|".join(_CONNECTIVES) + r")[,\s]+)",
    flags=re.IGNORECASE,
)

_WS_RE = re.compile(r"\s+")

_STOPWORDS = frozenset(
    """a an the of to in on at as is are was were be been being and or with by
    for from into onto over under his her its their then than that this these
    those it he she they them we you i one ones there here where when while
    who whom whose which what""".split()
)


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text).strip(" ,.;")


def split_clauses(caption: str) -> list[str]:
    """Split a caption at sentence boundaries and temporal connectives."""
    parts = [_clean(p) for p in _BOUNDARY_RE.split(_WS_RE.sub(" ", caption.strip()))]
    return [p for p in parts if p]


def _split_longest(chunks: list[str]) -> list[str]:
    """Split the longest chunk in two (prefer a comma near the middle,
    else the middle word boundary)."""
    i = max(range(len(chunks)), key=lambda k: len(chunks[k]))
    text = chunks[i]
    mid = len(text) // 2
    commas = [m.start() for m in re.finditer(",", text)]
    if commas:
        cut = min(commas, key=lambda c: abs(c - mid))
        left, right = text[:cut], text[cut + 1 :]
    else:
        words = text.split(" ")
        if len(words) < 2:
            left, right = text, text  # degenerate: duplicate rather than emit empty
        else:
            h = len(words) // 2
            left, right = " ".join(words[:h]), " ".join(words[h:])
    left, right = _clean(left), _clean(right)
    return chunks[:i] + [left or text, right or text] + chunks[i + 1 :]


def _merge_shortest_adjacent(chunks: list[str]) -> list[str]:
    """Merge the adjacent pair whose combined length is smallest."""
    i = min(range(len(chunks) - 1), key=lambda k: len(chunks[k]) + len(chunks[k + 1]))
    merged = chunks[i] + ", " + chunks[i + 1]
    return chunks[:i] + [merged] + chunks[i + 2 :]


def decompose_caption(caption: str, n: int = 4) -> list[str]:
    """Deterministically coerce a caption into exactly `n` non-empty events."""
    caption = caption.strip()
    if not caption:
        return ["(no caption)"] * n
    chunks = split_clauses(caption) or [_clean(caption) or caption]
    while len(chunks) < n:
        chunks = _split_longest(chunks)
    while len(chunks) > n:
        chunks = _merge_shortest_adjacent(chunks)
    return chunks


def content_words(text: str) -> list[str]:
    """Lowercased alphanumeric words minus stopwords (for token scoring)."""
    words = re.findall(r"[A-Za-z0-9']+", text.lower())
    kept = [w for w in words if w not in _STOPWORDS and len(w) > 1]
    return kept or words or [text]
