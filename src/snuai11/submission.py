"""Kaggle submission formatting — Answer is a RANK string with spaces:
"[1, 2, 3, 4]" means Input_i is the Answer[i-1]-th frame chronologically."""

from __future__ import annotations

import csv
from pathlib import Path

from . import perm


def format_answer(rank: perm.Perm) -> str:
    return "[" + ", ".join(str(r + 1) for r in rank) + "]"


def parse_answer(raw: str) -> perm.Perm:
    digits = [int(c) - 1 for c in str(raw) if c.isdigit()]
    if len(digits) != 4 or sorted(digits) != [0, 1, 2, 3]:
        raise ValueError(f"Answer {raw!r} is not a permutation of 1-4")
    return tuple(digits)  # type: ignore[return-value]


def write_submission(
    rows: list[tuple[str, perm.Perm]],
    out_path: Path | str,
    sample_submission: Path | str | None = None,
) -> Path:
    """Write submission.csv; if sample_submission given, enforce identical
    id set and row order."""
    out_path = Path(out_path)
    by_id = {i: r for i, r in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate ids in submission rows")

    if sample_submission is not None:
        with open(sample_submission, newline="", encoding="utf-8-sig") as f:
            sample_ids = [row["Id"] for row in csv.DictReader(f)]
        missing = [i for i in sample_ids if i not in by_id]
        extra = [i for i in by_id if i not in set(sample_ids)]
        if missing or extra:
            raise ValueError(
                f"id mismatch vs sample_submission: {len(missing)} missing, {len(extra)} extra"
            )
        ordered = sample_ids
    else:
        ordered = [i for i, _ in rows]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Id", "Answer"])
        for i in ordered:
            w.writerow([i, format_answer(by_id[i])])
    return out_path
