"""Data loading, uniform permutation augmentation.

Layout: root/{train,test}.csv + root/{train,test}/<Id>/<4 images>.
Answer column IS the rank tuple (1-indexed): rank[i] = Answer[i+1]-1.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path

from . import perm
from .submission import parse_answer

INPUT_COLUMNS = ["Input_1", "Input_2", "Input_3", "Input_4"]


@dataclass(frozen=True)
class Sample:
    id: str
    image_paths: tuple[Path, Path, Path, Path]  # Input_1..4 (shuffled order)
    caption: str
    rank: perm.Perm | None  # None for test rows
    label: int | None  # perm.index_of(rank)


def _resolve_image(folder: Path, value: str) -> Path:
    value = str(value).strip()
    for cand in (folder / value, folder / f"{value}.jpg", folder / f"{value}.png"):
        if cand.exists():
            return cand
    matches = list(folder.glob(f"*{value}*"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"cannot resolve image {value!r} under {folder}")


def load_samples(root: Path | str, split: str) -> list[Sample]:
    import pandas as pd

    root = Path(root)
    df = pd.read_csv(root / f"{split}.csv", dtype=str)
    image_root = root / split

    out: list[Sample] = []
    for row in df.to_dict(orient="records"):
        sid = row["Id"]
        folder = image_root / sid
        paths = tuple(_resolve_image(folder, row[c]) for c in INPUT_COLUMNS)
        raw = row.get("Answer")
        if raw is not None and isinstance(raw, str) and raw.strip():
            rank = parse_answer(raw)
            label = perm.index_of(rank)
        else:
            rank, label = None, None
        out.append(
            Sample(id=sid, image_paths=paths, caption=row["Sentence"], rank=rank, label=label)  # type: ignore[arg-type]
        )
    return out


def shuffle_sample(sample: Sample, sigma: perm.Perm) -> Sample:
    """Re-shuffle a labeled sample by view permutation sigma.

    New slot j shows original input sigma[j]; the label transforms as
    rank_new = rank_orig o sigma (see perm.apply_view).
    """
    if sample.rank is None:
        raise ValueError("cannot shuffle an unlabeled sample")
    new_paths = tuple(sample.image_paths[sigma[j]] for j in range(4))
    new_rank = perm.apply_view(sample.rank, sigma)
    return replace(
        sample,
        image_paths=new_paths,  # type: ignore[arg-type]
        rank=new_rank,
        label=perm.index_of(new_rank),
    )


def uniform_augment(sample: Sample, rng: random.Random) -> Sample:
    """Uniform permutation augmentation (the proven default)."""
    sigma = perm.ALL_PERMS[rng.randrange(perm.N_CLASSES)]
    return shuffle_sample(sample, sigma)
