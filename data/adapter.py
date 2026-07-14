"""
Raw competition data -> structured Sample list.

Expected layout under `root`:
    root/train.csv, root/test.csv
    root/train/<Id>/...4 images..., root/test/<Id>/...4 images...

CSV columns: Id, Input_1, Input_2, Input_3, Input_4, Sentence, Answer
(test.csv has no Answer column). train.csv also has a `No_ordering` column
that is 100% redundant with Answer == identity (verified on the real data:
1478/1478 rows agree both directions) -- not surfaced separately here.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

INPUT_COLUMNS = ["Input_1", "Input_2", "Input_3", "Input_4"]

ALL_PERMUTATIONS: list[tuple[int, int, int, int]] = list(itertools.permutations(range(4)))
PERMUTATION_TO_INDEX: dict[tuple[int, int, int, int], int] = {
    perm: i for i, perm in enumerate(ALL_PERMUTATIONS)
}


@dataclass(frozen=True)
class Sample:
    id: str
    image_paths: tuple[Path, Path, Path, Path]  # Input_1..Input_4, in given (shuffled) order
    sentence: str
    # order[t] = index (0-3) into image_paths of the image at chronological step t.
    # order == (0,1,2,3) means the given order is already correct. None for test rows.
    order: tuple[int, int, int, int] | None
    permutation_index: int | None  # ALL_PERMUTATIONS.index(order); None for test rows


def _resolve_image(folder: Path, value: str) -> Path:
    value = str(value).strip()
    for candidate in (
        folder / value,
        folder / f"{value}.jpg",
        folder / f"{value}.jpeg",
        folder / f"{value}.png",
    ):
        if candidate.exists():
            return candidate
    matches = list(folder.glob(f"*{value}*"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"Could not resolve image for Input value {value!r} in {folder} "
        f"({len(matches)} glob matches)"
    )


def parse_answer(raw: str) -> tuple[int, int, int, int]:
    """
    `Answer` (e.g. "[4, 3, 1, 2]") gives each Input_i's chronological RANK,
    not the slot occupant: Answer[i-1] is Input_i's 1-indexed rank. So
    "[4, 3, 1, 2]" means Input_3 is 1st, Input_4 is 2nd, Input_2 is 3rd,
    Input_1 is 4th -> true order is Input_3, Input_4, Input_2, Input_1.

    Verified by inspecting actual train images against their Sentence for
    two non-self-inverse cases (train Id "u7w0lr", Answer [3,1,2,4], and
    "ernmJh", Answer [4,3,1,2]) -- the rank interpretation matched the
    visual narrative in both; the "slot occupant" interpretation did not.
    """
    ranks = [int(c) - 1 for c in str(raw).strip() if c.isdigit()]
    if len(ranks) != 4 or sorted(ranks) != [0, 1, 2, 3]:
        raise ValueError(f"Answer {raw!r} does not parse to a permutation of 1-4: got {ranks}")
    order = tuple(sorted(range(4), key=lambda i: ranks[i]))
    return order  # type: ignore[return-value]


def load_samples(
    root: Path | str,
    split: str,
    csv_path: Path | str | None = None,
    validate_images: bool = False,
) -> list[Sample]:
    root = Path(root)
    csv_path = Path(csv_path) if csv_path is not None else root / f"{split}.csv"
    image_root = root / split

    df = pd.read_csv(csv_path, dtype=str)

    samples = []
    for row in df.to_dict(orient="records"):
        sample_id = row["Id"]
        folder = image_root / sample_id
        image_paths = tuple(_resolve_image(folder, row[col]) for col in INPUT_COLUMNS)

        if validate_images:
            for p in image_paths:
                with Image.open(p) as im:
                    im.verify()

        answer_raw = row.get("Answer")
        if pd.notna(answer_raw) and str(answer_raw).strip():
            order = parse_answer(answer_raw)
            permutation_index = PERMUTATION_TO_INDEX[order]
        else:
            order = None
            permutation_index = None

        samples.append(
            Sample(
                id=sample_id,
                image_paths=image_paths,  # type: ignore[arg-type]
                sentence=row["Sentence"],
                order=order,
                permutation_index=permutation_index,
            )
        )
    return samples


if __name__ == "__main__":
    if len(sys.argv) > 1:
        root_arg = Path(sys.argv[1])
    else:
        here = Path(__file__).parent
        nested = here / "snuaichallenge_data"
        root_arg = nested if nested.exists() else here

    for split_name in ("train", "test"):
        csv_file = root_arg / f"{split_name}.csv"
        if not csv_file.exists():
            print(f"[skip] {csv_file} not found")
            continue

        loaded = load_samples(root_arg, split_name)
        print(f"{split_name}: {len(loaded)} samples")
        first = loaded[0]
        print(f"  id={first.id}")
        print(f"  images={[p.name for p in first.image_paths]}")
        print(f"  sentence={first.sentence[:80]!r}")
        print(f"  order={first.order}  permutation_index={first.permutation_index}")
