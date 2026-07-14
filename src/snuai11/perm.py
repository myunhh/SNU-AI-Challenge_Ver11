"""Permutation conventions — single source of truth (SSOT).

Encodings
---------
rank tuple  : rank[i] = chronological position (0-based) of Input_i.
              Kaggle `Answer` = "[rank[0]+1, rank[1]+1, rank[2]+1, rank[3]+1]".
order tuple : order[t] = which input (0-3) is shown at chronological step t.
              rank and order are inverse permutations of each other.

The 24-class index space and the letter labels A..X are defined over RANK
tuples: class c <-> ALL_PERMS[c] <-> letter chr(ord('A') + c).

Never mix an index computed from an order tuple with the rank-based class
space: the two agree only for self-inverse permutations (12/24 of them,
~49.5% of real train rows).
"""

from __future__ import annotations

import itertools

Perm = tuple[int, int, int, int]

N_CLASSES = 24
ALL_PERMS: list[Perm] = list(itertools.permutations(range(4)))  # type: ignore[assignment]
_PERM_TO_INDEX: dict[Perm, int] = {p: i for i, p in enumerate(ALL_PERMS)}

LETTERS: list[str] = [chr(ord("A") + i) for i in range(N_CLASSES)]  # A..X
IDENTITY: Perm = (0, 1, 2, 3)


def index_of(rank: Perm) -> int:
    """24-class index of a rank tuple."""
    return _PERM_TO_INDEX[tuple(rank)]  # type: ignore[index]


def rank_of_index(idx: int) -> Perm:
    return ALL_PERMS[idx]


def letter_of_index(idx: int) -> str:
    return LETTERS[idx]


def index_of_letter(letter: str) -> int:
    letter = letter.strip().upper()
    idx = ord(letter) - ord("A")
    if not (0 <= idx < N_CLASSES) or len(letter) != 1:
        raise ValueError(f"not a class letter A-X: {letter!r}")
    return idx


def invert(p: Perm) -> Perm:
    """Inverse permutation. order_to_rank == rank_to_order == invert."""
    inv = [0, 0, 0, 0]
    for a, b in enumerate(p):
        inv[b] = a
    return tuple(inv)  # type: ignore[return-value]


order_to_rank = invert
rank_to_order = invert


def compose(p: Perm, q: Perm) -> Perm:
    """(p o q)[i] = p[q[i]]."""
    return tuple(p[q[i]] for i in range(4))  # type: ignore[return-value]


def apply_view(rank_orig: Perm, sigma: Perm) -> Perm:
    """Rank tuple seen by a TTA view.

    A view permutation sigma places original input sigma[j] at view slot j.
    The view's rank tuple is then rank_view[j] = rank_orig[sigma[j]].
    """
    return compose(rank_orig, sigma)


def unapply_view(rank_view: Perm, sigma: Perm) -> Perm:
    """Inverse of apply_view: recover original-space rank from a view rank."""
    return compose(rank_view, invert(sigma))


def view_class_map(sigma: Perm) -> list[int]:
    """m[c_view] = c_orig for remapping 24-class scores out of a TTA view."""
    return [index_of(unapply_view(ALL_PERMS[c], sigma)) for c in range(N_CLASSES)]


def kendall_distance(a: Perm, b: Perm) -> int:
    """Number of discordant pairs between two rank tuples (0..6)."""
    d = 0
    for i in range(4):
        for j in range(i + 1, 4):
            if (a[i] - a[j]) * (b[i] - b[j]) < 0:
                d += 1
    return d


def pairwise_score(pred: Perm, truth: Perm) -> float:
    """1 - KendallTau/6 — the partial-credit metric the LB tracks."""
    return 1.0 - kendall_distance(pred, truth) / 6.0


def adjacent_swap_neighbors(rank: Perm) -> list[Perm]:
    """The 3 rank tuples at Kendall distance 1 (swap chronological steps t,t+1).

    Dominant holdout error mode (KT=1) — used as DPO hard negatives.
    """
    order = rank_to_order(rank)
    out: list[Perm] = []
    for t in range(3):
        o = list(order)
        o[t], o[t + 1] = o[t + 1], o[t]
        out.append(order_to_rank(tuple(o)))  # type: ignore[arg-type]
    return out
