"""Quantitative error analysis of ver8-dpo-ckpt600 vs the v11 answer key.

Outputs:
  - stdout: aggregate statistics
  - wrong59.tsv: per-item table (pred/truth/related-checkpoint preds/structure)
"""
from __future__ import annotations

import csv
import re
from itertools import combinations
from pathlib import Path

GRADE = Path(__file__).resolve().parent.parent
DATA = GRADE.parent


def load(path: Path) -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sid = (row["Id"] or "").strip()
            if not sid:
                continue
            ranks = tuple(int(c) for c in re.findall(r"[1-4]", row["Answer"] or ""))
            if len(ranks) == 4 and sorted(ranks) == [1, 2, 3, 4]:
                out[sid] = ranks
    return out


key = load(GRADE / "submission.csv")
subs = {
    "dpo600": load(GRADE / "submission-ver8-dpo-ckpt600_0.91099.csv"),
    "sft1600": load(GRADE / "submission-ver8-ckpt1600_0.90226.csv"),
    "dpo200": load(GRADE / "submission-ver8_DPO_ckpt200_0.90401.csv"),
    "dpo400": load(GRADE / "submission-ver8-dpo-ckpt400.csv"),
    "dpo800": load(GRADE / "submission-ver8-dpo-ckpt800.csv"),
    "dpo1000": load(GRADE / "submission-ver8-dpo-ckpt1000.csv"),
    "ver4": load(GRADE / "submission-ver4-ckpt1600_0.90226.csv"),
}

# captions
captions: dict[str, str] = {}
with open(DATA / "test.csv", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        captions[row["Id"].strip()] = row["Sentence"]

# the 73 items visually audited for key v10/v11
audit73 = set()
with open(GRADE / "_recheck" / "mismatch73.tsv", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2 and parts[0] != "Id":
            audit73.add(parts[0])


def seq(ranks: tuple[int, ...]) -> tuple[int, ...]:
    """Temporal sequence of input indices: seq[0] = which Input is 1st in time."""
    return tuple(sorted(range(1, 5), key=lambda i: ranks[i - 1]))


def kendall(p: tuple[int, ...], t: tuple[int, ...]) -> int:
    return sum(
        (p[i] < p[j]) != (t[i] < t[j]) for i, j in combinations(range(4), 2)
    )


def classify(p: tuple[int, ...], t: tuple[int, ...]) -> str:
    d = kendall(p, t)
    sp, st = seq(p), seq(t)
    if d == 6:
        return "full-reversal"
    if d == 1:
        # which adjacent temporal slots got swapped
        for k in range(3):
            s2 = list(st)
            s2[k], s2[k + 1] = s2[k + 1], s2[k]
            if tuple(s2) == sp:
                return f"adjacent-swap@{k + 1}-{k + 2}"
        return "d1"
    # single element displaced? (one input moved, others keep relative order)
    for i in range(4):
        rp = [x for x in sp if x != sp[i]]
        rt = [x for x in st if x != sp[i]]
        if rp == rt:
            return f"one-moved(d={d})"
    if d == 5 and sp == tuple(reversed(st[1:] + st[:1])):
        pass
    return f"scramble(d={d})"


wrong = [i for i in key if i in subs["dpo600"] and subs["dpo600"][i] != key[i]]
print(f"total wrong: {len(wrong)}\n")

rows = []
for sid in wrong:
    p, t = subs["dpo600"][sid], key[sid]
    d = kendall(p, t)
    typ = classify(p, t)
    sft = subs["sft1600"].get(sid)
    origin = (
        "inherited-same" if sft == p
        else ("dpo-introduced(sft-was-right)" if sft == t else "both-wrong-diff")
    )
    traj = "".join(
        "X" if subs[c].get(sid) == t else ("=" if subs[c].get(sid) == p else "o")
        for c in ("dpo200", "dpo400", "dpo800", "dpo1000")
    )
    rows.append(
        dict(
            Id=sid,
            pred=str(list(p)),
            truth=str(list(t)),
            pred_seq="".join(map(str, seq(p))),
            truth_seq="".join(map(str, seq(t))),
            kendall=d,
            type=typ,
            origin=origin,
            sft1600=str(list(sft)) if sft else "?",
            ver4=str(list(subs["ver4"][sid])) if sid in subs["ver4"] else "?",
            dpo_traj_200_400_800_1000=traj,
            audited73="Y" if sid in audit73 else "N",
            caption=captions.get(sid, ""),
        )
    )

# aggregates
from collections import Counter

print("--- kendall distance (of 6 pairs) ---")
for k, v in sorted(Counter(r["kendall"] for r in rows).items()):
    print(f"  d={k}: {v}")
print("\n--- error type ---")
for k, v in Counter(r["type"] for r in rows).most_common():
    print(f"  {k}: {v}")
print("\n--- origin vs SFT parent (ver8-ckpt1600) ---")
for k, v in Counter(r["origin"] for r in rows).most_common():
    print(f"  {k}: {v}")
print("\n--- in 73-item visual audit set? ---")
for k, v in Counter(r["audited73"] for r in rows).most_common():
    print(f"  {k}: {v}")

# trajectory legend: for each of dpo200/400/800/1000 -- X=matches truth, ==matches dpo600's (wrong) answer, o=third value
print("\n--- dpo trajectory across 200/400/800/1000 (X=right, ==same wrong ans as ckpt600, o=other) ---")
for k, v in Counter(r["dpo_traj_200_400_800_1000"] for r in rows).most_common():
    print(f"  {k}: {v}")

# identity permutation involvement
id_pred = sum(1 for r in rows if r["pred"] == "[1, 2, 3, 4]")
id_truth = sum(1 for r in rows if r["truth"] == "[1, 2, 3, 4]")
print(f"\npred=identity: {id_pred}   truth=identity(model missed): {id_truth}")
n_id_pred_all = sum(1 for s in subs["dpo600"].values() if s == (1, 2, 3, 4))
n_id_key_all = sum(1 for s in key.values() if s == (1, 2, 3, 4))
print(f"identity count overall: dpo600={n_id_pred_all}, key={n_id_key_all}")

# first/last frame correctness among wrong items
first_ok = sum(1 for r in rows if r["pred_seq"][0] == r["truth_seq"][0])
last_ok = sum(1 for r in rows if r["pred_seq"][3] == r["truth_seq"][3])
print(f"\namong 59 wrong: first-frame correct {first_ok}, last-frame correct {last_ok}")

# where do the other checkpoints stand on these 59?
print("\n--- accuracy of each checkpoint on these 59 items ---")
for name, sub in subs.items():
    ok = sum(1 for sid in wrong if sub.get(sid) == key[sid])
    print(f"  {name}: {ok}/59")

# full-test agreement of each checkpoint with key
print("\n--- full-test agreement with key (self-referential, for trend only) ---")
for name, sub in subs.items():
    ok = sum(1 for sid in key if sub.get(sid) == key[sid])
    print(f"  {name}: {ok}/819 = {ok / 819:.4f}")

out = Path(__file__).parent / "wrong59.tsv"
with open(out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote {out} ({len(rows)} rows)")
