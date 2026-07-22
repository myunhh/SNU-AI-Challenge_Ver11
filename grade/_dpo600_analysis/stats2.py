"""Precise cuts: swap-position split, reversal detection, key-proxy vs real-LB gap."""
import csv, re
from itertools import combinations, permutations
from pathlib import Path

HERE = Path(__file__).parent
G = HERE.parent


def load(p):
    out = {}
    for row in csv.DictReader(open(p, encoding="utf-8-sig", newline="")):
        r = tuple(int(c) for c in re.findall(r"[1-4]", row["Answer"] or ""))
        if len(r) == 4 and sorted(r) == [1, 2, 3, 4]:
            out[row["Id"].strip()] = r
    return out


key = load(G / "submission.csv")
dpo = load(G / "submission-ver8-dpo-ckpt600_0.91099.csv")
sft = load(G / "submission-ver8-ckpt1600_0.90226.csv")

wrong = [i for i in key if dpo.get(i) != key[i]]


def seq(r):
    return tuple(sorted(range(1, 5), key=lambda i: r[i - 1]))


def kend(p, t):
    return sum((p[i] < p[j]) != (t[i] < t[j]) for i, j in combinations(range(4), 2))


# reversal-like: is model seq a reversal of key seq (fully or a contiguous block)?
def is_full_reverse(p, t):
    return seq(p) == tuple(reversed(seq(t)))


n_full_rev = sum(is_full_reverse(dpo[i], key[i]) for i in wrong)

# "opposite direction" proxy: kendall distance >= 4 (more pairs wrong than right)
n_majority_rev = sum(kend(dpo[i], key[i]) >= 4 for i in wrong)

# among d==1 single swaps, which adjacent temporal position flipped
pos = {1: 0, 2: 0, 3: 0}
for i in wrong:
    if kend(dpo[i], key[i]) != 1:
        continue
    st, sp = seq(key[i]), seq(dpo[i])
    for k in range(3):
        s2 = list(st)
        s2[k], s2[k + 1] = s2[k + 1], s2[k]
        if tuple(s2) == sp:
            pos[k + 1] += 1

print("=== DPO600 vs SFT1600 vs key ===")
print(f"key agreement: dpo600={sum(dpo[i]==key[i] for i in key)}/819, "
      f"sft1600={sum(sft[i]==key[i] for i in key)}/819")
print("REAL leaderboard (from filenames): dpo600=0.91099, sft1600=0.90226  (+0.87pt)")
print("=> vs key, dpo600 (760) is BELOW sft1600 (762), but real LB is ABOVE.")
print("   The key MISranks dpo vs sft; several 'DPO regressions' are real DPO gains.\n")

print(f"wrong items: {len(wrong)}")
print(f"full-reversal of key order       : {n_full_rev}")
print(f"majority-reversed (kendall>=4)    : {n_majority_rev}")
print(f"d==1 single-swap position (temporal adjacent pair flipped):")
print(f"   1st-2nd:{pos[1]}  2nd-3rd:{pos[2]}  3rd-4th:{pos[3]}")

# first-frame and last-frame hit rate on wrong items
ff = sum(seq(dpo[i])[0] == seq(key[i])[0] for i in wrong)
lf = sum(seq(dpo[i])[3] == seq(key[i])[3] for i in wrong)
print(f"\namong {len(wrong)} wrong: correct FIRST frame {ff}, correct LAST frame {lf}")

# how many wrong items does SFT also get wrong (persistent) vs dpo-only
sft_also = sum(sft.get(i) != key[i] for i in wrong)
print(f"of {len(wrong)} dpo-wrong: SFT also wrong on {sft_also}  (dpo-only new errors: {len(wrong)-sft_also})")
