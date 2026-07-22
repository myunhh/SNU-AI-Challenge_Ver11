"""
Grade a submission CSV against an answer-key CSV.

Both files must have columns: Id, Answer
Answer format: "[1, 4, 2, 3]" -- the i-th number is Input_i's chronological
rank (same convention as train.csv / adapter.py).

Usage:
    python grade.py <submission.csv> [--truth submission.csv] [--show-wrong N]

Reports:
    - exact-match accuracy (all 4 ranks correct)
    - per-position accuracy (fraction of the 4 slots correct, averaged)
    - pairwise-order accuracy (fraction of the 6 image pairs in correct
      relative order, averaged)
    - Id coverage problems (missing / extra / invalid rows)
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from itertools import combinations
from pathlib import Path


def load_answers(path: Path) -> tuple[dict[str, tuple[int, ...]], list[str]]:
    """Return ({Id: ranks}, [ids with missing/invalid Answer])."""
    answers: dict[str, tuple[int, ...]] = {}
    invalid: list[str] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "Id" not in reader.fieldnames or "Answer" not in reader.fieldnames:
            sys.exit(f"error: {path} must have 'Id' and 'Answer' columns (got {reader.fieldnames})")
        for row in reader:
            sample_id = (row["Id"] or "").strip()
            if not sample_id:
                continue
            ranks = tuple(int(c) for c in re.findall(r"[1-4]", row["Answer"] or ""))
            if len(ranks) != 4 or sorted(ranks) != [1, 2, 3, 4]:
                invalid.append(sample_id)
            else:
                answers[sample_id] = ranks
    return answers, invalid


# scored submissions the answer key was built from (file, leaderboard score).
# Grading any of these (or a near-copy) against the key is self-referential:
# the printed accuracy is inflated by several points.
KEY_SOURCES: list[tuple[str, float | None]] = [
    ("submission-0710-1_0.45724.csv", 0.45724),
    ("submission_spaced.csv", 0.77486),
    ("submission-0711-1_0.82373.csv", 0.82373),
    ("submission-0712-1_0.85863.csv", 0.85863),
    ("submission-0713-1_0.85863.csv", 0.85863),  # identical to 0712-1
    ("submission-0712-2_0.86561.csv", 0.86561),
    ("submission-0713-3_0.83420.csv", 0.83420),
    ("submission-ver10-ckpt20_0.75741.csv", 0.75741),
    ("submission-ver4-ckpt1600_0.90226.csv", 0.90226),
    ("submission-ver4-ckpt2000_0.89354.csv", 0.89354),
    ("submission-ver8_DPO_ckpt200_0.90401.csv", 0.90401),
    ("submission-ver8-refactored_0.85863.csv", 0.85863),
    ("submission-ver8-tta4-ckpt600_0.91623.csv", 0.91623),  # v12 source, see below
    ("submission-ver8-tta5-ckpt600_0.91797.csv", 0.91797),  # 2026-07-21 night champion
    ("submission-ver8-champ-tta8-balanced_0.93019.csv", 0.93019),  # current champion, 2026-07-22
    ("submission_key_v1_backup.csv", None),  # original claude-vision key
]

# calibration of the current key (v8 = v7 vote + manual visual arbitration):
# v7 is a log-odds-weighted majority vote across the 11 scored KEY_SOURCES
# files above (weight = log(acc*23/(1-acc)) per source). Leave-one-out
# cross-validation (rebuild the key excluding each source in turn, then
# compare to that source's real leaderboard score) gives mean bias -0.0085
# (key agreement runs ~1pt high on average) and MAE 0.012, max single-source
# residual 0.027. v7 superseded the old v6 key, which was well-calibrated on
# scores up to 0.866 but underestimated the newer ver4/ver8 checkpoints
# (~0.90 actual) by 5-6pts. v8 additionally hand-resolves the 17 items where
# v7's vote margin was thin (<2 weight units, i.e. a near coin flip) by
# inspecting the actual test images + caption against both candidates --
# 6 of 17 confirmed the v7 vote, 11 of 17 reverted to the pre-v7 answer or a
# third value neither vote nor v6 had. See submission_key_v8_manual17.csv.
# v9 (submission_key_v9_manual1.csv) fixes one further item found by re-inspecting images
# after ver8-ckpt1600 disagreed with the key: YZuSuG (pumpkin-carving clip) had v6/v7/v8 all
# place an already-cut frame before the uncut frame, which is physically impossible (carving
# can't be undone) -- corrected [1,4,3,2] -> [2,4,3,1].
# v10 (submission_key_v10_visual73.csv, now = submission.csv) is a full manual visual audit
# of all 73 items where ver8-ckpt1600 disagreed with v9 -- each Id's images + caption were
# re-read independently (not just voted among key/ver8-ckpt1600/ver4-ckpt1600) via 6 parallel
# review passes. 21 of 73 were corrected (wrong direction/order not supported by the images,
# several were physically-impossible orderings like already-finished states preceding raw
# states); the other 52 confirmed the existing key value -- ver8-ckpt1600 and ver4-ckpt1600
# are 97.4% identical submissions, so their agreement alone was treated as weak evidence, not
# proof of a key error. A few of the 21 fixes (8L7TfG, Z9CdGI, a94blB) landed on a value that
# matched NONE of key/ver8-ckpt1600/ver4-ckpt1600 -- flagged lower-confidence, worth a second
# look if this key's calibration drifts.
# v11 (submission_key_v11_selfcheck.csv, now = submission.csv): re-audited all 73 items again
# personally (single-pass, not delegated) as a check on v10. 72/73 held up; a94blB (moth/
# cartographer story) was wrong in v10 -- the two "desk" frames were misassigned by light
# source (candlelit desk = attic/night start, window-lit desk = dawn observatory/end, not the
# reverse) -- corrected [3,1,2,4] -> [3,4,2,1]. Z9CdGI remains flagged uncertain (re-check
# surfaced a plausible counter-ordering for the two near-rock frames but no confident
# resolution). Raw self-referential agreement with ver8-ckpt1600: 746/819=0.9109 (v9) ->
# 761/819=0.9292 (v10) -> 762/819=0.9304 (v11).
# v12 (2026-07-20): champion serving changed TTA3->TTA4 (Klein balanced 4-view set,
# Ver8/scripts/tta.py BALANCED4), confirmed via real LB 0.91099 -> 0.91623 (+0.524pp) --
# the single highest-accuracy source this key has ever incorporated. Folded in via the
# same log-odds weighted vote as v7 (weight = log(acc*23/(1-acc)) per scored source,
# now including submission-ver8-tta4-ckpt600_0.91623.csv), restricted to the 66 items
# where v11 and the new TTA4 submission disagreed -- items where all other sources
# already agreed with v11 are untouched, so this is a targeted update, not a rebuild.
# 21 of 66 flipped to the vote winner. 3 of 66 were deliberately excluded from the
# mechanical vote and left at their v11 value despite the vote favoring TTA4's answer,
# because they carry a documented higher-confidence override that a raw vote shouldn't
# clobber: YZuSuG (v9's physical-impossibility fix -- cut frame can't precede uncut
# frame -- the vote wanted to revert it), Z9CdGI (v10/v11 already flagged this
# unresolved/uncertain after a full manual re-check, so an unaudited vote isn't better
# evidence), 8L7TfG (v10 already flagged this fix low-confidence; a second unaudited
# flip on top of a low-confidence flip compounds uncertainty rather than resolving it).
# None of the 21 applied flips were individually visually re-audited (unlike v8-v11's
# manual passes) -- they rest on source-weighted vote only. Calibration below (bias/MAE)
# is carried over from v11 and not re-measured for v12; treat v12 estimates as
# provisional until a leave-one-out re-check is done.
# v13 (submission_key_v13_audit26.csv, now = submission.csv, 2026-07-21): full manual visual
# audit of the 26 items where ALL of TTA5/6/7/8 unanimously disagree with the TTA4 champion
# submission (the set that decides TTA-variant ranking). Each item's 4 images + caption were
# re-read; 20/26 got a med+ confidence verdict, of which 13 confirmed v12 and 7 were corrected:
# JwI2O1, YZuSuG, ko74ge, FrNHna, 2sqUTP, b44PjE, eEcInl (b44PjE/eEcInl revert v12's
# vote-only flips; YZuSuG [2,4,3,1]->[2,3,4,1]: v9's lid-direction physics was right but the
# eyes/mouth sub-order was wrong -- in frame _hgs the nose is still marker-only while _oxc has
# it cut open, so _hgs must precede _oxc). 3 items stay flagged low-confidence and unchanged:
# UwdDWd (oven staging order genuinely ambiguous), Z9CdGI (slackline -- audit suggests an
# out-and-back crossing reading [4,1,3,2] but not confident; twice-flagged before), a7rmwJ
# (gym I1-vs-I4 order soft). Under v13 the champion TTA family converges: TTA4 767 = TTA5 767
# > TTA6/7 766 > TTA8 764 -- i.e. v11's apparent TTA5/6 edge (+6) was unaudited-vote bias, and
# TTA-variant ranking is a measured coin flip. IMPORTANT measured caveat: near the champion
# (~0.91+) this key CANNOT rank candidates -- key11 ordered dpo600_tta3 > ver4 > champ_tta4
# while real LB says the exact opposite, and it also got the TTA3->TTA4 direction wrong
# (est -6 EM vs real +0.524pp). Use est only as a coarse gate (>=1.5pp gaps); ±1pp decisions
# need real LB slots. Calibration still carried from v11, not re-measured.
#
# 2026-07-21 night: champion serving changed TTA4->TTA5 (identity + 4 seeded random shuffles,
# tta.py's pre-existing seed-shuffle path, NOT balanced) -- real LB 0.91623 -> 0.91797
# (+0.174pp, exactly +1 EM on the 573-item public grid, see README_lb_scoring note below).
# v13 itself called TTA4 vs TTA5 a dead tie (767/819 each) and a from-key Monte-Carlo
# projection (conditioning on the champion's public-573 draw) expected roughly +4-5 EM with
# P(win)=0.97 -- the *direction* was right but the real *magnitude* landed at the pessimistic
# edge of that projection's 90% CI. Net lesson, consistent with the TTA3->TTA4 miscall above:
# near the champion this key (and any MC sampling built on it) is a coarse yes/no gate on
# direction, not a magnitude estimator -- don't size expected gains off it, and don't skip
# real LB slots just because the projected win probability looks high.
#
# 2026-07-22: champion serving changed TTA5->TTA8-balanced (`--tta 8 --tta-balanced8`,
# Klein V + inverse-closed coset -- BALANCED8 in tta.py) -- real LB 0.91797 -> 0.93019,
# +1.22pp / +7 EM on the 573-item public grid. This is the single largest serving-only
# jump this key has ever seen (more than double TTA3->TTA4's +0.524pp), yet v13 estimates
# it at only 766/819 = est 0.9268 -- UNDERestimating the real gain, the opposite miscue
# direction from the TTA4->TTA5 case above (which overestimated). Two data points now
# disagree on which way this key's magnitude error runs -- treat it as unbiased-but-noisy
# at best, not systematically conservative or optimistic. Diffs vs TTA5/TTA4 submissions
# are 34/819 (4.2%) and 28/819 (3.4%) respectively -- unremarkable, no format/alignment
# red flags.
CALIBRATION_BIAS = 0.0085
CALIBRATION_MAE = 0.012


def pairwise_correct(pred: tuple[int, ...], truth: tuple[int, ...]) -> float:
    """Fraction of the 6 image pairs whose relative order matches."""
    good = sum(
        ((pred[i] < pred[j]) == (truth[i] < truth[j]))
        for i, j in combinations(range(4), 2)
    )
    return good / 6


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade a submission against the answer key.")
    parser.add_argument("submission", type=Path, help="submission CSV to grade")
    parser.add_argument(
        "--truth",
        type=Path,
        default=Path(__file__).parent / "submission.csv",
        help="answer-key CSV (default: submission.csv next to this script)",
    )
    parser.add_argument(
        "--show-wrong", type=int, default=10, metavar="N",
        help="show up to N mismatched items (default 10, 0 to hide)",
    )
    args = parser.parse_args()

    truth, truth_invalid = load_answers(args.truth)
    if truth_invalid:
        print(f"warning: answer key has {len(truth_invalid)} invalid rows: {truth_invalid[:5]}...")
    pred, pred_invalid = load_answers(args.submission)

    missing = [i for i in truth if i not in pred]
    extra = [i for i in pred if i not in truth]
    graded_ids = [i for i in truth if i in pred]

    exact = 0
    position_sum = 0.0
    pairwise_sum = 0.0
    wrong: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for sample_id in graded_ids:
        p, t = pred[sample_id], truth[sample_id]
        if p == t:
            exact += 1
        else:
            wrong.append((sample_id, p, t))
        position_sum += sum(a == b for a, b in zip(p, t)) / 4
        pairwise_sum += pairwise_correct(p, t)

    n = len(graded_ids)
    print(f"answer key : {args.truth}  ({len(truth)} items)")
    print(f"submission : {args.submission}  ({len(pred)} valid items)")
    print("-" * 60)
    if pred_invalid:
        print(f"invalid Answer rows in submission : {len(pred_invalid)}  e.g. {pred_invalid[:5]}")
    if missing:
        print(f"ids missing from submission       : {len(missing)}  e.g. {missing[:5]}")
    if extra:
        print(f"ids not in answer key (ignored)   : {len(extra)}  e.g. {extra[:5]}")
    if not n:
        sys.exit("error: no gradable items (no overlapping valid Ids)")

    print(f"graded items                      : {n}")
    print(f"exact-match accuracy              : {exact}/{n} = {exact / n:.4f}")
    print(f"per-position accuracy             : {position_sum / n:.4f}")
    print(f"pairwise-order accuracy           : {pairwise_sum / n:.4f}")

    # self-referential check: is this (nearly) one of the key's source files?
    key_dir = args.truth.parent
    for src_name, lb in KEY_SOURCES:
        src_path = key_dir / src_name
        if not src_path.exists():
            continue
        if src_path.resolve() == args.submission.resolve():
            print("-" * 60)
            print(f"WARNING: {src_name} is one of the key's source files, so the")
            print("         score above is self-referential and inflated.")
            if lb is not None:
                print(f"         Its actual leaderboard score is known: {lb}")
            break
        src, _ = load_answers(src_path)
        shared = [i for i in graded_ids if i in src]
        if not shared:
            continue
        agree = sum(pred[i] == src[i] for i in shared) / len(shared)
        if agree >= 0.95:
            print("-" * 60)
            print(f"WARNING: submission is {agree:.1%} identical to key source {src_name}.")
            print("         The key was built from that file, so the score above is")
            print("         self-referential and inflated by several points.")
            if lb is not None:
                print(f"         Its actual leaderboard score is known: {lb}")
            break
    else:
        est = exact / n - CALIBRATION_BIAS
        print(f"estimated leaderboard score       : {est:.4f}  (+/- ~{CALIBRATION_MAE:.3f})")

    if wrong and args.show_wrong:
        print("-" * 60)
        print(f"mismatches (up to {args.show_wrong} of {len(wrong)}):")
        for sample_id, p, t in wrong[: args.show_wrong]:
            print(f"  {sample_id}: submitted {list(p)}  !=  answer {list(t)}")


if __name__ == "__main__":
    main()
