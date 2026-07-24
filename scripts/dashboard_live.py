#!/usr/bin/env python3
"""Live dashboard for tonight's (2026-07-24) jobs: the running DPO-on-
ckpt1200 test819 inference, then the boost_frac pre-flight smoke + 3-leg
triage that starts once it finishes (scripts/boost_frac_triage.sh, launched
automatically by the cron loop). Stacks term_dashboard.py's render_infer()
the same way dashboard_two_infer.py does for its (now-finished) chain;
generalized to N jobs with per-job totals since the boost triage moves
through 4 sequential dirs of different sizes (n=2, then three n=48 legs),
not just 2 same-sized ones. Panels for dirs that don't exist yet show a
one-line "대기 중" placeholder instead of erroring.

There is no loss/acc/step panel here -- everything running tonight is
inference (run_pre.py), not training, so those fields don't exist. Use
term_dashboard.py's default (training) mode for that, if training is ever
running again.

Usage:
  python3 scripts/dashboard_live.py --watch       # auto-refresh every 20s
  python3 scripts/dashboard_live.py --watch 10    # custom interval
  python3 scripts/dashboard_live.py               # one snapshot, then exit
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import term_dashboard as td  # noqa: E402

JOBS = [
    ("DPO-on-ckpt1200 test819 (adapter_final, TTA8, motion=0.3)",
     Path("runs/test_v11_dpo1200_final_tta8"), 819),
    ("boost pre-flight smoke (boost_frac=0.5, n=2, fail-fast)",
     Path("runs/triage_boost_smoke2"), 2),
    # 2026-07-24 01:xx: upgraded from n=48 in-sample smoke to full real
    # test819 (submission-quality) for all 3 legs -- see
    # scripts/boost_frac_triage.sh header for why/timing.
    ("boost triage boost_frac=0.0 (control, full test819)",
     Path("runs/triage_boost_0.0"), 819),
    ("boost triage boost_frac=0.2 (full test819)",
     Path("runs/triage_boost_0.2"), 819),
    ("boost triage boost_frac=0.5 (full test819)",
     Path("runs/triage_boost_0.5"), 819),
]


def render_all(width: int) -> str:
    blocks = []
    for label, infer_dir, total in JOBS:
        td.SFT_LOG = td.ROOT / infer_dir.with_suffix(".log")
        header = f"=== {label} : {infer_dir} ==="
        if not (td.ROOT / infer_dir).exists():
            blocks.append(header + "\n  (대기 중 — 아직 시작 안 됨, 이전 단계가 끝나야 시작)")
            continue
        blocks.append(header + "\n" + td.render_infer(width, td.ROOT / infer_dir, total))
    return "\n\n".join(blocks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", nargs="?", const=20, type=int, default=None, metavar="SECONDS")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.no_color:
        td.C.set(False)

    width = min(max(shutil.get_terminal_size((100, 24)).columns, 64), 96)

    if args.watch is None:
        print(render_all(width))
        return

    redraw = not args.no_color

    def ctrl(seq: str) -> None:
        if redraw:
            sys.stdout.write(seq)

    ctrl("\x1b[?1049h\x1b[?25l")
    try:
        while True:
            ctrl("\x1b[H\x1b[J")
            print(render_all(width))
            sys.stdout.flush()
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass
    finally:
        ctrl("\x1b[?25h\x1b[?1049l")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
