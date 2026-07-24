#!/usr/bin/env python3
"""Two-job inference dashboard — stacks term_dashboard.py's render_infer()
for the queued reheat -> checkpoint-1200 test-inference chain
(scripts/run_ws8_reheat_then_ckpt1200.sh). Reuses that module's rendering
and error/alive-check logic as-is; this file only adds "show two dirs
stacked" on top.

Usage:
  python3 scripts/dashboard_two_infer.py --watch
  python3 scripts/dashboard_two_infer.py            # one snapshot
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
    ("reheat (train+infer motion=0.3)", Path("runs/test_v11_ws8_reheat")),
    # NOT pre-motion: train_args.json shows this checkpoint was trained with
    # motion_weight=0.3 too — only inference here omits --motion-weight (0).
    # See scripts/run_ws8_reheat_then_ckpt1200.sh 2026-07-23 correction note.
    ("checkpoint-1200 (train motion=0.3, infer motion=0)", Path("runs/test_v11_ckpt1200")),
]


def render_all(width: int) -> str:
    blocks = []
    for label, infer_dir in JOBS:
        td.SFT_LOG = td.ROOT / infer_dir.with_suffix(".log")
        header = f"=== {label} : {infer_dir} ==="
        if not (td.ROOT / infer_dir).exists():
            blocks.append(header + "\n  (대기 중 — 아직 시작 안 됨, 이전 단계가 끝나야 시작)")
            continue
        blocks.append(header + "\n" + td.render_infer(width, td.ROOT / infer_dir, 819))
    return "\n\n".join(blocks)


def main() -> None:
    ap = argparse.ArgumentParser()
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
