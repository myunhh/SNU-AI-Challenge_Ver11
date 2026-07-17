#!/usr/bin/env python
"""Average same-run checkpoints (LoRA adapter + Score24Head) into one model.

Post-hoc OPTION — not part of the confirmed pipeline. Tail-checkpoint
averaging of a single run (SWA-style) often buys a small free gain; judge it
against the raw final checkpoint via in-sample train eval before spending an
LB slot. See snuai11.soup for the same-trajectory caveat and rule note.

  python scripts/avg_adapters.py runs/sft32b_v11_ws8/checkpoint-1400 \
      runs/sft32b_v11_ws8/adapter_final --out runs/sft32b_v11_ws8/soup_tail2
  python run_pre.py --adapter runs/sft32b_v11_ws8/soup_tail2/adapter ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from snuai11.soup import soup_checkpoints  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoints", nargs="+", type=Path,
                    help="2+ checkpoint dirs of the SAME run (each: adapter/ + head.pt)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = soup_checkpoints(args.checkpoints, args.out)
    print(f"[soup] {len(args.checkpoints)} checkpoints -> {out} (adapter/ + head.pt)")


if __name__ == "__main__":
    main()
