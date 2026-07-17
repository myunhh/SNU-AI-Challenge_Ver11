#!/usr/bin/env python
"""Inference one-command -> runs/test_v11/submission.csv.

Defaults (2026-07-17): score24 + FitPrune stage1 + balanced TTA4 (Klein set)
+ full-token stage2 on EVERY sample (--stage2 always; 24h 예산 대비 무료).
The pre-2026-07-17 pipeline is reproducible with: --tta 3 --stage2 cascade.
Point it at a trained adapter:
  python run_pre.py --adapter runs/sft32b_v11_ws8/adapter_final/adapter
In-sample sanity check on train (no local holdout — Ver11 trains on 100%):
  python run_pre.py --split train --eval --adapter ... --limit 200
"""

import sys

from run_common import bootstrap_path, ensure_data, resolve_model_id

bootstrap_path()


def main() -> None:
    argv = sys.argv[1:]
    ensure_data("data")
    if "--model-id" not in argv:
        argv = ["--model-id", resolve_model_id()] + argv
    from snuai11.infer import main as infer_main

    infer_main(argv)


if __name__ == "__main__":
    main()
