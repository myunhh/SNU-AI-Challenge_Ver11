#!/usr/bin/env python
"""Training one-command: clone + pip + `python run_fit.py` just works.

Defaults = Ver11 confirmed recipe: 32B bnb-4bit QLoRA + Cross-Targeted
FitPrune(50% keep, active during training) + Stackelberg two-time-scale
(body 2e-4 / head 1e-3) + score24 head, 2000 steps.

Any extra CLI args pass straight through to snuai11.train_sft
(e.g. `python run_fit.py --steps 100 --out runs/smoke`).
DPO phase 2: `python run_fit.py --phase dpo --adapter runs/sft32b_v11/adapter_final/adapter`.
"""

import sys

from run_common import bootstrap_path, ensure_data, resolve_model_id

bootstrap_path()


def main() -> None:
    argv = sys.argv[1:]
    ensure_data("data")
    if "--model-id" not in argv:
        argv = ["--model-id", resolve_model_id()] + argv
    from snuai11.train_sft import main as train_main

    train_main(argv)


if __name__ == "__main__":
    main()
