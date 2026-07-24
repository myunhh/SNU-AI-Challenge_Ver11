#!/bin/bash
# 2026-07-24 ~10:45: continuation of boost_frac_sftonly.sh after Claude found
# and fixed a real contamination bug: runs/triage_boost_0.5's progress.jsonl
# had 216 rows scored with the DPO-on-ckpt1200 adapter (leftover from the
# earlier wrong-base boost_frac_remaining.sh run) silently mixed with rows
# scored under the SFT-only adapter, because infer.py's check_resume_config
# didn't check "adapter"/"head" for drift. Both are fixed now: the 216
# contaminated rows were stripped from runs/triage_boost_0.5/progress.jsonl
# (202 correct SFT-only-adapter rows kept), and check_resume_config now
# covers adapter/head so this can't happen silently again.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/py3_11/lib
PY=$HOME/anaconda3/envs/py3_11/bin/python
ADAPTER=runs/sft32b_v11_ws8_checkpoint-1200/adapter
HEAD=runs/sft32b_v11_ws8_checkpoint-1200/head.pt

for bf in 0.5 0.2; do
  echo "[boost-fixed] $(date '+%F %T') start/resume full test819 boost_frac=$bf (SFT-only ckpt1200, motion=0.3)"
  bash scripts/run_pre_supervised.sh "$ADAPTER" "$HEAD" "runs/triage_boost_${bf}" \
    --split test --tta 8 --motion-weight 0.3 --boost-frac "$bf"
  echo "[boost-fixed] $(date '+%F %T') done boost_frac=$bf, grading against local key"
  "$PY" grade/grade.py "runs/triage_boost_${bf}/submission.csv" > "runs/grade_boost_${bf}.txt" 2>&1
done

echo "[boost-fixed] $(date '+%F %T') both legs done -- summary:"
for bf in 0.0 0.2 0.5; do
  echo "=== boost_frac=$bf ==="
  cat "runs/grade_boost_${bf}.txt" 2>/dev/null
  echo
done
