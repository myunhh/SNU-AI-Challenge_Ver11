#!/bin/bash
# 2026-07-24 08:1x: follow-up to boost_frac_triage.sh after discovering its
# boost_frac=0.0 leg was config-IDENTICAL (diff config.json: only "out")
# to the already-finished, already-graded runs/test_v11_dpo1200_final_tta8
# (EM 0.9096, est LB 0.9011 -- see runs/grade_dpo1200_final_tta8.txt). That
# leg was killed at ~536/819 to reclaim the GPU on the last day instead of
# burning another ~50min re-deriving a known number. This script does ONLY
# the two informative legs (0.2, 0.5), same adapter/flags/grading convention
# as boost_frac_triage.sh, writing to the same runs/triage_boost_<bf> dirs
# so scripts/dashboard_live.py and the running cron monitor need no changes.
# Pre-flight smoke already passed clean for boost_frac=0.5 (n=2, EM 1.0) --
# not repeating it here; 0.2 is a strictly smaller perturbation.
#
# Order 0.5 BEFORE 0.2 (2026-07-24 08:1x re-sequenced, last-day time
# pressure): 0.5 is the more aggressive dose. If it shows no lift (or hurts)
# vs the already-known 0.0 baseline (EM 0.9096, est 0.9011 --
# runs/grade_dpo1200_final_tta8.txt), the strictly-weaker 0.2 dose is very
# unlikely to lift either -- an early null on 0.5 makes 0.2 skippable,
# recovering ~3h. Running the higher-information leg first means a decision
# is possible after ONE leg instead of requiring both.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/py3_11/lib
PY=$HOME/anaconda3/envs/py3_11/bin/python
ADAPTER=runs/ckpt1200-dpo-new/adapter_final/adapter
HEAD=runs/ckpt1200-dpo-new/adapter_final/head.pt

for bf in 0.5 0.2; do
  echo "[boost-remaining] $(date '+%F %T') start full test819 boost_frac=$bf"
  bash scripts/run_pre_supervised.sh "$ADAPTER" "$HEAD" "runs/triage_boost_${bf}" \
    --split test --tta 8 --motion-weight 0.3 --boost-frac "$bf"
  echo "[boost-remaining] $(date '+%F %T') done boost_frac=$bf, grading against local key"
  "$PY" grade/grade.py "runs/triage_boost_${bf}/submission.csv" > "runs/grade_boost_${bf}.txt" 2>&1
done

echo "[boost-remaining] $(date '+%F %T') both remaining boost_frac legs done -- summary:"
for bf in 0.2 0.5; do
  echo "=== boost_frac=$bf ==="
  cat "runs/grade_boost_${bf}.txt" 2>/dev/null
  echo
done
