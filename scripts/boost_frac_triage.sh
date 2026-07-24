#!/bin/bash
# Full real test819 (--split test, TTA8 balanced) for all 3 boost_frac
# values, 2026-07-24 01:xx user call: plenty of wall-clock until the ~9-10am
# check-in, so go straight to submission-quality runs instead of the n=48
# in-sample smoke this script used earlier tonight (git blame/history has
# that version if it's ever needed again). Each leg is graded against the
# local key immediately after it finishes (not batched at the end), so
# whichever legs are done by check-in time are already gradeable even if
# the full sweep hasn't finished. Adapter is ckpt1200-dpo-new/adapter_final,
# NOT a champion adapter from a different track (e.g. Ver8) -- this
# adapter's own training (train_args.json: phase=dpo, motion_weight=0.3)
# actually saw FitPrune-pruned inputs, so it is the compatible base for
# boosting an already-pruned sequence further. motion_weight=0.3 at serve
# time matches both that training config and the concurrent full-819
# evaluation of this same adapter (runs/test_v11_dpo1200_final_tta8). The
# boost_frac=0.0 leg is a same-adapter, same-everything-else-config control
# so the other legs are a clean paired delta against a REAL 819-item score,
# not a small-n proxy.
#
# Timing (computed 2026-07-24 ~00:45, from the concurrent job's observed
# rate ~12.7s/sample): ~2.9h per 819-sample leg, ~8.6h for all 3 -> finishes
# around 10:50am at pure throughput, i.e. LATER than the user's 9-10am
# check-in if nothing crashes, and possibly later still if OOM retries
# happen. This was communicated to the user before launch. If checking in
# before all 3 are done: scripts/dashboard_live.py --watch shows live
# per-leg progress, and runs/grade_boost_<bf>.txt exists for every leg
# that HAS finished regardless of whether the other legs are done.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/py3_11/lib
PY=$HOME/anaconda3/envs/py3_11/bin/python
ADAPTER=runs/ckpt1200-dpo-new/adapter_final/adapter
HEAD=runs/ckpt1200-dpo-new/adapter_final/head.pt

# --boost-frac's idx-branch (vlm.py forward_prepared/scored_idx) has never
# run against a real model -- CPU tests only so far (GPU was occupied all
# night). run_pre_supervised.sh retries forever on ANY crash assuming
# transient OOM; if this new code has a real bug that would loop forever
# reloading the 32B model and burning GPU-hours with nothing to show for it
# on the LAST day. So: a tiny direct (non-retrying) smoke first, fail fast.
# One retry, but ONLY when the failure looks like the transient VRAM-
# boundary blip you'd expect right after a prior huge job just released the
# card (not a real bug) -- distinguished by grepping the log for the CUDA
# OOM signature. Any other error (shape mismatch, AttributeError, etc)
# fails fast on attempt 1, no retry: that is a real bug in the new code.
for attempt in 1 2; do
  echo "[boost-triage] $(date '+%F %T') pre-flight smoke attempt $attempt: boost_frac=0.5, n=2"
  rm -f runs/triage_boost_smoke2/progress.jsonl runs/triage_boost_smoke2/eval.json
  "$PY" run_pre.py --adapter "$ADAPTER" --head "$HEAD" --out runs/triage_boost_smoke2 \
    --split train --eval --limit 2 --tta 8 --motion-weight 0.3 --boost-frac 0.5 \
    2>&1 | tee runs/triage_boost_smoke2.log
  smoke_rc=${PIPESTATUS[0]}
  if [ "$smoke_rc" -eq 0 ]; then
    break
  fi
  if [ "$attempt" -eq 1 ] && grep -qi "CUDA out of memory\|OutOfMemoryError" runs/triage_boost_smoke2.log; then
    echo "[boost-triage] $(date '+%F %T') smoke attempt 1 hit CUDA OOM (looks like a boundary blip, not a bug) -- one retry after 15s"
    sleep 15
    continue
  fi
  echo "[boost-triage] $(date '+%F %T') PRE-FLIGHT SMOKE FAILED (rc=$smoke_rc, attempt=$attempt) -- --boost-frac has a real bug, not stopping to retry-loop it. See runs/triage_boost_smoke2.log. ABORTING, full sweep NOT started."
  exit 1
done
echo "[boost-triage] $(date '+%F %T') pre-flight smoke passed -- proceeding to full test819 sweep"

for bf in 0.0 0.2 0.5; do
  echo "[boost-triage] $(date '+%F %T') start full test819 boost_frac=$bf"
  bash scripts/run_pre_supervised.sh "$ADAPTER" "$HEAD" "runs/triage_boost_${bf}" \
    --split test --tta 8 --motion-weight 0.3 --boost-frac "$bf"
  echo "[boost-triage] $(date '+%F %T') done boost_frac=$bf, grading against local key"
  "$PY" grade/grade.py "runs/triage_boost_${bf}/submission.csv" > "runs/grade_boost_${bf}.txt" 2>&1
done

echo "[boost-triage] $(date '+%F %T') all boost_frac values done -- summary:"
for bf in 0.0 0.2 0.5; do
  echo "=== boost_frac=$bf ==="
  cat "runs/grade_boost_${bf}.txt" 2>/dev/null
  echo
done
