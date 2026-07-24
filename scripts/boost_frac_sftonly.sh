#!/bin/bash
# 2026-07-24 ~09:0x: corrected boost sweep. Prior boost_frac_triage.sh /
# boost_frac_remaining.sh runs used runs/ckpt1200-dpo-new (a DPO-on-ckpt1200
# adapter) as the base -- reasonable when launched (DPO's own verdict wasn't
# back yet), but that DPO run graded WORSE than the confirmed real-LB
# champion (est 0.9011 vs SFT-only ckpt1200's real LB 0.92670), so boosting
# on top of it is no longer the useful experiment. User caught this and
# asked to switch base to SFT-only ckpt1200 + motion_weight=0.3 serve.
# Those DPO-adapter boost results are preserved, not deleted:
#   runs/triage_boost_0.0_dpoadapter (complete, backfilled)
#   runs/triage_boost_0.5_dpoadapter_partial194 (killed at 194/819)
#
# boost_frac=0.0 for THIS (correct) sweep is backfilled from
# runs/test_v11_ckpt1200_tta8_motion03/submission.csv -- config-identical
# (same adapter, motion=0.3, TTA8, no boost), already complete+graded
# (EM 753/819=0.9194, est 0.9109 -- notably higher local numbers than the
# confirmed-champion's own motion=0.0 serve variant, est 0.9036 for real LB
# 0.92670 -- so this motion=0.3 serve config may be a strong candidate on
# its own merit, independent of boost).
#
# No pre-flight smoke repeated here: the boost code path (vlm.py
# forward_prepared idx-branch, M-RoPE/deepstack duplication) already ran
# cleanly for 194/819 real samples on this exact model architecture with
# boost_frac=0.5 (zero crashes, zero OOM, zero shape errors) before being
# killed only for using the wrong ADAPTER -- swapping which LoRA adapter is
# loaded doesn't change tensor shapes or the boost logic, so the earlier
# smoke's mechanical validation still holds. Re-smoking would just cost
# time with no real risk reduction.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/py3_11/lib
PY=$HOME/anaconda3/envs/py3_11/bin/python
ADAPTER=runs/sft32b_v11_ws8_checkpoint-1200/adapter
HEAD=runs/sft32b_v11_ws8_checkpoint-1200/head.pt

for bf in 0.5 0.2; do
  echo "[boost-sftonly] $(date '+%F %T') start full test819 boost_frac=$bf (SFT-only ckpt1200, motion=0.3)"
  bash scripts/run_pre_supervised.sh "$ADAPTER" "$HEAD" "runs/triage_boost_${bf}" \
    --split test --tta 8 --motion-weight 0.3 --boost-frac "$bf"
  echo "[boost-sftonly] $(date '+%F %T') done boost_frac=$bf, grading against local key"
  "$PY" grade/grade.py "runs/triage_boost_${bf}/submission.csv" > "runs/grade_boost_${bf}.txt" 2>&1
done

echo "[boost-sftonly] $(date '+%F %T') both remaining boost_frac legs done -- summary:"
for bf in 0.0 0.2 0.5; do
  echo "=== boost_frac=$bf ==="
  cat "runs/grade_boost_${bf}.txt" 2>/dev/null
  echo
done
