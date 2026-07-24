#!/bin/bash
# In-sample (train split, --eval) triage across the 3 ckpt1200-dpo-new
# checkpoints (600/800/adapter_final), serving-matched to today's best real
# candidate (TTA8 balanced, motion_weight=0.3, stage2=always). Cheap screen
# before committing a full test819 (~2-3h) run to any single winner —
# adjacent-checkpoint differences in this project have historically been
# ~4-7 EM/819, so a full sweep across all 3 is not worth the wall-clock.
set -u
cd "$(dirname "$0")/.."
LIMIT=150

for tag in checkpoint-600 checkpoint-800 adapter_final; do
  echo "[triage] $(date '+%F %T') start $tag"
  bash scripts/run_pre_supervised.sh \
    "runs/ckpt1200-dpo-new/$tag/adapter" \
    "runs/ckpt1200-dpo-new/$tag/head.pt" \
    "runs/triage_dpo1200_${tag}" \
    --split train --eval --limit $LIMIT --tta 8 --motion-weight 0.3
  echo "[triage] $(date '+%F %T') done $tag"
done

echo "[triage] $(date '+%F %T') all 3 checkpoints done — summary:"
for tag in checkpoint-600 checkpoint-800 adapter_final; do
  echo "=== $tag ==="
  cat "runs/triage_dpo1200_${tag}/eval.json" 2>/dev/null
  echo
done
