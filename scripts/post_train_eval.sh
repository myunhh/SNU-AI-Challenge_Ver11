#!/usr/bin/env bash
# Motion-blend round: after the ws8 SFT run finishes, in-sample sweep over
# late checkpoints (serving-matched protocol: TTA4 + stage2 always +
# motion-weight 0.3) to shortlist test candidates. Idempotent — finished
# sweep entries are skipped, so it can be re-run after interruptions.
# Shortlisting only (2026-07-21 postmortem: in-sample rank is NOT the final
# judge) — the real verdict is test 819 + grade.py est vs baseline 0.9011.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD"
OUT=${1:-runs/sft32b_v11_ws8}
SWEEP=runs/sweep_insample
mkdir -p "$SWEEP"
for ck in checkpoint-200 checkpoint-600 checkpoint-1000 checkpoint-1200 checkpoint-1400 adapter_final; do
  adpt="$OUT/$ck/adapter"
  if [ ! -f "$adpt/adapter_config.json" ]; then
    echo "[sweep] skip $ck (adapter missing)"
    continue
  fi
  if [ ! -f "$SWEEP/$ck/eval.json" ]; then
    echo "[sweep] $ck ..."
    python run_pre.py --split train --eval --limit 300 --tta 4 --stage2 always \
      --motion-weight 0.3 --adapter "$adpt" --out "$SWEEP/$ck" \
      > "$SWEEP/$ck.log" 2>&1 || { echo "[sweep] $ck FAILED (see $SWEEP/$ck.log)"; continue; }
  fi
  echo "[sweep] $ck: $(tr -d '\n' < "$SWEEP/$ck/eval.json")"
done
echo "[sweep] done"
