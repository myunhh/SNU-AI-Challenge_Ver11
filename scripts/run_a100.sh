#!/usr/bin/env bash
# A100 one-command launcher.
#   bash scripts/run_a100.sh            # smoke(3 steps) -> full SFT 2000 steps
#   bash scripts/run_a100.sh smoke      # GPU smoke only
#   bash scripts/run_a100.sh sft        # SFT without smoke
#   bash scripts/run_a100.sh dpo        # DPO phase 2 on the SFT output
#   bash scripts/run_a100.sh test       # test 819 -> submission.csv
# Extra args pass through, e.g.: bash scripts/run_a100.sh sft --steps 1000
#
# Auto-detaches into tmux (session "snuai11") so an SSH drop doesn't kill a
# multi-hour run. Reattach anytime with: tmux attach -t snuai11
# Set SNUAI_NO_TMUX=1 to run in the foreground instead.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${TMUX:-}" ] && [ -z "${SNUAI_NO_TMUX:-}" ] && command -v tmux >/dev/null 2>&1; then
  SESSION="snuai11"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[run_a100] tmux session '$SESSION' is already running."
    echo "  attach: tmux attach -t $SESSION"
    exit 0
  fi
  cmd="$(printf '%q ' "$0" "$@")"
  tmux new-session -d -s "$SESSION" "$cmd"
  tmux set-option -t "$SESSION" remain-on-exit on
  echo "[run_a100] started in detached tmux session '$SESSION' (survives SSH disconnects)"
  echo "  attach:  tmux attach -t $SESSION"
  echo "  detach:  Ctrl-b d   |   progress: tail -f runs/*.log"
  exit 0
fi

export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
# conda libstdc++ fix (harmless if the env layout differs)
if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi
mkdir -p runs

STAGE="${1:-auto}"
[ $# -gt 0 ] && shift || true

case "$STAGE" in
  smoke)
    python scripts/smoke_gpu.py --train "$@"
    ;;
  sft)
    python run_fit.py "$@" 2>&1 | tee -a runs/sft_v11.log
    ;;
  dpo)
    python run_fit.py --phase dpo --adapter runs/sft32b_v11/adapter_final/adapter "$@" 2>&1 | tee -a runs/dpo_v11.log
    ;;
  test)
    python run_pre.py --adapter runs/sft32b_v11/adapter_final/adapter "$@" 2>&1 | tee -a runs/test_v11.log
    ;;
  auto)
    echo "[run_a100] stage 1/2: GPU smoke (parity + back>0 gate)"
    python scripts/smoke_gpu.py --train
    echo "[run_a100] stage 2/2: full SFT"
    python run_fit.py "$@" 2>&1 | tee -a runs/sft_v11.log
    ;;
  *)
    echo "unknown stage: $STAGE (smoke|sft|dpo|test)" >&2
    exit 1
    ;;
esac
