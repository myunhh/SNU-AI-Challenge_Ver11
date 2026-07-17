#!/usr/bin/env bash
# A100 one-command launcher.
#   bash scripts/run_a100.sh            # smoke -> ws8 warm-start SFT (확정 레시피, 2026-07-16)
#   bash scripts/run_a100.sh smoke      # GPU smoke only
#   bash scripts/run_a100.sh ws8        # Ver8 warm-start SFT (transfer gate -> 1500 steps)
#   bash scripts/run_a100.sh sft        # cold-start SFT (구 경로, 인자 수동 지정)
#   bash scripts/run_a100.sh dpo        # DPO phase 2 on the ws8 SFT output
#   bash scripts/run_a100.sh test       # test 819 -> submission.csv
# Extra args pass through, e.g.: bash scripts/run_a100.sh ws8 --steps 1000
#
# ws8 env knobs:
#   SNUAI_WS8_ADAPTER  Ver8 챔피언 어댑터 경로 (기본 ../Ver8/runs/checkpoint-200-Ver8 DPO)
#                      A100 박스에 Ver8 워크스페이스가 없으면 dev box에서 그 디렉터리만
#                      scp로 가져와 경로를 지정할 것 (adapter_config.json + 537MB safetensors).
#   SNUAI_SKIP_GATE=1  10스텝 warm-start 전이 게이트 생략
#   SNUAI_NGPU         GPU 수 강제 (기본: torch.cuda.device_count() — 2 이상이면 DDP torchrun)
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

# --- ws8 warm-start recipe (검증 근거는 CLAUDE.md "Ver8 warm-start 재본런" 절) ---
# body-lr 5e-5: Ver8 DPO 가중치 보존 (2e-4는 초반에 챔피언 지식을 덮어씀)
# lr-ratio 5(기본) -> head peak 2.5e-4: 1차 본런 실측 플래토-탈출 임계(2.87e-4) 바로 아래
# accum 16: 유효배치 4->16, 플래토 근본 원인(그래디언트 노이즈) 직접 해결
# cosine(기본): poly는 총 LR 예산을 body 기준 ~20x 깎는 것으로 확정되어 기각
# kt-weight 0.5(기본, 2026-07-17): 기대-Kendall 보조항 — LB 쌍순서 부분점수 정렬 +
#   Ver8 DPO가 심어둔 인접스왑 마진 구조를 SFT가 씻어내지 않게 유지. 끄려면 --kt-weight 0.
WS8_ADAPTER="${SNUAI_WS8_ADAPTER:-../Ver8/runs/checkpoint-200-Ver8 DPO}"
WS8_RECIPE=(--adapter "$WS8_ADAPTER" --body-lr 5e-5 --accum 16 --schedule cosine)
WS8_OUT="runs/sft32b_v11_ws8"

launch_fit() {  # DDP 자동 감지: GPU 2장 이상이면 torchrun (--accum 16은 world_size 2로 나누어떨어짐)
  local ngpu="${SNUAI_NGPU:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
  if [ "$ngpu" -ge 2 ]; then
    torchrun --nproc_per_node=2 run_fit.py "$@"
  else
    python run_fit.py "$@"
  fi
}

run_ws8() {
  if [ ! -f "$WS8_ADAPTER/adapter_config.json" ]; then
    echo "[ws8] adapter not found: $WS8_ADAPTER (set SNUAI_WS8_ADAPTER)" >&2
    exit 1
  fi
  if [ -z "${SNUAI_SKIP_GATE:-}" ]; then
    echo "[ws8] transfer gate: 10 steps (warmup 중이라 LR 미미 -> 초기 loss = 전이 품질 측정)"
    rm -rf runs/ws8_gate
    launch_fit "${WS8_RECIPE[@]}" "$@" --steps 10 --out runs/ws8_gate 2>&1 | tee runs/ws8_gate.log
    python - <<'PY'
import json, sys
rows = [json.loads(l) for l in open("runs/ws8_gate/train_log.jsonl")]
# "ce"(KT 보조항 제외한 순수 CE)로 판정해야 ln(24)=3.178 기준이 유지된다.
# 구 로그(ce 필드 없음) 호환으로 loss 폴백.
loss = sum(r.get("ce", r["loss"]) for r in rows) / len(rows)
ok = loss < 3.0
print(f"[ws8-gate] mean initial CE = {loss:.3f} -> {'PASS' if ok else 'FAIL'} (threshold 3.0)")
if not ok:
    print("[ws8-gate] warm-start가 전이되지 않음 (ln24=3.178 근방 = 무전이).")
    print("[ws8-gate] 1순위 용의자는 해상도 이동 — --max-pixels 602112 (Ver8 원 해상도)로 게이트 재시도 권장.")
sys.exit(0 if ok else 1)
PY
  fi
  echo "[ws8] full warm-start SFT -> $WS8_OUT"
  launch_fit "${WS8_RECIPE[@]}" --steps 1500 --out "$WS8_OUT" "$@" 2>&1 | tee -a runs/sft_v11_ws8.log
}

case "$STAGE" in
  smoke)
    python scripts/smoke_gpu.py --train "$@"
    ;;
  ws8)
    run_ws8 "$@"
    ;;
  sft)
    python run_fit.py "$@" 2>&1 | tee -a runs/sft_v11.log
    ;;
  dpo)
    python run_fit.py --phase dpo --adapter "$WS8_OUT/adapter_final/adapter" "$@" 2>&1 | tee -a runs/dpo_v11.log
    ;;
  test)
    python run_pre.py --adapter "$WS8_OUT/adapter_final/adapter" "$@" 2>&1 | tee -a runs/test_v11.log
    ;;
  auto)
    echo "[run_a100] stage 1/2: GPU smoke (parity + back>0 gate)"
    python scripts/smoke_gpu.py --train
    echo "[run_a100] stage 2/2: Ver8 warm-start SFT"
    run_ws8 "$@"
    ;;
  *)
    echo "unknown stage: $STAGE (smoke|ws8|sft|dpo|test)" >&2
    exit 1
    ;;
esac
