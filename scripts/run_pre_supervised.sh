#!/bin/bash
# 24GB 카드에서 run_pre.py가 VRAM 23~24GiB 경계를 오가다 프래그멘테이션 누적으로
# 이따금 CUDA OOM 나는 문제(2026-07-22 실측, ckpt1200 test 819건 중 index 728)에 대한
# 대응. progress.jsonl은 id 기준 resume이라(src/snuai11/infer.py) 프로세스가 죽어도
# 그대로 재실행하면 이어서 진행된다는 점을 이용해, 크래시(OOM 등)가 나면 그 지점을
# oom_events.jsonl에 기록하고 자동으로 재시작한다.
#
# 주의: --limit로 청크를 잘라 주기적으로 재시작하는 방식은 시도했다가 폐기함 —
# run_pre.py는 split=test일 때 매 실행 끝에 sample_submission 전체 id와 대조해
# submission.csv를 쓰므로, 일부만 처리된 상태로 끝나면 "id mismatch" ValueError로
# 항상 크래시한다(진짜 OOM이 아닌 가짜 크래시, 2026-07-23 실측). 그래서 --limit 없이
# 전체 819건을 목표로 돌리고, 재시작은 실제 크래시에만 반응(reactive)한다.
#
# 사용법:
#   bash scripts/run_pre_supervised.sh <adapter> <head> <out> [run_pre.py 추가 인자...]
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/py3_11/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$HOME/anaconda3/envs/py3_11/bin/python

ADAPTER="$1"; HEAD="$2"; OUT="$3"; shift 3
TOTAL=819

mkdir -p "$OUT"
progress="$OUT/progress.jsonl"
oom_log="$OUT/oom_events.jsonl"

while true; do
  done_n=$(wc -l < "$progress" 2>/dev/null || echo 0)
  echo "[supervisor] $(date '+%F %T') 시작/재시작: done=$done_n/$TOTAL"
  "$PY" run_pre.py --adapter "$ADAPTER" --head "$HEAD" --out "$OUT" "$@" \
      2>&1 | tee -a "$OUT.log"
  rc=${PIPESTATUS[0]}
  new_done=$(wc -l < "$progress" 2>/dev/null || echo 0)

  if [ "$rc" -eq 0 ]; then
    echo "[supervisor] 정상 종료 (done=$new_done/$TOTAL) $(date '+%F %T')"
    break
  fi

  vram=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  used="${vram%,*}"; total="${vram#*,}"
  err=$(tail -n 200 "$OUT.log" | grep -m1 -E "CUDA out of memory|Error|Killed" || echo "unknown")
  echo "{\"ts\": \"$(date -Iseconds)\", \"rc\": $rc, \"done_before\": $done_n, \"done_after\": $new_done, \"crashed_at_index\": $new_done, \"vram_used_mib\": ${used:-null}, \"vram_total_mib\": ${total:-null}, \"err\": $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$err")}" >> "$oom_log"
  echo "[supervisor] 비정상 종료 rc=$rc (done $done_n -> $new_done) — 5초 대기 후 재시작"
  sleep 5
done
