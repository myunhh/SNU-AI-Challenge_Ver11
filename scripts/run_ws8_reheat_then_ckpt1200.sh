#!/bin/bash
# Chain: reheat-adapter(motion 0.3, 학습+추론 모두) test 추론 -> checkpoint-1200
# 다운로드 대기 -> checkpoint-1200 test 추론(motion 0, 즉 --motion-weight 미지정 =
# 추론만 motion 미적용). 둘 다 test.csv 819건, run_pre.py 그대로 재사용(새 추론
# 코드 없음). 각 단계 로그는 runs/<out>.log에 남아 term_dashboard.py --infer-out가
# 그대로 읽는다.
#
# 🔴 2026-07-23 정정: checkpoint-1200은 "pre-motion 베이스"가 아니다 —
# runs/sft32b_v11_ws8_checkpoint-1200/train_args.json에 motion_weight: 0.3으로
# 찍혀 있고(train_sft.py의 save()가 체크포인트 가중치와 같은 호출에서 원자적으로
# 쓰는 값이라 신뢰 가능), scripts/post_train_eval.sh 주석도 "serving-matched
# protocol: motion-weight 0.3"이라 학습이 이미 motion=0.3이었음을 전제하고 있다.
# 즉 아래 checkpoint-1200 추론은 "학습 0.3 / 추론 0(미지정)"인 학습-서빙 불일치
# 상태다 — "motion 켬 vs 끔"을 깨끗이 비교한 게 아니다. 이 체인의 결과(reheat 패
# — EM 740 vs 747)는 "motion 신호 유무" 판정이 아니라 "동일 motion=0.3 레시피
# 위에 300스텝을 더 얹었더니(+추론 motion도 같이 켰더니) 나빠졌다"는 결과로 읽어야
# 한다. 진짜 "motion=0.3 serving-matched"를 보려면 checkpoint-1200 그대로(재학습
# 없이) --motion-weight 0.3을 추론에 넘겨 다시 돌려봐야 한다(미검증, 저비용).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src
export LD_LIBRARY_PATH=$HOME/anaconda3/envs/py3_11/lib
PY=$HOME/anaconda3/envs/py3_11/bin/python

REHEAT_ADAPTER="runs/sft32b_v11_ws8_reheat_adapter_final/adapter_final/adapter"
REHEAT_HEAD="runs/sft32b_v11_ws8_reheat_adapter_final/adapter_final/head.pt"
REHEAT_OUT="runs/test_v11_ws8_reheat"

CKPT1200_ADAPTER="runs/sft32b_v11_ws8/checkpoint-1200/adapter"
CKPT1200_HEAD="runs/sft32b_v11_ws8/checkpoint-1200/head.pt"
CKPT1200_OUT="runs/test_v11_ckpt1200"
EXPECT_SIZE=537005424   # 이 LoRA 구성(r16, 7proj)의 다른 모든 체크포인트와 동일한 크기

echo "[chain] ===== reheat 어댑터(motion_weight=0.3) test 819건 시작 $(date '+%F %T') ====="
if [ -f "$REHEAT_OUT/report.json" ]; then
  echo "[chain] reheat 이미 완료 — 건너뜀"
else
  "$PY" run_pre.py --adapter "$REHEAT_ADAPTER" --head "$REHEAT_HEAD" \
      --motion-weight 0.3 --out "$REHEAT_OUT" 2>&1 | tee "$REHEAT_OUT.log"
  RC=${PIPESTATUS[0]}
  echo "[chain] reheat 완료 rc=$RC $(date '+%F %T')"
  if [ $RC -ne 0 ]; then
    echo "[chain] reheat 실패 — checkpoint-1200 진행 보류, 확인 필요"; exit $RC
  fi
fi

echo "[chain] ===== checkpoint-1200 다운로드 대기 $(date '+%F %T') ====="
while true; do
  if [ -f "$CKPT1200_ADAPTER/adapter_model.safetensors" ] && [ -f "$CKPT1200_HEAD" ]; then
    SZ=$(stat -c%s "$CKPT1200_ADAPTER/adapter_model.safetensors" 2>/dev/null || echo 0)
    if [ "$SZ" = "$EXPECT_SIZE" ]; then
      echo "[chain] checkpoint-1200 크기 일치($SZ bytes) — 완료로 판단 $(date '+%F %T')"
      break
    fi
    # 크기가 다르면(다운로드 중 or 예상외 구성) 안정화될 때까지 재확인
    sleep 15
    SZ2=$(stat -c%s "$CKPT1200_ADAPTER/adapter_model.safetensors" 2>/dev/null || echo 0)
    if [ "$SZ" = "$SZ2" ] && [ "$SZ" != "0" ]; then
      echo "[chain] checkpoint-1200 크기($SZ bytes)가 예상($EXPECT_SIZE)과 다르지만 15초간 안정 — 진행"
      break
    fi
  fi
  sleep 15
done

echo "[chain] ===== checkpoint-1200(학습 motion_weight=0.3, 추론은 --motion-weight 미지정=0) test 819건 시작 $(date '+%F %T') ====="
if [ -f "$CKPT1200_OUT/report.json" ]; then
  echo "[chain] checkpoint-1200 이미 완료 — 건너뜀"
else
  "$PY" run_pre.py --adapter "$CKPT1200_ADAPTER" --head "$CKPT1200_HEAD" \
      --out "$CKPT1200_OUT" 2>&1 | tee "$CKPT1200_OUT.log"
  RC=${PIPESTATUS[0]}
  echo "[chain] checkpoint-1200 완료 rc=$RC $(date '+%F %T')"
fi

echo "[chain] 전체 체인 완료 $(date '+%F %T')"
