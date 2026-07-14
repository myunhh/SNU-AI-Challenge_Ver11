# SNU AI Challenge 2026 — Ver11

**Cross-Targeted FitPrune(4×4) + One-pass score24 + Stackelberg 두-시간축 QLoRA**

셔플된 비디오 프레임 4장 + 스토리라인 캡션 → 시간순 재배열(24-way).
베이스: `unsloth/Qwen3-VL-32B-Instruct-bnb-4bit` (NF4 사전양자화, vision 비양자화).
설계 정본: [VER11.md](VER11.md).

## A100 퀵스타트 (3줄 + 데이터)

```bash
git clone https://github.com/myunhh/SNU-AI-Challenge_Ver11 && cd SNU-AI-Challenge_Ver11
pip install -r requirements-gpu.txt
# 데이터(~4GB): 개발 머신에서 rsync -a --copy-links <dev>:~/SNU-AI-Challenge/Ver11/data/ ./data/
#   (또는 bash scripts/pack_data.sh 로 만든 tar.gz를 SNUAI_DATA_URL 로 지정)
python run_fit.py                      # 학습 (스모크 게이트까지 원하면: bash scripts/run_a100.sh)
```

모델은 로컬 경로가 없으면 HF hub에서 자동 다운로드(~19GB).
`$SNUAI_MODEL_ID`로 오버라이드 가능.

## 명령어

```bash
pytest tests/ -q                                   # CPU 안전망 (48개)
python scripts/smoke_gpu.py --train                # GPU 스모크: parity·프루닝·back>0 게이트
python run_fit.py                                  # SFT 2000스텝 → runs/sft32b_v11/
python run_fit.py --phase dpo \
  --adapter runs/sft32b_v11/adapter_final/adapter  # DPO(인접스와프 마진) 400스텝
python run_pre.py --holdout-val --adapter <ADPT>   # 홀드아웃 945 평가 (EM·쌍순서)
python run_pre.py --adapter <ADPT>                 # test 819 → runs/test_v11/submission.csv
python scripts/ab_gate.py runs/cal_A runs/cal_B --name X   # 채택 게이트 (paired bootstrap)
```

플래그 요약: `--no-prune`(프루닝 끔) · `--keep-ratio 0.5` · `--diversity-frac 0.2` ·
`--tta 3` · `--tau 0.10`(캐스케이드 문턱) · `--uniform-lr`(Stackelberg ablation 대조군).

## 검증 현황 (2026-07-14, RTX 4090)

| 검증 | 결과 |
|---|---|
| CPU 테스트 | 48개 통과 (순열규약·TTA리맵·분해·프루닝선택·제출형식·마진손실) |
| 수술 경로 parity (8B·32B) | 스톡 forward 대비 max\|diff\| = **0.000e+00** |
| 프루닝 forward (32B-4bit) | visual 50% 컷, 0.41s/forward (full 0.57s), VRAM 피크 **19.9GiB** |
| 학습 신호 (8B 4bit 3스텝) | body·head 모두 grad>0 (back>0) |
| 추론 E2E (TTA3+캐스케이드) | 정상 (~1s/샘플@8B; 32B 예상 2~5s ≪ 105s 하드리밋) |

## 주의

- **transformers==5.12.1 고정** — 프루닝 수술 경로가 이 버전 소스와 라인 단위로 검증됨.
  올리려면 `scripts/smoke_gpu.py`의 parity 체크를 반드시 재통과시킬 것.
- conda env에서 pandas GLIBCXX 에러 시: `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib`
  (스크립트들은 자동 처리).
- 홀드아웃 945는 **Ver11 자체 분할**(sha1 기반) — 이전 버전들의 홀드아웃 수치와 직접 비교 금지,
  모든 A/B는 Ver11 내부에서만.
