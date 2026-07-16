# CLAUDE.md — Ver11

셔플된 비디오 프레임 4장 + 캡션 → 시간순 재배열(4!=24). 이 저장소는 Ver10까지의 코드를
상속하지 않은 **독립 구현**이다(지식은 `../PROJECT_SUMMARY.md`에서 계승).

구성: Cross-Targeted FitPrune(4×4 캡션-교차 50% 토큰 컷) + one-pass score24 헤드
(lm_head 글자행 초기화) + Stackelberg 두-시간축 QLoRA(body 2e-4 / head 1e-3, head wd 0.1)
+ TTA3 + 마진 캐스케이드(τ 미만이면 풀토큰 재검). 설계 근거·논문 매핑·하이퍼파라미터 전체는
`../PROJECT_SUMMARY.md`의 "Ver11 설계 상세" 절, 공통 규정·전 버전 함정은 `../CLAUDE.md` 참고.

안전 하한 = **Ver4 32B-4bit ckpt1600+TTA3, LB 0.90226** — 단 현재 대회 전체 챔피언은 Ver8 DPO
LB 0.90401(`../TODO.md` 참고). Ver11은 대체가 아니라 초과 도전.

## 진입점

```bash
pytest tests/ -q                                    # CPU 안전망 (47개)
python scripts/smoke_gpu.py --train                 # GPU 스모크: parity·프루닝·back>0 게이트
python run_fit.py                                   # SFT 2000스텝, train 100%(9,535) → runs/sft32b_v11/
python run_fit.py --phase dpo \
  --adapter runs/sft32b_v11/adapter_final/adapter    # DPO(인접스와프 마진) 400스텝
python run_pre.py --adapter <ADPT>                   # test 819 → runs/test_v11/submission.csv

# 멀티GPU DDP (2026-07-16 추가, 2장) — device_map을 LOCAL_RANK로 고정 + 스텝당 1회 수동
# grad all-reduce(SUM). --accum(기본4)은 world_size로 나누어떨어져야 함.
torchrun --nproc_per_node=2 run_fit.py --out runs/sft32b_v11_ddp
torchrun --nproc_per_node=2 run_fit.py --phase dpo \
  --adapter runs/sft32b_v11_ddp/adapter_final/adapter --out runs/dpo32b_v11_ddp
```

플래그: `--no-prune` · `--keep-ratio 0.5` · `--diversity-frac 0.2` · `--tta 3` · `--tau 0.10` ·
`--uniform-lr`(Stackelberg ablation 대조군).

## 검증 현황 (2026-07-15, RTX 4090 — 로컬 검증 전부 완료)

parity(스톡 forward 대비 max|diff|=0, 8B·32B) · 프루닝 50% 컷 정상(VRAM 피크 19.9GiB) ·
back>0(학습 신호) · E2E 캐스케이드 정상 발동(margin<0.10만 escalate, 819건 4090 ~29분).
남은 건 A100 SFT 본런 → (선택)DPO → test → LB 슬롯뿐.

## 이 버전 고유 함정

- **transformers==5.12.1 고정** — 프루닝은 `Qwen3VLTextModel`을 직접 호출하는 수술 경로
  (`Qwen3VLModel.forward`는 deepstack 인자를 안 받음). 버전을 올리면 `smoke_gpu.py`의 parity
  체크(스톡 forward와 max|diff|=0) 재통과 필수.
- **position_ids는 (3,B,L) 시맨틱 M-RoPE 좌표** — 프루닝은 마지막 차원 index-select만 허용.
  2D/None을 넘기면 텍스트 RoPE로 조용히 격하된다(에러 없음).
- bnb는 5.12.1의 merge 시 사용자 `quantization_config`를 무시함 — `config.quantization_config`의
  skip 목록을 직접 패치해야 vision 재양자화가 막힘(`vlm._patch_skip_modules`).
- **로컬 홀드아웃 폐지(2026-07-15)**: train 9,535건 전량 학습. τ·keep-ratio는 Ver4/Ver10 트랙
  검증치 상속, 프루닝 on/off 같은 신규 설계 선택은 LB 제출로만 판정 — 이 판정 방식 자체의 데이터
  누수 소지는 `../TODO.md`·`../PROJECT_SUMMARY.md` §6 참고(아직 미확인 상태).
- **DDP는 Ver10/grpo.py와 같은 패턴**(`DistributedDataParallel` wrapper 미사용, 스텝 끝 1회 수동
  `all_reduce(SUM)`) — 여기는 backward 호출횟수가 매 스텝 항상 `local_accum`으로 고정(조건부 스킵
  없음)이라 DDP wrapper를 써도 안전했겠지만, 두 트레이너의 방식을 통일해뒀다. body/head 파라미터
  초기화(신규 LoRA일 때)는 `torch.manual_seed(args.seed)`가 모델 로딩 전 모든 rank에 동일하게
  걸려 있어 rank 간 일치 보장. **A100 2장 환경에서 실제로 검증 안 됨(GPU 없는 환경에서 작성)** —
  본런 전 `--steps 2`짜리 짧은 스모크로 먼저 `train_log.jsonl`에 정상 loss가 찍히는지 확인할 것.
  fresh clone에서 데이터가 없을 때 여러 rank가 동시에 다운로드하는 경합은 `run_common.ensure_data`가
  rank0만 다운로드하고 나머지는 폴링 대기하도록 막아둠(2026-07-16).
- **2026-07-16 SFT 1차 본런 학습 불안정 진단 진행 중** — loss가 `ln(24)`(≈3.178) 근방에서 예산의
  약 65%를 허비하다 코사인 후반에야 풀리는 문제 발견(`../PROJECT_SUMMARY.md` §2 참고). 로컬 4090
  8B 진단런(accum16/poly/lr-ratio3)으로 원인 검증 시도했으나 GPU가 중간에 다른 작업(Ver8-Refactored
  reranker 학습)에 밀려 step 130에서 결론 없이 중단됨 — A100 재본런 전에 이 진단부터 재확인하거나,
  8×A100처럼 GPU가 여유 있으면 진단 없이 바로 DDP로 유효배치를 늘려(accum을 world_size로 분산해도
  전체 accum은 그대로 유지되므로, `--accum` 자체를 8~16으로 올리고 world_size로 나누는 방식을 권장)
  재본런하는 것도 근본 원인(유효배치 과소) 직접 해결책이 될 수 있음.
