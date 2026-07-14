# CLAUDE.md — Ver11

셔플된 비디오 프레임 4장 + 캡션 → 시간순 재배열(4!=24). **설계 정본: [VER11.md](VER11.md)**,
빠른 시작: [README.md](README.md). 이 저장소는 Ver10까지의 코드를 상속하지 않은 **독립 구현**이다
(지식은 `../PROJECT_SUMMARY.md`에서 계승).

구성: Cross-Targeted FitPrune(4×4 캡션-교차 50% 토큰 컷) + one-pass score24 헤드
(lm_head 글자행 초기화) + Stackelberg 두-시간축 QLoRA(body 2e-4 / head 1e-3, head wd 0.1)
+ TTA3 + 마진 캐스케이드(τ 미만이면 풀토큰 재검).

## 진입점

- `python run_fit.py` (학습) / `python run_pre.py` (추론) — clone+pip 후 바로 동작
- `bash scripts/run_a100.sh [smoke|sft|dpo|holdout|test]` — 스테이지 런처
- `python scripts/smoke_gpu.py --train` — parity·프루닝·back>0 게이트 (**모델/transformers를 바꾸면 필수 재실행**)

## 함정 (이 저장소에서 실제로 밟았거나 코드가 방어 중인 것)

- **transformers==5.12.1 고정.** 프루닝은 `Qwen3VLTextModel`을 직접 호출하는 수술 경로
  (`snuai11/vlm.py` 참조 — `Qwen3VLModel.forward`는 deepstack 인자를 안 받음). 버전을 올리면
  smoke의 parity 체크(스톡 forward와 max|diff|=0)로 검증할 것.
- **사전양자화 32B의 vision 재양자화 버그**: bnb는 사용자 quantization_config를 무시하므로
  (merge에 loading attribute 없음) `config.quantization_config`의 skip 목록을 직접 패치해야
  한다 — `vlm._patch_skip_modules` + `verify_vision_not_quantized`가 처리·검증. 실측으로
  한 번 재현 후 수정됨(2026-07-14).
- **Answer = rank** (`Answer[i-1]` = Input_i의 시간순 순위, 1-indexed). order 인코딩과 섞으면
  비자기역원 순열에서 조용히 틀린다. 순열 연산은 `snuai11/perm.py` 함수만 사용(직접 구현 금지),
  `tests/test_perm.py`가 방어선. 제출 형식은 공백 포함 `"[1, 2, 3, 4]"`.
- **position_ids는 (3,B,L) 시맨틱 M-RoPE 좌표** — 프루닝은 마지막 차원 index-select만 허용.
  2D/None을 넘기면 텍스트 RoPE로 조용히 격하된다(에러 없음).
- **캡션 분해는 규칙 기반만**(`decompose.py`) — 생성 텍스트 학습 투입 금지 규정.
- 홀드아웃 945는 **Ver11 자체 sha1 분할** — 이전 버전 수치와 절대치 비교 금지. 채택은
  `scripts/ab_gate.py`(paired bootstrap, ΔEM≥+2pp AND CI 하한>0, 쌍순서 병행)로만.
- conda env `py3_11` 사용. pandas GLIBCXX 에러 → `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib`.
  editable install이 Ver1을 가리키는 전역 함정 → 항상 `PYTHONPATH=$PWD/src` (스크립트 자동 처리).

## 규정 요약 (위반 = 실격)

오프라인 추론, 3090 24GB 1대·24h/819건, 모델 총량 80GB, 외부 API·외부 데이터·앙상블 금지,
생성 모델 증강 금지, 2026-05-31 이전 공개 모델만, 허용 기법 = Quantization·LoRA·CoT·TTA.
모든 튜닝 결정은 홀드아웃으로만(test 분포 참조 금지).
