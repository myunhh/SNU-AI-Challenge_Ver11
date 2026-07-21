# CLAUDE.md — Ver11

셔플된 비디오 프레임 4장 + 캡션 → 시간순 재배열(4!=24). 이 저장소는 Ver10까지의 코드를
상속하지 않은 **독립 구현**이다(지식은 `../PROJECT_SUMMARY.md`에서 계승).

구성: Cross-Targeted FitPrune(4×4 캡션-교차 50% 토큰 컷) + one-pass score24 헤드
(lm_head 글자행 초기화) + Stackelberg 두-시간축 QLoRA(body 2e-4 / head 1e-3, head wd 0.1)
+ 기대-Kendall 보조손실(`--kt-weight 0.5`, 07-17) + 균형 TTA4(Klein 세트) + 풀토큰 stage-2
(기본 always — `--stage2 cascade`가 구 τ=0.10 마진 캐스케이드). 설계 근거·논문 매핑·하이퍼
파라미터 전체는 `../PROJECT_SUMMARY.md`의 "Ver11 설계 상세"~"본런 전 정확도 개선(07-17)" 절,
공통 규정·전 버전 함정은 `../CLAUDE.md` 참고.

안전 하한 = **Ver4 32B-4bit ckpt1600+TTA3, LB 0.90226** — 단 현재 대회 전체 챔피언은 **Ver8 DPO
checkpoint-600 +TTA5, LB 0.91797**(2026-07-21 밤 승격, 상세는 `../CLAUDE.md`). Ver11은
대체가 아니라 초과 도전이었으나 07-21 재검증으로 챔피언 미달 판정, 트랙 종결(아래 "이 버전
고유 함정" 최신 절 참고).

## 🚀 A100 박스 임무 지시서 (2026-07-20 — 이 박스의 Claude가 처음 읽을 것)

VESSL A100 박스(2×A100 기준)에서 이 절만 따라가면 본런 전 과정이 완료된다. 이 저장소는
독립 clone이라 **`../CLAUDE.md`·`../TODO.md`·`../Ver8` 같은 상위 참조는 이 박스에 없다** —
필요한 결정·규정은 전부 이 절에 요약돼 있고, 없는 파일을 찾아 헤매지 말 것.

### 확정 결정사항 (dev box에서 2026-07-20 사용자 확정)

- **warm-start 어댑터 = Ver8 DPO checkpoint-600 (현 챔피언, LB 0.91099).** 아래 07-16 절의
  ckpt200(0.90401) 언급은 당시 챔피언 기준의 옛 기록이다. 같은 DPO 궤적이라 LoRA 구성 동일
  (r16/α32/7proj·언어 64층, peft 0.19.1) — 전이 실패는 어차피 10스텝 게이트가 걸러준다.
  dev box에서 scp로 들어온 `DPO-checkpoint-600/`(adapter_model.safetensors +
  adapter_config.json)를 `export SNUAI_WS8_ADAPTER=$PWD/DPO-checkpoint-600`으로 지정.
- **레시피는 러너 기본값 그대로** (body-lr 5e-5 / accum 16 / cosine / steps 1500 /
  kt-weight 0.5 / out `runs/sft32b_v11_ws8`). **하이퍼파라미터 임의 변경 금지** — 전부
  1차 본런 실측 진단에서 유도된 값이다(아래 07-16 절).

### 실행 순서

```bash
# 0) 준비물 점검 — 하나라도 없으면 진행하지 말고 사용자에게 요청
ls data/train.csv data/test.csv DPO-checkpoint-600/adapter_model.safetensors
bash scripts/setup_a100.sh            # deps + base 모델은 HF hub 자동(unsloth 32B bnb-4bit)

# 1) 본런 (tmux 'snuai11'로 detach — SSH 끊겨도 유지)
export SNUAI_WS8_ADAPTER=$PWD/DPO-checkpoint-600   # tmux 첫 실행 전에 export해야 상속됨
bash scripts/run_a100.sh              # auto: smoke_gpu → 10스텝 전이 게이트 → 1500스텝

# 2) 본런 완료가 확인되면 사용자 승인 대기 없이 곧바로 이어서 실행 (학습→추론 한 번에)
bash scripts/run_a100.sh test         # test 819 → submission.csv 생성 (LB 제출은 하지 말 것)
```

- **VRAM 피크 상시 기록(3090 이식성 판정 + 체크포인트별 프로파일)**: A100은 80GB라 OOM이
  안 나지만, 최종 서빙 후보 환경은 24GB급(3090/4090)이다. **학습 시작 전에** 백그라운드
  폴러(예: `nvidia-smi --query-compute-apps=pid,used_memory` 1~2초 간격 → timestamp와 함께
  `runs/vram_log.jsonl`에 append하는 소형 스크립트)를 띄워 **본런~test 추론 전 구간**을 감시
  하고, 종료 후 타임스탬프로 구간을 나눠 ① **체크포인트 저장 주기(200스텝)별 학습 피크 VRAM**
  ② **test 추론 전체 피크 VRAM**을 표로 정리해 `runs/vram_peaks.json`에 저장할 것.
  **트레이너/추론 코드 자체는 수정하지 말 것**(본런 직전 코드 변경 금지 — 외부 폴러 방식이라
  코드 무변경으로 가능). 추론 피크가 **23GiB를 넘으면**(24GB 카드 가용분 초과) — **추론은
  중단하지 말고 끝까지 진행하되**, 완료 보고에 "**3090(24GB)에서는 OOM 예상**"이라고 피크
  수치와 함께 별도 항목으로 명시할 것. 참고 기준: 구 파이프라인(TTA3+cascade)의 4090 실측
  피크는 19.9GiB였고, 신규 기본값(TTA4 + stage-2 always)은 이보다 높아질 수 있어 이번 A100
  런이 첫 실측이다.
- **진행 대시보드**: 본런이 궤도에 오르면(게이트 통과 후) **모니터링 대시보드를 만들어
  주기적으로 갱신**할 것 — Artifact 도구가 가능하면 Artifact 페이지로, 아니면
  `runs/dashboard.html` 단일 HTML 파일로. 내용: ① 게이트 판정(ce 평균 vs 3.0) ② train_log.jsonl
  기반 loss/ce/kt 곡선 + windowed acc ③ 스텝 진행률·ETA·경과 시간(과금 감각) ④ 체크포인트별
  학습 피크 VRAM 표 ⑤ test 추론 단계 진행률과 피크 VRAM(23GiB 기준선 표시, 3090 OOM 판정
  포함). 데이터 소스는 위 `train_log.jsonl`·`vram_log.jsonl`이면 충분하고, 갱신 주기는
  체크포인트 저장(200스텝)마다면 된다.

### 게이트·모니터링 판정 기준

- **전이 게이트(10스텝, `runs/ws8_gate.log`)**: `train_log.jsonl`의 **`ce` 필드**(총 loss 아님 —
  kt 보조항 포함된 `loss`로 판정 금지) 평균 **< 3.0** 필수. ln24≈3.178 근방이면 무전이 —
  `bash scripts/run_a100.sh ws8 --max-pixels 602112`로 게이트만 1회 재시도해 해상도 이동 원인을
  분리하고, **그래도 ≥3.0이면 즉시 중단하고 사용자 보고**(과금 중이다 — 임기응변으로 레시피를
  바꿔가며 재시도하지 말 것).
- **DDP 첫 실검증이 이 박스에서 일어난다**(2×A100에서 돌려본 적 없음, 아래 함정 절 참고).
  게이트 10스텝이 곧 스모크다 — `train_log.jsonl`에 loss/ce/kt가 정상 스케일로 찍히는지,
  워닝·NCCL 에러가 없는지 확인하고 본런 진입을 지켜본 뒤 자리를 떠라.
- **본런 중**: ce가 시작부터 3.0 훨씬 아래(warm-start 효과)여야 정상. 체크포인트는 200스텝마다
  `runs/sft32b_v11_ws8/`에 저장. 오래(수십 분) 멈춘 것 같으면 rank0 생존 확인 — rank0가
  체크포인트 저장 중 죽으면 나머지가 NCCL barrier에서 무한대기한다(아래 함정 절). 죽었으면
  전체 kill 후 재개하되, **재개 방법은 즉흥 조합하지 말고 `run_fit.py` 코드에서 resume 시맨틱을
  확인한 뒤** 진행(불명확하면 사용자 보고).

### 금지사항 (대회 규정 + 트랙 원칙 요약 — 위반 시 실격 소지)

- **LB 제출 금지** — submission.csv 생성까지만. 제출 슬롯(1일 2회)은 사용자가 dev box에서 조율.
- **자기(Claude) 생성 텍스트를 학습 데이터에 넣지 말 것** — 외부 상용 API 산출물의 학습 투입은
  규정 3.2 위반. 코드 작성·로그 판독은 무관.
- 홀드아웃 분할 재도입 금지(이 트랙은 07-15부로 전량 학습 + LB 판정) · test.csv 라벨 추정/분석
  금지(규정 3.4) · 다른 모델과의 출력 앙상블 금지 · transformers==5.12.1 업그레이드 금지(parity
  재통과 없이는).
- 학습·추론 실행에는 항상 진행바가 보이게 유지(기존 코드에 있음 — 새 스크립트를 짜게 되면 동일 적용).

### 완료 보고 (사용자에게 전달할 것)

① 게이트 ce 평균과 판정 ② 본런 loss/ce 곡선 요약(플래토 없이 하강했는지, 최종 windowed acc)
③ 총 소요 시간(과금 판단용) ④ `submission.csv` 경로 ⑤ **test 추론 피크 VRAM과 3090(24GB)
OOM 여부 판정**(23GiB 초과 시 "3090에서는 OOM 예상"을 별도 항목으로) + **체크포인트별 학습
피크 VRAM 표**(`runs/vram_peaks.json`) ⑥ 대시보드 위치(Artifact URL 또는 `runs/dashboard.html`)
⑦ 이상 징후 전부. 이후의 (선택) DPO
400스텝·`avg_adapters.py` 소프·S3 paired A/B는 **사용자 지시가 있을 때만** 진행.

## 진입점

```bash
pytest tests/ -q                                    # CPU 안전망 (82개)
python scripts/smoke_gpu.py --train                 # GPU 스모크: parity·프루닝·back>0 게이트
bash scripts/run_a100.sh ws8                        # ★ 확정 레시피: Ver8 warm-start SFT
                                                    #   (전이 게이트 10스텝 → 1500스텝, DDP 자동)
python scripts/auto_pipeline.py                     # SFT 완료 대기→체크포인트 스윕(in-sample)→
                                                    #   1·2등 실검증(실제 test+grade.py)→DPO 600
                                                    #   고정→최종 test+grade.py 1회, 전부 자동
                                                    #   (results.jsonl 캐시로 재실행시 이어감).
                                                    #   2026-07-21 실측: DPO가 역효과였던 사례
                                                    #   있음 — 위 "이 버전 고유 함정" 최신 항목 참고
python run_fit.py                                   # (구) cold-start SFT 2000스텝 → runs/sft32b_v11/
python run_fit.py --phase dpo \
  --adapter runs/sft32b_v11/adapter_final/adapter    # DPO(인접스와프 마진) 400스텝
python run_pre.py --adapter <ADPT>                   # test 819 → runs/test_v11/submission.csv

# 멀티GPU DDP (2026-07-16 추가, 2장) — device_map을 LOCAL_RANK로 고정 + 스텝당 1회 수동
# grad all-reduce(SUM). --accum(기본4)은 world_size로 나누어떨어져야 함.
torchrun --nproc_per_node=2 run_fit.py --out runs/sft32b_v11_ddp
torchrun --nproc_per_node=2 run_fit.py --phase dpo \
  --adapter runs/sft32b_v11_ddp/adapter_final/adapter --out runs/dpo32b_v11_ddp
```

플래그: `--no-prune` · `--keep-ratio 0.5` · `--diversity-frac 0.2` · `--tta 4`(기본, 균형 Klein 세트;
`3`=구 시드셔플) · `--stage2 {always,cascade,off}`(추론 풀토큰 2단계 정책, 기본 always) ·
`--tau 0.10`(cascade 모드에서만) · `--kt-weight 0.5`(기대-Kendall 보조손실, 0=순수 CE/DPO) ·
`--uniform-lr`(Stackelberg ablation 대조군).

## 검증 현황 (2026-07-15, RTX 4090 — 로컬 검증 전부 완료)

parity(스톡 forward 대비 max|diff|=0, 8B·32B) · 프루닝 50% 컷 정상(VRAM 피크 19.9GiB) ·
back>0(학습 신호) · E2E 캐스케이드 정상 발동(margin<0.10만 escalate, 819건 4090 ~29분 —
구 TTA3+cascade 기준 실측; 07-17 신규 기본값 TTA4+stage2 always는 ~1.7×인 ~50분 추정).
남은 건 A100 SFT 본런 → (선택)DPO → test → LB 슬롯뿐.

## 이 버전 고유 함정

- **2026-07-21 A100 auto_pipeline 본런 완료 — SFT 실검증이 in-sample 오버피팅을 실제로
  잡음, 이번 라운드 DPO는 오히려 역효과, TTA4+stage2=always VRAM 25.7~26GiB 확인(24GB
  카드 OOM 확정)** (진입점: `python scripts/auto_pipeline.py` — SFT 체크포인트 스윕
  (in-sample) → 1·2등만 실제 test 819건+grade.py 실검증 → DPO(600 고정, 스윕 없음) →
  최종 test+grade.py 1회. 도중 TTA5~8·DPO 200~800 체크포인트 스윕·다후보 비교는 전부
  폐기하고 Ver8 자체 실측 근거로 DPO600+TTA4 고정 채택 — 근거는 `scripts/auto_pipeline.py`
  모듈 docstring 참고):
  - **SFT 실검증이 실제로 순위를 뒤집었다**: in-sample(train 400건) 1등은 checkpoint-1400
    (pairwise 0.9950)이었지만 실제 test 819건+grade.py로는 adapter_final(est_lb 0.9169)이
    checkpoint-1400(0.9157)을 역전했다 — in-sample 순위를 그대로 믿었으면 틀린 체크포인트
    로 DPO를 시작할 뻔한 실사례. 체크포인트 스텝이 늘수록 in-sample pairwise가 계속
    우상향(0.9708→0.9950)하는 건 여전히 오버핏/암기 혼입 가능성이 있어 상대 순위
    참고용일 뿐임을 재확인.
  - **🔴 이번 라운드 DPO(600스텝, base=adapter_final)가 실제 test에서 역효과**: DPO 전
    SFT-only est_lb 0.9169(EM 739/819, per-position 0.9417, pairwise 0.9697) → DPO 후
    est_lb **0.9035**(EM 728/819, per-position 0.9335, pairwise 0.9670) — 4개 지표가
    전부 같은 방향으로 나빠져 측정 노이즈(±0.010)로 보기 어렵다. DPO 학습 로그 자체는
    정상(loss 1.0→0.57, acc 0.5→0.7, 발산 없음) — DPO 목적함수는 잘 최적화됐는데 실제
    test 일반화는 오히려 깎였다. 가설(미검증): SFT 단계의 `--kt-weight 0.5`(기대-
    Kendall)가 이미 pairwise-order를 타겟팅하는데 그 위에 DPO의 인접스왑 마진 손실
    (`--dpo-ce-weight 0.2`/`--dpo-beta 1.0`)을 또 얹어 같은 목적을 과교정했을 가능성 —
    Ver8 자체 DPO(순수 CE SFT 위에서 ckpt200→600, +0.007)와 달리 이번 SFT는 이미 kt-aux로
    보정된 상태라 전제가 다름. **(주의: 0.9169/0.9035 전부 grade.py 추정치, 실제 LB
    미제출 — 현재 실제 LB 챔피언은 여전히 Ver8 ckpt600 0.91099)**. 현재 최고 실측(추정)
    후보는 DPO 없는 SFT-only adapter_final(`runs/auto_pipeline/eval_sft_test/adapter_final/
    submission.csv`)이고, 파이프라인이 자동 복사해둔 `runs/test_v11_final/submission.csv`
    (DPO600 결과, 더 낮음)를 그대로 최종으로 오인하지 말 것 — 사용자 판단 대기 중.
  - **VRAM: TTA4+stage2=always 순수 추론 피크 25.7~26GiB 확인 → 24GB 서빙 카드 OOM
    확정**: 학습 종료 후 순수 추론 구간만 걸러 측정(`runs/vram_log.jsonl`) — 전체 피크
    25.7GiB, 최종 DPO600+TTA4 test 추론(819건) 구간 단독으로도 피크 **25.69GiB**, 샘플의
    58.5%가 23GiB 초과(`runs/auto_pipeline/vram_over23_final.jsonl`에 2,238건 기록).
    원인: `predict_sample()`이 TTA 4개 뷰의 `Prepared`(풀길이 비전 인코딩)를 전부 동시에
    들고 있고, `--stage2 always`가 전 샘플에 대해 매번 풀토큰(비프루닝) 재forward를
    하기 때문 — 프루닝은 stage-1 LLM 입력 길이만 줄일 뿐, 비전 인코딩 자체와 stage-2
    풀토큰 패스는 프루닝과 무관하게 그대로 무겁다. 구버전(TTA3+cascade, 4090 실측
    19.9GiB)은 margin<0.10인 어려운 샘플만 이 풀토큰 패스를 탔던 것과 대조. **수정은
    설계만 하고 미적용**(다른 환경에서 별도 테스트 예정) — 우선순위: ① `--max-pixels`
    하향(예: 602112, 코드 무변경·즉시 가능) ② `stage2=="always"`일 때 뷰별로
    stage1+stage2를 바로 이어 끝내고 그 뷰의 prep을 즉시 해제(동시 보유 4→1, parity
    재검증 필요) ③ `--stage2 cascade`는 평균 비용만 줄이고 최악피크는 못 줄여 ①과
    병행 필요.
  - (사소, 수정됨) `scripts/auto_pipeline.py`의 `grade_submission()`이 grade.py 출력
    `"... : 0.9157  (+/- ~0.010)"`을 파싱할 때 `split("+/-")[0].strip()`으로는 여는
    괄호가 안 지워져 크래시하는 버그가 있었음(정규식으로 수정, 이미 끝난 test 추론
    결과는 재사용하도록 크래시 복구 로직도 추가).

- **🔴 2026-07-21 밤 dev box 재검증 — 위 07-21 A100 절의 3대 결론 전부 정정** (상세·근거는
  `docs/postmortem_2026-07-21_a100_and_plan.md` 정본, 원 텍스트는 사료로 보존):
  - **est_lb 0.9169는 LB 추정치가 아니다**: 어떤 로컬 키 버전(v1~v13)으로도 재현 불가.
    ws8-ckpt600↔챔피언 일치도 756/819에서 외삽하면 adapter_final↔챔피언 일치도 ≈758/819
    −0.0085 = 정확히 0.9169 — **챔피언 제출물을 truth로 잘못 잡은 자기-일치도**로 판정.
    정직한 재채점(key v13): ws8-ckpt600 = 745/819(est 0.9011) vs 챔피언 767/819. pos/pw
    지표상 adapter_final도 동급 → **Ver11은 챔피언 대비 약 −2pp, 트랙 종결. LB 슬롯 금지.**
  - **"TTA4+stage2 always = 24GB OOM 확정"은 반증됨**: 동일 설정(1,126,400px·프루닝 on)으로
    로컬 4090(24GB)에서 test 819건 완주(07-21 02:13, 총 2.57h, 평균 11.3s/건, OOM 없음 —
    `runs/test_ver11_ckpt600/`). A100의 25.7GiB는 토치 할당자 캐시가 포함된 nvidia-smi
    프로세스 관측치로, 80GB 카드에선 캐시를 반납할 압력이 없어 live 피크보다 크게 찍힌다.
    3090은 여전히 미실측(동급 24GB라 통과 가능성 높음). 설계해둔 뷰-순차 처리 수정(②)은
    보류 — 서빙 코드는 동결이 우선.
  - DPO 역효과(−11 EM)는 같은 계기 내 상대 비교라 방향 유효 — 단 절대 수준이 챔피언 이하
    구간에서의 악화라 재도전 가치 없음.
  - 부수 확정 사실: **공개 LB = 573/819(70%) 그리드, 최종 순위는 비공개 30%**(공식 페이지
    확인) · **grade.py est는 상위권(±1pp) 판별 불능**(실LB 순위를 역순으로 추정한 실측
    사례 — champ_tta4/dpo600/ver4). 키는 26건 육안 감수로 v13 갱신(`../grade/`), 챔피언
    다음 후보는 Ver8 쪽 균형 8뷰(BALANCED8, 코드·테스트·스크립트 준비 완료)로 이관.

- **2026-07-17 A100 본런 전 정확도 개선 4건 + DDP 로그 버그 수정** (CPU 테스트 82개 전부 통과,
  이전 파이프라인은 아래 플래그로 재현 가능):
  - **🔴 DDP 로그 loss 스케일 버그 수정(본런 전 필수였음)**: 이전 코드는 rank당
    step_loss(=Σ loss/accum, 자기 rank 몫만)를 로그 윈도우에 쌓아 **2-GPU에서 train_log.jsonl의
    loss가 실제의 1/world_size로 찍혔다**. 이대로면 ws8 전이 게이트(threshold 3.0)가 무전이
    (ln24≈3.178 → 로그 ~1.59)도 **거짓 PASS**시키고, "플래토 탈출 시점" 판독도 전부 왜곡됐을 것.
    샘플 단위 기록으로 교체(단일 GPU 로그 값은 수학적으로 동일 — 그래디언트/가중치는 원래 무관).
    게이트는 이제 `"ce"` 필드(KT 보조항 제외 순수 CE, 구 로그는 loss 폴백)로 판정.
  - **기대-Kendall 보조 손실(`--kt-weight`, 기본 0.5, SFT·DPO 공통)**: `loss += λ·E_{c~p}[KT(c,GT)/6]`.
    LB가 쌍순서(1−KT/6) 부분점수라는 정황(Ver7 포렌식)에 학습목표를 직접 정렬하고, Ver8 DPO(현
    챔피언, 0.90226→0.90401이 정확히 인접스왑 마진 학습의 이득)가 심어둔 오류 기하를 순수 CE
    SFT가 씻어내지 않게 유지. one-hot GT에서 정확히 0이라 CE 최적점과 충돌 없음, 균등분포에서
    정확히 0.5(S4 평균 KT=3). train_log에 `loss`(총)/`ce`/`kt` 분리 기록 — **ln(24) 플래토 판독은
    ce 기준**. `--kt-weight 0`이면 이전 손실과 완전 동일.
  - **균형 TTA4(추론 기본 `--tta 4`)**: Klein 4원군 {e,(01)(23),(02)(13),(03)(12)} — 4뷰에 걸쳐
    각 입력이 각 슬롯을 **정확히 1회씩** 방문(sharply transitive)해 슬롯-위치 편향이 기대값이
    아니라 정확히 상쇄. TTA 자체는 홀드아웃 +3.81pp로 검증된 메커니즘(Ver3)의 강화이며 샘플 무관
    고정 세트라 완전 결정적. `--tta 3`은 구 시드셔플 경로와 바이트 동일하게 보존.
  - **stage-2 정책(추론 `--stage2`, 기본 always)**: 풀토큰 forward 집계를 저마진 캐스케이드에서
    전 샘플로 확장 — 풀토큰 패스는 프루닝이 버린 증거를 되살리는 정보 추가이고(τ 게이트는 원래
    효율 장치), test 819건 기준 시간 ~1.7×(4090 ~50분 추정)로 24h 예산 대비 무료에 가까움.
    `--stage2 cascade`가 기존 τ=0.10 캐스케이드와 동작 동일(진행 기록의 `margin`은 계속 stage-1
    마진으로 τ 포렌식 호환, `margin_final` 필드 신설). `--no-prune`이면 stage-2 자동 생략.
  - **(옵션) `scripts/avg_adapters.py`**: 동일 런 tail 체크포인트 균등 평균(SWA류 soup). 러너
    비연결 사후 도구 — in-sample 평가로 raw 최종본 대비 우위 확인 후에만 고려. 단일 모델 산출이라
    출력-앙상블 금지 조항과는 무관하지만, 쓸 경우 재현성 검증 대비 설명 준비(독립 런 간 평균 금지,
    같은 trajectory만).

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
  없음)이라 DDP wrapper를 써도, Ver10의 all-or-nothing 조건부 backward 교착 버그도 원래 없었지만,
  두 트레이너의 방식을 통일해뒀다. body/head 파라미터 초기화(신규 LoRA일 때)는 `torch.manual_seed
  (args.seed)`가 모델 로딩 전 모든 rank에 동일하게 걸려 있어 rank 간 일치 보장(CUDA RNG도 함께
  시딩됨을 PyTorch 소스로 확인 — `torch.cuda.manual_seed_all` 경유). **A100 2장 환경에서 실제로
  검증 안 됨(GPU 없는 환경에서 작성)** — 본런 전 `--steps 2`짜리 짧은 스모크로 먼저
  `train_log.jsonl`에 정상 loss가 찍히는지 확인할 것.
- **⚠️ 2026-07-16 재검토로 발견·수정된 로그 오염 버그**: `running`/`hits`의 rank 간 `all_gather_object`
  병합을 매 스텝 부르고 있었는데, rank0가 자기 변수를 병합 결과로 덮어쓴 뒤 다음 스텝에 또 다른
  rank의 "그 시점까지의 로컬 히스토리 전체"를 다시 흡수해 O(n²)로 부풀며 초반 값이 중복 반영되는
  버그였다(그래디언트/가중치엔 영향 없음, `train_log.jsonl`의 loss/acc만 왜곡). 이번 SFT 본런에서
  확인하려는 게 정확히 "loss가 ln(24) 플래토를 언제 벗어나는지"라 로그가 왜곡되면 그 판단 자체가
  무의미해질 뻔했다 — gather를 `log_boundary`(10스텝 윈도우) 안으로 옮겨 윈도우당 1회만 하도록 고침.
- **잔여 리스크(코드로 못 막음)**: rank0가 체크포인트 저장 도중 죽으면 다른 rank는 그 barrier에서
  NCCL 기본 타임아웃까지 대기 — 오래 멈춘 것 같으면 rank0 생존을 확인 후 죽여서 마지막 체크포인트
  (`save_steps`마다 저장)에서 재개할 것.
  fresh clone에서 데이터가 없을 때 여러 rank가 동시에 다운로드하는 경합은 `run_common.ensure_data`가
  rank0만 다운로드하고 나머지는 폴링 대기하도록 막아둠(2026-07-16).
- **2026-07-16 SFT 1차 본런 학습 불안정 → Ver8 warm-start 재본런 레시피로 확정** (4-에이전트
  검증 워크플로 + S1 데이터 게이트 완료, 진입점: `bash scripts/run_a100.sh ws8`):
  - **진단 확정치**: 1차 본런(cosine/accum4/head 1e-3)은 step 1260~1290(예산 63% 소진)에야
    플래토 탈출, 탈출 시점 head lr = 0.287~0.301×peak(≈2.9e-4). 탈출 후 남은 cosine LR 면적은
    전체의 7.3%뿐인데 그걸로 acc 0.40까지 감 — 종료 시점에도 미수렴(LR 고갈로 중단된 상태).
  - **poly 스케줄 기각**: `stackelberg.py`의 poly는 warmup 직후부터 즉시 감쇠(step 20에 body
    이미 0.161×peak)라 2000스텝 총 LR 예산이 cosine 대비 body 19.9×/head 6.3× 작음. 8B 진단런
    (step 130 중단)은 무결론으로 종결 — 재개하지 않는다. lr_head/lr_body 비율이 로그에서
    3→8로 벌어지는 건 버그가 아니라 이론 스케줄(지수 0.6/0.4 차이)임.
  - **재본런 = Ver8 챔피언 warm-start**(사용자 결정): `--adapter "../Ver8/runs/checkpoint-200-Ver8
    DPO"`(LB 0.90401). 호환성 검증 완료 — base(unsloth 32B 4bit)·LoRA(r16/α32/타겟 7proj·언어
    64층)·peft 0.19.1 전부 일치, 코드 변경 불필요. head.pt가 없으므로 `init_from_lm_head`로
    초기화되는데 이게 곧 Ver8의 "24글자 제한 로짓 스코어링"을 수학적으로 정확히 재현하는
    올바른 warm-start 경로. ln(24) 플래토는 시작점이 유능해지므로 원천 제거됨.
  - **레시피**: body-lr **5e-5**(2e-4는 DPO 정련 가중치를 덮어씀), lr-ratio 5(기본; head peak
    2.5e-4 = 실측 탈출 임계 바로 아래), cosine(기본), **accum 16**(유효배치 4→16 — 플래토 근본
    원인 직접 해결, DDP 2장이면 rank당 8), steps 1500(24k 샘플 ≈2.5 epoch, save 200마다),
    out `runs/sft32b_v11_ws8`. 프롬프트·max_pixels(1,126,400)는 Ver11 기본 유지 — Ver8과의
    분포 이동(프롬프트 문구/캡션 위치/범례 형식, 해상도 602,112→1,126,400 ≈ 비주얼 토큰 2배,
    FitPrune 미경험)이 SFT가 적응할 대상 그 자체다.
  - **전이 게이트**(러너에 내장): 본런 전 10스텝 런의 평균 loss < 3.0 필수(warmup 중이라 LR
    미미 → 초기 loss = 전이 품질). ≥3.0(ln24 근방)이면 무전이 — 1순위 용의자는 해상도 이동,
    `--max-pixels 602112`로 게이트만 재시도해 원인 분리.
- **2026-07-16 FitPrune 스코어링 기하 확정 — 모달리티별 자기-평균 센터링**: 코사인 스코어는 시각
  토큰을 이미지 센트로이드로, 텍스트 앵커를 캡션 앵커 평균(μ_T)으로 **각자** 센터링한 뒤 계산
  (`fitprune.per_event_scores`). 이전 형태(비센터링 → 시각 센트로이드를 텍스트에도 적용)는 공유
  방향이 지배해 4개 이벤트 맵이 사실상 동일해지는 rank-1 퇴화가 있었다(`runs/prune_viz` 실측;
  32B embed 테이블 400캡션 실측 — 이벤트 평균방향 쌍코사인 raw +0.20 → μ_T 센터링 후 −0.33 ≈
  4벡터 이론 최대 대비 −1/3). diversity(farthest-point) 선택도 같은 센터링 공간에서 수행(비센터링
  코사인은 좁은 밴드에 몰려 선택이 노이즈에 지배됨). 전수 9,535 train 캡션에서 단일-앵커 퇴화 0건
  (1-word 캡션용 raw-방향 폴백은 코드에 있음). 이벤트별 시각화는 반드시 4이벤트 전체를 넘긴
  `per_event_scores`의 행을 써야 선택 경로와 같은 μ_T가 적용된다(1이벤트만 재채점 금지 —
  `visualize_prune.py`는 반영됨). **keep-set이 바뀌므로 이 날짜 이전 프루닝 선택과의 A/B 비교
  금지**(시간·VRAM 측정치는 유효, parity 게이트는 무관).
- **2026-07-16 FitPrune stuff-over-things 수정 — 객체성 블렌드 + MMR 선택 (기본 on)**: `runs/prune_viz`
  실측에서 캡션 장면명사(pool/ice/snow)와 매칭되는 대면적 배경 텍스처가 top-k 예산을 중복으로
  독식하고, 순서 판별의 실제 단서인 소형 전경 물체(스키어·공·흰모자·얼굴)가 잘리는 패턴 확인.
  수정 2개: ① `objectness_weight`(기본 0.3) — 센트로이드 residual의 norm(정규화가 버리던 크기
  정보 = 전경성)을 per-image min-max 후 코사인과 블렌드, ② `mmr_lambda`(기본 0.5) — top-k+
  diversity-fill을 greedy MMR(`score − λ·relu(kept와의 최대 코사인)`)로 대체, 중복 배경이 소수
  대표로 압축되고 예산이 novel 토큰으로 흐름. λ 하한은 유도값: 두 항이 [0,1] 스케일이라 최악
  케이스(전경=코사인 최저, 배경=완전중복 최고) 구제에 λ > 1−2w = 0.4 필요, 0.5는 여유분.
  `--objectness-weight 0 --mmr-lambda 0`이 이전 파이프라인을 정확히 재현(min-max는 단조라 랭킹
  불변, 테스트로 고정). **keep-set이 또 바뀌므로 이 시점 이전 선택과 A/B 금지.** 가중치는
  튜닝값이 아닌 구조적 선택 — S3 슬롯이 나면 paired A/B로 ablate. 동일 샘플 3건 재렌더로
  정성 게이트 통과(스키어·공·수영모·얼굴·렌즈 전부 keep-set 복귀). 전체 경위·근거·잔여
  개선안은 `docs/fitprune_fix_2026-07-16.md` 참고.
- **프루닝 재설계(per-event/배정-인지)는 S1 게이트 불통과로 보류(2026-07-16)**: train 캡션
  9,535건 중 94%가 자연 절(clause) 4개 미만(1절 35%/2절 45%) → "4이벤트"는 대부분 기계적
  단어-중간 분할이라 "이벤트↔프레임 1:1 배정"의 전제가 성립 안 함(복합 퇴화 51.5% > kill 기준
  30~40%). 16변형 추론 원안은 3중 결함(1이미지 입력에서 score24 정의 불능 — 이미지당 ≥1토큰만
  학습됨 / decompose는 순서 무관 COVER 시맨틱이라 이벤트 순서≠시간 순서 / 충돌-재예측 미정의,
  결정적 24순열 브루트포스가 상위호환)으로 기각. max-pool은 union 시맨틱이라 파편 품질에
  강건 — 현행 유지가 데이터로 재확인됨. 잔여 후속: "병합=유실" 가설의 이득 상한 측정(S3:
  train 서브셋 500~1000건 `--no-prune` vs 현행 paired A/B, 4090 빌 때) — 격차 ≈0이면 이 계열
  재설계 전체 종결.
