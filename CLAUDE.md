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
pytest tests/ -q                                    # CPU 안전망 (82개)
python scripts/smoke_gpu.py --train                 # GPU 스모크: parity·프루닝·back>0 게이트
bash scripts/run_a100.sh ws8                        # ★ 확정 레시피: Ver8 warm-start SFT
                                                    #   (전이 게이트 10스텝 → 1500스텝, DDP 자동)
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
back>0(학습 신호) · E2E 캐스케이드 정상 발동(margin<0.10만 escalate, 819건 4090 ~29분).
남은 건 A100 SFT 본런 → (선택)DPO → test → LB 슬롯뿐.

## 이 버전 고유 함정

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
