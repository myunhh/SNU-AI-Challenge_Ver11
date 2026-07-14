# Ver11 설계·수행 계획서 — Cross-Targeted FitPrune + Stackelberg 두-시간축 score24

> 상태(2026-07-14): **구현 완료·GPU 검증 완료(8B/32B parity·back>0)**, A100 학습 대기.
> 안전 하한 = **Ver4 32B-4bit ckpt1600+TTA3, LB 0.90226**(현 챔피언) — Ver11은 대체가 아니라 초과 도전.
> 마감 07-24 23:59, 제출 1일 2회.

## 1. 제안 배경과 핵심 아이디어

**문제**: 4장 고해상도 프레임은 비주얼 토큰(기본 예산 시 샘플당 ~4.4k)이 지배해 32B급
VLM의 VRAM·프리필을 압박하고, 캡션과 무관한 배경 노이즈가 순서 판별 신호를 희석한다.
일괄(텍스트 무관) 토큰 프루닝은 "캡션에 안 적힌 미세 단서"를 날려 Exact Match를 위협한다.

**솔루션 3축** (전부 구현·검증됨):

1. **Cross-Targeted FitPrune (4×4)** — 캡션을 규칙 기반으로 4개 시간축 이벤트로 분해하고,
   셔플 때문에 이미지↔이벤트 대응을 모르므로 **각 이미지를 4개 이벤트 전부와 교차 스코어링**
   (max-pool over events). 상위 80%의 keep 예산은 중요도로, 20%는 **diversity 토큰**
   (기선택 집합과 최소 코사인 유사도, LearnPruner의 max-min greedy)으로 채워 이미지당
   비주얼 토큰 50%만 유지. 스코어링은 LLM 임베딩 공간의 텍스트-비주얼 유사도(단일 matmul)라
   **추가 forward·attention 실체화가 전혀 없다**(SDPA/FlashAttention 유지, FitPrune의
   cross-신호 우위 결론을 training-free로 이식).
2. **One-pass score24 + Score24Head** — 24개 순열을 단일 forward의 24-로짓으로 동시 채점.
   헤드는 Linear(5120→24, bias 없음)이며 **lm_head의 A~X 글자 행으로 초기화**되어 초기값이
   제약 디코딩과 수학적으로 동일한 지점에서 출발. 생성이 없으므로 파싱 실패가 구조적으로 0%.
3. **Stackelberg 두-시간축 최적화 (ICML 2026)** — 바디 M(LoRA 어댑터, leader)에 α=2e-4,
   헤드 w(follower)에 β=5α=1e-3을 부여하는 **단일 루프 동시 업데이트**. 헤드가 w\*(M)을
   실시간 추적하면 바디가 내려가는 축약 목적함수 Φ(M)=f(M,w\*(M))의 초기 곡률이 날카로워져
   (논문 실측 λmax 4~8배) 학습 초반 평탄 지대를 빨리 탈출한다. **헤드 weight decay 0.1은
   이론의 하중 조건**(Assumption 3.1: w에 대한 강볼록성 → w\*(M) 유일성)이라 기본 탑재.

## 2. 아키텍처

### 학습 (A100)

```
train.csv 9,535 ── sha1 분할 ──► fit 8,590 / holdout 945(불가침)
      │
      ▼ (스텝마다)
uniform 순열 증강(σ~U(S4), rank′=rank∘σ)
      │
      ▼
┌─ prepare (no_grad — vision·embed 동결) ──────────────────────────────┐
│ processor(4 img, legend 프롬프트) → input_ids/pixel_values/grid      │
│ visual tower → per-image merged tokens [Nᵢ,5120] + deepstack 3장     │
│ masked_scatter(전체 placeholder) → inputs_embeds                     │
│ get_rope_index → position_ids (3,1,L) [시맨틱 M-RoPE 좌표]           │
└──────────────────────────────────────────────────────────────────────┘
      │
      ▼ p=0.75로 프루닝 적용 (25%는 풀토큰 → 캐스케이드 경로도 학습 분포에 포함)
┌─ Cross-Targeted FitPrune ─────────────────────────────────────────────┐
│ 캡션 → 규칙 분해 C1..C4 → content-word 임베딩 [Tⱼ,5120]              │
│ scoreᵢₖ = max_j max_t cos(visᵢₖ, Cⱼₜ)   (4×4 교차, max-pool)          │
│ keep = top-40%(중요도) ∪ 10%(diversity max-min greedy) → 인덱스 정렬  │
│ inputs_embeds/position_ids/attn/vismask/deepstack 동일 마스크 슬라이스│
└──────────────────────────────────────────────────────────────────────┘
      │
      ▼ language_model 직접 호출 (use_cache=False, grad ckpt)
┌─ Qwen3-VL-32B LLM (NF4 동결 + LoRA r16, 언어 레이어 full-path 한정) ──┐
│  = Stackelberg Body M, lr α_k (cosine, 2e-4)                          │
└──────────────────────────────────────────────────────────────────────┘
      │ last_hidden[마지막 프롬프트 위치] (post-RMSNorm)
      ▼
┌─ Score24Head (fp32 Linear 5120→24, lm_head 글자행 초기화) ────────────┐
│  = Stackelberg Head w, lr β_k = 5α_k, weight decay 0.1               │
└──────────────────────────────────────────────────────────────────────┘
      │
      ├─ [SFT 2000스텝]  CE(logits24, label)
      └─ [DPO 400스텝]   -log σ(β(z_gt − z_neg)),  neg = 인접스와프 3종(KT=1)
                          + 0.2·CE 앵커   ← 홀드아웃 최빈 오답 유형 직격
```

### 추론 (RTX 3090/4090, 오프라인)

```
샘플 ──► TTA 뷰 3개 (항등 + 샘플 id 시드 셔플 2, 결정적)
           │  뷰마다: prepare(비전 1회) → 보관(스테이지2 재사용)
           ▼
Stage 1: 프루닝(50%) forward ×3 → 24-로짓 → softmax → 뷰별 원공간 리맵
           → Laplace 로그평균 집계 → margin = p₁−p₂
           │
   margin ≥ τ(0.10) ──► 즉시 확정 (재검 없음)
   margin < τ        ──► Stage 2: 같은 prepare 텐서로 풀토큰 forward ×3
                          (비전 재계산 0, "같은 질문 재질의"가 아니라 정보 추가)
                          → 6개 뷰-점수 통합 집계 → 확정
           │
           ▼
Answer "[r1, r2, r3, r4]" (공백 포함) → submission.csv (sample_submission 대조)
```

## 3. 논문 → 구현 매핑 (실제 PDF 정독 결과 기준)

| 논문 | 가져온 것 | 버린 것(근거) |
|---|---|---|
| **FitPrune** (AAAI'25, 2409.10197) | ① text-conditioned cross 신호가 지배적(Table 5: cross-only 60.1 vs self-only 55.4 GQA) → 이벤트-교차 스코어링의 근거 ② 50% keep = 안전지대(50% 비율에서 −0.1 GQA) ③ 이중 기준 교집합 정신 → 중요도+diversity 결합 | 레이어별 레시피·이진탐색·attention 통계 수집(전부 LLM 내부 attention 실체화 필요 — 우리는 pre-LLM 1회 컷이라 무의미, SDPA 유지가 더 큼). **주의: 논문의 "-0.5%"는 in-LLM 스케줄 기준 — pre-LLM 컷은 그대로 이전 안 됨. 우리 완충 2중: 프루닝 켠 채 QLoRA 학습 + 저마진 풀토큰 캐스케이드** |
| **Stackelberg** (ICML'26, 28198) | ① 단일 루프 동시 업데이트(내부 루프 금지 — 논문이 분석한 그대로) ② β/α = 5 (검증 밴드 3~10×) ③ **헤드 wd 0.1 = 강볼록성 조건** ④ 불안정 시 헤드 LR만 낮춤(Uniform-Large가 문서화된 실패 모드) ⑤ `--uniform-lr` 3-arm ablation 대조군 | 다중 follower 스텝(논문에 없음), 이론 스케줄 강제(실험은 상수 LR — 우리는 cosine 기본 + `--schedule poly`로 이론 모드 옵션). **주의: O(k^{-2/3})는 Φ 강볼록 조건부 — 보고서에 무조건 보장으로 쓰지 말 것** |
| **LearnPruner** (ICLR'26, 15007) | ① diversity 토큰 max-min greedy 알고리즘 원문 그대로(λ 소량, keep의 10~20%) ② attention 스코어의 position-bias 경고 → 임베딩 유사도 채택 근거 강화 ③ "저신뢰 샘플=stage-1 정보손실" 실패 분석 → 마진 캐스케이드 정당화 | LPM(0.53M 학습 모듈 — 백본 구조 변형, 앙상블/단일모델 규정 회색), stage-2 in-LLM 프루닝(§7 확장으로 설계만 보존 — 이득이 89%+ 극한 프루닝 영역 수치이고 우리 50% keep에선 한계효용 미미 + attention 실체화 필요) |

## 4. 대회 규정 준수 매핑

| 규정 | Ver11 대응 |
|---|---|
| 단일 모델·앙상블 금지 | 백본 1개 + 그 파생물(LoRA·lm_head 행 초기화 헤드)만. 별도 모델 0 (SigLIP2류 배제). TTA·캐스케이드는 동일 모델 반복 호출(허용 범주) |
| 생성 텍스트 학습 투입 금지 | **캡션 분해는 규칙 기반만**(decompose.py, 모델 호출 0). synthetic CoT 없음 |
| 허용 기법 목록 | Quantization(NF4)·LoRA·TTA 그대로. 프루닝은 입력 전처리·경량화 범주 |
| 3090 24GB·24h·오프라인 | 실측 VRAM 피크 19.9GiB(4090, 32B 추론). 샘플 ~2-5s ≪ 105s/샘플. HF_HUB_OFFLINE 리허설은 제출 전 필수(§6) |
| 05-31 이전 공개 모델 | Qwen3-VL-32B-Instruct(공식) 사전양자화판 |
| 데이터 누수 금지 | 홀드아웃 945 sha1 분할, split.json 기록, 모든 τ·keep-ratio·채택 결정은 홀드아웃 A/B(ab_gate.py)로만 |

## 5. 학습·추론 하이퍼 (기본값 = 코드 기본값)

- LoRA: r16 / α32 / dropout 0.05, 언어 레이어 q,k,v,o,gate,up,down full-path만(vision·lm_head·embed 제외, 로딩 후 검증 함수 강제)
- SFT: 2000스텝 × accum 4(=8k 샘플, ~1.9에폭), cosine, warmup 20, clip 1.0
- Stackelberg: body 2e-4 / head 1e-3 / head-wd 0.1 (`--lr-ratio`, `--uniform-lr`, `--schedule poly`)
- FitPrune: keep 0.5, diversity 0.2(keep의), 학습 적용확률 0.75, `--no-prune` A/B 콕
- DPO: 400스텝, body 5e-5/head 2.5e-4, β=1.0, CE 앵커 0.2
- 추론: TTA3 + τ=0.10 캐스케이드, max_pixels=1,126,400(이미지당 ≤1100토큰 캡)
- 사전양자화 로딩 함정 수정 내장: config.quantization_config의 skip 목록에
  `model.visual` 접두 경로 주입(5.12.1 merge가 bnb 사용자 config를 무시하므로 config 직접 변경)
  + `verify_vision_not_quantized` 하드 게이트

## 6. 실행 마일스톤

```bash
# M0. CPU 안전망 (완료 — 48개 통과)
pytest tests/ -q
# M1. GPU 스모크 (완료 — 8B·32B: parity 0.0, 프루닝 50%, back>0, VRAM 19.9GiB)
python scripts/smoke_gpu.py --train
# M2. A100 SFT 본런 (clone → pip → 데이터 → 원커맨드; 내장 스모크 게이트 포함)
bash scripts/run_a100.sh                      # 또는 python run_fit.py
# M3. (선택) DPO 마진 강화
bash scripts/run_a100.sh dpo
# M4. 홀드아웃 945 평가 + 게이트 (로컬 4090 가능)
python run_pre.py --holdout-val --adapter runs/sft32b_v11/adapter_final/adapter
python scripts/ab_gate.py runs/holdout_noprune runs/holdout_v11 --name fitprune   # 프루닝 자체 A/B
# M5. 통과 시에만 test 819 → LB 슬롯
python run_pre.py --adapter ...
```

**승격 조건**: 비교 기준선이 **LB 0.90226**(Ver4 ckpt1600)으로 상향된 상태.
홀드아웃은 Ver11 자체 분할이라 절대치 비교 불가 → **같은 홀드아웃에서
Ver11-노프루닝 vs Ver11-프루닝, uniform-lr vs stackelberg를 paired bootstrap으로 게이트**하고,
LB 검증은 슬롯 1회(1일 2회 중 검증용 보존)로 판정. 어느 게이트든 지면 Ver4 구성 유지.

**제출 전 필수 잔여**: ① `HF_HUB_OFFLINE=1` test 819 풀런 리허설 + 3090 환산 시간 기록
② τ·keep-ratio 홀드아웃 캘리브레이션 ③ 게이트 JSON 보존(본선 검증자료).

## 7. 리스크와 설계된 확장

1. **pre-LLM 컷의 미측정 영역**: FitPrune은 layer-0 일괄 컷을 측정한 적 없음(최악 참조치:
   uninformed 50%@layer4 = −3.2%). 완충: text-conditioned 스코어 + 학습 중 적용 + 캐스케이드.
   그래도 **M4의 프루닝 on/off A/B가 최종 심판** — 지면 `--no-prune`으로 즉시 강등 가능(코드 동일).
2. **홀드아웃 절대치 비교 불가**(자체 분할): 모든 판정은 Ver11 내부 paired A/B + LB 슬롯.
3. **AdamW+LoRA는 Stackelberg 논문 미검증 조합**(RMSProp 상수 LR까지 검증): `--uniform-lr`
   3-arm 대조가 ablation 근거를 만든다. 불안정 시 헤드 LR부터 인하(논문 지침).
4. **확장(설계만, 미탑재)** — LearnPruner stage-2: LLM layer 24/64(깊이비 0.375)에서
   text→vision attention으로 2차 컷(R1:R2=3). 필요 조건: 해당 레이어 QK 실체화 훅 +
   시퀀스 축소 후속 레이어 처리. 현 50% keep에서 한계효용이 작아 보류했으며, 속도가
   병목이 되면(예: TTA 확장 시) 1순위 재개 후보.
5. **일정**: 마감 D-10. A100 SFT ~6-10h 예상(스텝당 ~1.5-2.5s×2000×accum4).
   병목 시 ckpt-200 단위 부분 판정 가능(save-steps 200).
