# Ver11 프루닝 개선 — 모션 신호 블렌드 구현 계획서 (2026-07-22)

**작성**: dev box Claude (Sonnet 5) · **대상**: A100 박스 Fable 세션(구현 담당) — 이 문서만
읽고 처음부터 작업 가능하도록 자기완결적으로 작성했다. 이 세션의 이전 대화 맥락은 없다고
가정할 것.

## 0. 이 문서의 목적과 전제

Ver11(캡션-조건부 vision-token 프루닝, "Cross-Targeted FitPrune")은 2026-07-21에 챔피언
대비 −2.7pp로 "종결" 판정을 받았다(`docs/postmortem_2026-07-21_a100_and_plan.md`). 오늘
(07-22) 챔피언(Ver8) 쪽에서 진행한 **캡션 의존도 진단**이 "이미지만으로도(캡션 없이)
정답률이 랜덤의 5배(0.21 vs 0.042)"라는 걸 보여줬고, 이게 Ver11의 프루닝이 **캡션 단어와
무관한 진짜 정보(움직임·전경 물체)를 구조적으로 버리고 있을 수 있다**는 가설로 이어졌다.

**이 가설을 오늘 저비용으로 검증했다**(`scripts/measure_motion_signal.py`, GPU 재사용 —
기존 `runs/frame_contrast/embeds_cache.pt` 캐시, 추가 GPU 비용 0, 17초 소요):

- **motion score(같은 격자 위치의 4프레임 간 residual norm 평균) vs objectness
  score**(2026-07-16에 이미 prune_viz로 실물 검증된 "진짜 전경 물체" 프록시)의
  **스피어만 상관 +0.693**(p25=0.65, p75=0.75) — 235개 정렬 가능 샘플 × 4이미지 = 940건.
  강하고 일관된 양의 상관 — 모션 점수가 실제 전경 물체를 잘 잡는다는 방증.
- **motion score vs 기존 캡션 코사인 서로게이트의 상관 +0.011**(p25=−0.08, p75=+0.11,
  사실상 0 근방) — 즉 모션 신호는 기존 캡션 매칭 신호와 **거의 겹치지 않는 상보적 정보**다.

이 두 수치의 조합(전경은 잘 잡음 + 기존 신호와 안 겹침)이 이번 세션에 테스트한 프루닝
관련 아이디어 중 **가장 강한 신호**다(참고용 — 접근 가능하면: cross-frame contrast는
null, 어텐션 수확은 spearman +0.066로 약함 — `scripts/measure_frame_contrast.py`·
`scripts/measure_attention_vs_cosine.py`가 이 저장소 안에 있으니 상세 수치가 필요하면
그걸 직접 재실행해서 확인 가능, 외부 문서 불필요). 그래서
"코드로 구현해서 실제 재학습까지 가보자"는 결정이 났다.

**⚠️ 이 문서가 명시하는 게이트·검증 절차는 협상 대상이 아니다.** 오늘 세션에서 두 번
실측된 교훈 — ① Ver11의 in-sample 체크포인트 순위가 실제 test 순위와 달랐다(ckpt1400이
in-sample 1등이었지만 실제로는 adapter_final에 짐), ② 같은 날 DPO 라운드가 학습 로그는
멀쩡했는데 실제 test에서 역효과였다 — 를 반드시 반영해서, **중간 게이트를 건너뛰거나
in-sample 지표만으로 성공을 선언하지 말 것**.

### 0.1 착수 전 준비물 체크 (이 문서와 별개로 물리적으로 있어야 하는 것들)

이 저장소(Ver11)는 독립 git clone이라 **`../grade/`·`../Ver8/`·`../CLAUDE.md` 같은 형제
경로는 이 박스에 없을 수 있다** — 있으면 있는 대로 쓰고, 없으면 아래 폴백을 따를 것
(§4.2에도 반영돼 있음). 작업 시작 전에 다음을 실제로 `ls`로 확인:

1. **Ver8 챔피언 어댑터**(`SNUAI_WS8_ADAPTER`용 — warm-start 소스): 이전 Ver11 A100
   본런과 같은 경로에 이미 있으면 재사용, 없으면 dev box에서 scp 필요
   (`DPO-checkpoint-600/adapter_model.safetensors` + `adapter_config.json`).
2. **`../grade/grade.py` + `../grade/submission.csv`(답안키)**: 있으면 §4.2b를 그대로
   로컬에서 수행. **없으면** §4.2b는 "test 819건 `submission.csv`를 만드는 것까지만"
   하고, 그 파일을 dev box(사용자)에게 넘겨 채점을 요청할 것 — grade.py 로직을
   이 박스에서 재구현하려 하지 말 것(오늘 BALANCED8 중복구현 사고와 같은 패턴이 됨).
3. **data/train.csv, data/test.csv**: 이미 있을 것(기존 트랙에서 사용 중이던 데이터).

없는 게 있으면 임기응변으로 대체하지 말고 dev box에 먼저 보고할 것.

## 1. 배경 — 현재 코드 구조 (전부 실제 코드 읽고 확인함)

- `src/snuai11/fitprune.py` — 순수 텐서 로직(모델 의존성 0, CPU 테스트 가능). 현재
  `PruneConfig`(keep_ratio, diversity_frac, text_pool, event_pool, objectness_weight,
  mmr_lambda, enabled), `per_event_scores`, `cross_target_scores`, `objectness_scores`,
  `combined_scores`(cos+objectness 블렌드), `select_mmr`/`select_diverse`,
  `keep_indices_for_image`(단일 이미지 진입점).
- `src/snuai11/vlm.py`의 `Engine.keep_mask(self, prep: Prepared, caption: str, cfg:
  PruneConfig)` — **이미 `prep.per_image_embeds`(길이 4 리스트, 이 샘플의 4장 전부)에
  접근 가능**하다. 현재는 이걸 그냥 for-loop으로 돌며 각 이미지를 **독립적으로**
  `keep_indices_for_image(vis, events, cfg)` 호출한다(vlm.py:280-294).
- **중요한 사실 확인**: 학습(`train_sft.py:324-326`)과 추론(`infer.py:83-85`) 둘 다
  `prep = engine.prepare(...)` 한 번으로 그 샘플의 4장을 전부 처리한 뒤
  `engine.keep_mask(prep, sample.caption, prune_cfg)`를 호출한다 — 즉 **`keep_mask`는
  이미 4장 전부에 동시 접근 가능한 상태로 호출된다.** 2026-07-16 문서
  (`docs/fitprune_fix_2026-07-16.md` §6 항목6)가 우려했던 "4장 동시 인터페이스로 바꾸는
  공사"가 **호출부(train_sft.py/infer.py)에서는 전혀 필요 없다** — `fitprune.py`의
  스코어링 함수들과 `vlm.py`의 `keep_mask` **내부**만 고치면 된다. 호출부(두 파일)는
  건드릴 필요 없음(CLI 플래그 추가만 제외 — §2.6).

## 2. 기술 설계

### 2.1 새 함수 — `fitprune.py`에 `motion_scores` 추가

```python
@torch.no_grad()
def motion_scores(per_image_embeds: list[torch.Tensor], img_i: int) -> torch.Tensor | None:
    """이 이미지(img_i)의 각 토큰이 같은 격자 위치의 다른 3장과 얼마나 다른가.
    [N] 반환, 정적 배경은 낮고 움직이는/변하는 전경은 높다.

    None 반환 조건: 4장의 토큰 수(N)가 서로 다르면(다른 원본 해상도 — 실측
    ~22%에서 발생, measure_motion_signal.py로 확인) 위치 정렬이 무의미하므로
    계산하지 않고 None을 반환 — 호출자(combined_scores)는 이 경우 motion 항을
    생략(objectness/cos만 사용)하고 정상 진행해야 한다. 절대 크래시하면 안 됨.
    """
    n = per_image_embeds[img_i].shape[0]
    if any(t.shape[0] != n for t in per_image_embeds):
        return None
    v_i = per_image_embeds[img_i].float()
    others = [per_image_embeds[j].float() for j in range(len(per_image_embeds)) if j != img_i]
    diffs = torch.stack([(v_i - o).norm(dim=-1) for o in others], dim=0)  # [3, N]
    return diffs.mean(dim=0)  # [N]
```

`measure_motion_signal.py`(오늘 작성, 저장소에 이미 있음)의 로직을 그대로 옮긴 것 —
새로 설계하지 말고 이 파일을 그대로 참고할 것.

### 2.2 `PruneConfig`에 필드 추가

```python
@dataclass(frozen=True)
class PruneConfig:
    keep_ratio: float = 0.5
    diversity_frac: float = 0.2
    text_pool: str = "max"
    event_pool: str = "max"
    objectness_weight: float = 0.3
    mmr_lambda: float = 0.5
    motion_weight: float = 0.0   # 신규. 0=기존 동작과 완전히 동일(ablation 기본값)
    enabled: bool = True
```

**`motion_weight` 기본값은 반드시 0.0** — 07-16 objectness_weight/mmr_lambda 도입 때와
같은 원칙(모든 신규 항은 0이면 이전 파이프라인을 정확히 재현). 이래야 `--motion-weight 0`
ablation이 지금 챔피언(Ver8 warm-start 기반 SFT)의 기존 동작과 **비트 단위 동일**함을
보장하는 회귀 테스트를 짤 수 있다.

### 2.3 `combined_scores`·`keep_indices_for_image` 시그니처 변경

현재:
```python
def combined_scores(visual, event_embeds, cfg=PruneConfig()) -> torch.Tensor: ...
def keep_indices_for_image(visual, event_embeds, cfg=PruneConfig()) -> torch.Tensor: ...
```

**변경 — 두 함수 다 `per_image_embeds: list[torch.Tensor]`와 `img_i: int`를 추가로 받게
시그니처를 바꾼다** (siblings 접근용, `visual`은 `per_image_embeds[img_i]`와 동일하니
중복 인자로 남겨도 되고 내부에서 `per_image_embeds[img_i]`로 대체해도 됨 — 후자가
깔끔함. 택일해서 일관되게 적용):

```python
def combined_scores(per_image_embeds, img_i, event_embeds, cfg=PruneConfig()) -> torch.Tensor:
    visual = per_image_embeds[img_i]
    cos = _minmax(cross_target_scores(visual, event_embeds, cfg))
    parts = [((1.0 - cfg.objectness_weight), cos)]
    total_w = 1.0 - cfg.objectness_weight
    if cfg.objectness_weight > 0.0:
        parts.append((cfg.objectness_weight, _minmax(objectness_scores(visual))))
        total_w += cfg.objectness_weight  # 주의: 아래처럼 3항 정규화로 재작성 권장
    mot = motion_scores(per_image_embeds, img_i) if cfg.motion_weight > 0.0 else None
    if mot is None:
        # motion_weight>0인데 그리드 불일치로 계산 불가 -> 기존 cos(+objectness)만으로 폴백
        # (motion_weight=0인 경우와 동일 코드 경로를 타야 회귀 테스트가 이 폴백까지 커버함)
        return _blend_cos_objectness(cos, visual, cfg)  # 기존 로직 그대로, 헬퍼로 추출 권장
    mot_n = _minmax(mot)
    # 세 항 정규화: w_cos + w_obj + w_mot = 1이 되도록 cfg 필드 그대로 가중합
    # (objectness_weight·motion_weight는 "코사인에서 떼어오는 지분"으로 해석 —
    #  07-16 objectness 도입 때의 "min-max 후 가중합, w=0이면 코사인 랭킹 완전 보존"
    #  원칙을 3항으로 자연 확장. 정확한 수식은 구현 시 아래 테스트로 고정할 것:
    #  motion_weight=0 -> 기존 2항 결과와 완전 동일해야 함)
    w_obj, w_mot = cfg.objectness_weight, cfg.motion_weight
    w_cos = 1.0 - w_obj - w_mot
    assert w_cos >= 0, "objectness_weight + motion_weight > 1"
    obj_n = _minmax(objectness_scores(visual))
    return w_cos * cos + w_obj * obj_n + w_mot * mot_n
```

위는 **의사코드 — 정확한 블렌드 대수식은 구현자가 확정**하되 다음 불변량을 **테스트로
고정**할 것(§3):
1. `motion_weight=0.0`이면 결과가 (부동소수 오차 내에서) 지금 코드의 `combined_scores`
   결과와 **완전히 동일**해야 한다(회귀 방지 — 가장 중요한 테스트).
2. `objectness_weight + motion_weight`가 1을 넘으면 안 되거나(assert), 넘어도 안전하게
   클램프하거나 — 둘 중 하나를 명시적으로 선택하고 테스트로 고정.
3. 그리드 불일치로 `motion_scores`가 `None`을 반환하면 `motion_weight>0`이어도 크래시
   없이 폴백(1의 결과와 동일)해야 한다.

`keep_indices_for_image`도 같은 방식으로 `per_image_embeds, img_i` 인자로 바꿔서
내부에서 `combined_scores(per_image_embeds, img_i, event_embeds, cfg)`를 호출하도록
수정.

### 2.4 `vlm.py`의 `keep_mask` 재작성

현재(vlm.py:280-294):
```python
def keep_mask(self, prep, caption, cfg):
    ...
    events = self.event_embeds(caption)
    for img_i in range(len(prep.per_image_embeds)):
        vis = prep.per_image_embeds[img_i]
        kept_local = keep_indices_for_image(vis, events, cfg)
        ...
```

변경 — `vis = prep.per_image_embeds[img_i]` 줄을 지우고 호출을
`keep_indices_for_image(prep.per_image_embeds, img_i, events, cfg)`로 바꾸기만 하면 됨
(루프 구조 자체는 그대로 — **호출 순서상 4장이 이미 다 있으니 추가 forward나 캐싱 불필요**).

### 2.5 그리드 불일치 처리(§2.1에 이미 포함, 재강조)

**실측 22%의 샘플에서 4장의 토큰 수(N)가 서로 다르다**(다른 원본 프레임 해상도 —
`measure_motion_signal.py` 실행 결과 235/300). 이 22%에서 `motion_scores`가 `None`을
반환하고 `combined_scores`가 조용히 cos(+objectness)로 폴백하는 게 **의도된 동작**이다.
**리사이즈/보간으로 억지로 정렬시키려 하지 말 것** — 새로운 버그 표면을 여는 것보다
"모션 항이 그 샘플에서만 빠진다"는 단순한 동작이 안전하다. 이 폴백 비율(22% 근방)이
학습 로그에 찍히면 좋음(§2.6 참고 — 카운터 추가는 선택사항이지 필수 아님).

### 2.6 CLI 플래그 추가 — `train_sft.py`·`infer.py`

기존 패턴 그대로(train_sft.py:188-189, infer.py:131-132) 미러링:

```python
# train_sft.py, infer.py 둘 다 (parity 원칙 — 하나만 고치면 학습/추론 불일치)
ap.add_argument("--motion-weight", type=float, default=0.0,
                help="cross-frame residual-norm blend weight (0 = 기존 동작과 동일)")
```

그리고 `PruneConfig(...)` 생성부(train_sft.py:258-264, infer.py:61-67) 둘 다에
`motion_weight=args.motion_weight` 한 줄씩 추가.

## 3. 테스트 요구사항 (CPU 전용, GPU 불필요 — 구현 직후 바로 실행 가능)

`tests/test_fitprune.py`에 추가(기존 63개 옆에):

1. **회귀(가장 중요)**: `motion_weight=0.0`일 때 `combined_scores`/
   `keep_indices_for_image` 출력이 이번 변경 이전 코드와 **비트 단위 동일** — 기존
   테스트 스위트가 이미 이 값(디폴트 0.0으로 바뀌기 전엔 존재하지 않던 인자)으로
   호출되도록 시그니처 변경분을 반영해서 통과해야 함.
2. **motion 계산 정합성**: `measure_motion_signal.py`의 `motion_scores` 함수와 새
   `fitprune.motion_scores`가 같은 입력에 같은 출력을 내는지(합성 픽스처로) 확인.
3. **그리드 불일치 폴백**: 4장 중 하나만 토큰 수가 다른 합성 픽스처를 만들어
   `motion_scores`가 `None`을 반환하는지, `combined_scores(motion_weight=0.5)`가
   크래시 없이 `motion_weight=0` 결과와 동일하게 폴백하는지 확인.
4. **가중치 정규화**: `objectness_weight+motion_weight` 조합 몇 개(0.3+0.3, 0.5+0.5,
   0.7+0.5=초과 케이스)에서 §2.3에서 확정한 정책(assert 또는 클램프)이 실제로 그렇게
   동작하는지.
5. **`keep_mask` 통합 테스트**: 합성 4-이미지 `Prepared`로 `keep_mask`가 여전히 올바른
   shape의 bool mask를 내는지(기존 유사 테스트가 있으면 그 옆에 motion_weight>0 케이스
   추가).

**이 5개 전부 통과 + 기존 82개(pytest tests/ -q) 전부 통과를 확인한 뒤에만 §4로 진행.**
GPU 스모크(`python scripts/smoke_gpu.py --train`)도 이 단계에서 1회 돌려 parity(스톡
forward 대비 max|diff|=0)가 깨지지 않았는지 확인 — **motion_weight=0 기본값에서는 절대
깨지면 안 됨**(회귀 신호).

## 4. 롤아웃 절차 — 기존 ws8 게이트 재사용, 새로 만들지 말 것

`scripts/run_a100.sh`의 기존 `ws8` 경로(10스텝 전이 게이트 → 1500스텝)를 그대로 쓰되
`--motion-weight`를 추가 인자로 통과시키기만 한다:

```bash
export SNUAI_WS8_ADAPTER=<Ver8 챔피언 어댑터 경로 — 이전 본런과 동일>
bash scripts/run_a100.sh ws8 --motion-weight 0.3   # 값은 §4.1 참고
```

`run_a100.sh`가 이미 하는 일(수정하지 말 것): 10스텝 전이 게이트(ce<3.0) →
`runs/sft32b_v11_ws8/`에 1500스텝 저장(200스텝마다 체크포인트).

### 4.1 motion_weight 초기값 — 유도 근거

07-16 objectness_weight=0.3 도입 때와 같은 원리로, **튜닝값이 아니라 구조적 선택**으로
`motion_weight=0.3`, 나머지를 `objectness_weight=0.3, cos=0.4`로 재분배해서 시작할 것을
권장(3항이 대략 균등하되 캡션-코사인이 근소 우세 — 프루닝의 "캡션 조건부"라는 설계
정체성은 유지). **S3식 ablation(0/0.15/0.3/0.5) 여유가 있으면 사후에 조정** — 이 문서는
초기값 하나만 정하고 스윕은 시간이 남을 때만.

### 4.2 ⚠️ 게이트 순서 — 절대 건너뛰지 말 것

1. §3 CPU 테스트 + GPU 스모크 parity 전부 통과.
2. 10스텝 전이 게이트(`run_a100.sh`가 자동 수행) — ce<3.0. **이건 이번 신규 코드가 아니라
   기존 warm-start 자체의 게이트라 이번 변경과 직접 상관없지만, 혹시 motion_weight
   블렌드 버그가 loss를 이상하게 만들면 여기서 잡힐 수 있으니 정상 통과하는지 눈으로
   확인**(ce가 갑자기 발산하거나 NaN이면 즉시 중단 후 dev box 보고).
3. 1500스텝 본런 — **완료까지 자동 진행해도 되지만, 200스텝마다 저장되는 체크포인트의
   `train_log.jsonl` loss/ce 곡선이 06-17 이전 본런과 비슷한 궤적(플래토 없이 하강)인지
   중간중간(예: 500·1000·1500스텝) 눈으로 확인할 것.**
4. **본런 끝나면 in-sample 지표(마지막 체크포인트의 train subset 정확도 등)를 절대
   최종 판정에 쓰지 말 것.** 07-21 A100 세션이 정확히 이 함정에 빠졌었다
   (ckpt1400 in-sample 1등이 실제로는 2등이었음 — `docs/postmortem_2026-07-21_a100_and_plan.md`
   §3 인용). 대신:
   a. 체크포인트 200/400/.../1500 + adapter_final 전부에 대해 **in-sample 스윕으로
      후보 2~3개만 추리고**(줄 세우기 참고용, 최종 판정 아님),
   b. 그 후보들에 대해 **반드시 실제 test 819건에 대해 `run_pre.py`(또는 predict 진입점)로
      `submission.csv`를 생성** — 이건 무조건 이 박스에서 할 수 있다(외부 의존 없음).
      그 다음 채점은 **§0.1에서 확인한 대로 두 갈래**:
      - `../grade/grade.py`가 있으면: `python ../grade/grade.py <submission.csv>`로 바로
        채점(자체 답안키 v13 기준 est — `../CLAUDE.md`가 이미 경고했듯 est는 ≥1.5pp
        격차만 신뢰, ±1pp는 실LB 슬롯 필요).
      - **없으면**: 채점을 이 박스에서 재구현하지 말고, 후보 2~3개의 `submission.csv`
        파일 경로/이름을 명확히 보고하며 dev box에 "채점해달라"고 요청하고 **그 응답을
        받을 때까지 "성공" 여부 판단을 보류**할 것 — 이 문서에 있는 "0.9011"이라는
        숫자만 보고 스스로 비교·판정하지 말 것(그 숫자는 dev box의 답안키로 나온 것이라
        이 박스에 그 답안키가 없으면 재현할 수 없다).
   c. **실측이 안전 하한(motion_weight=0 baseline, 즉 지금 Ver11 최선인 adapter_final,
      key v13 기준 est 0.9011)보다 실제로 나은지 확인한 뒤에만** "성공"이라고 보고할 것.
   `scripts/auto_pipeline.py`가 이미 이 (a)→(b) 패턴을 구현해뒀으니(단, 내부에서
   `../grade/grade.py`를 직접 호출하는 부분이 있을 수 있으니 §0.1 확인 결과에 따라
   그 부분만 "submission.csv 생성까지"로 조정해서 재사용할 것 — 새로 짜지 말 것.

## 5. 금지사항

- **LB 제출 금지** — 결과가 아무리 좋아도 사용자 확인 없이 제출하지 말 것(슬롯 경합 —
  지금 챔피언 트랙에서 TTA24가 같은 자원을 쓰고 있을 수 있음, 조율 필요).
- **§2·§4에 명시된 것 이외의 설계 변경 금지** — 예: motion_scores를 코사인 유사도가 아닌
  다른 거리 함수로 바꾼다든지, 이벤트별로 분해한다든지 하는 확장은 이 라운드에서 하지
  말 것(오늘 검증한 건 "잔차 노름 기반 단순 모션 점수"뿐 — 다른 변형은 검증 안 됨).
- **게이트 순서(§4.2) 재배열·생략 금지.**
- **Ver8 저장소를 건드리지 말 것** — 이 작업은 전부 Ver11 저장소 안에서.
- **origin(Ver11 GitHub 원격)을 먼저 확인**하고 시작할 것 — 오늘 세션에서 다른 A100
  세션이 origin에 이미 올라간 기능(균형 TTA8)을 모르고 독자 재구현한 사고가 있었다
  (BALANCED8 이중구현). **작업 시작 전 `git fetch && git log origin/main` 로 이 계획서
  기준 커밋 이후 누가 이미 손댔는지 반드시 확인.**

## 6. 보고 요구사항 (dev box에 전달)

① §3 테스트 전부 통과 확인(몇 개 중 몇 개) ② 10스텝 게이트 ce 값 ③ 본런 loss/ce 곡선
요약(플래토 여부) ④ in-sample 스윕 상위 후보 목록 ⑤ **그 후보들의 실제 test+grade.py
결과**(est 수치, 안전하한 0.9011 대비 우열) ⑤ 최종 권고(승격 후보 있음/없음) ⑥ 소요
시간(과금 판단용) ⑦ 이상 징후 전부.

## 7. 자원·일정 현실 (2026-07-22 기준)

- **마감 2026-07-24 23:59** — 남은 시간 대략 2.5일.
- 이 작업은 **A100 학습 슬롯 1개**를 통째로 씀(10스텝 게이트 수 분 + 1500스텝 본런,
  과거 실측 기준 수 시간~반나절대 — 정확한 소요는 이번 런에서 다시 잼). **같은 시간대에
  다른 A100에서 TTA24(챔피언, ~6.6시간, 저위험)나 Ver13 GRPO가 돌고 있을 수 있으니
  dev box와 자원 배정 조율할 것.**
- Ver11은 이미 −2.7pp 종결 판정을 받은 트랙이라, 이 작업이 성공해도 **다른 격차 요인
  (Stackelberg LoRA 설정·KT 보조손실·자체 score24 헤드 등)까지 다 해결해야 챔피언을
  넘는다는 보장은 없다** — §4.2의 실측 게이트를 반드시 통과시켜서 "성공"의 기준을
  분명히 하고, 안 넘으면 그 자체로 유효한(발표용) 결과로 보고할 것.
