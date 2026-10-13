# T03 — Exp B: 에피소드 생성기별 하위 성능 확인과 최종 집계

- 선행 문서: `CLAUDE.md` (§3, §8 Exp B, §9, §10), `reports/T02_report.md`, `results/expA/summary.md`
- 시간 예산: **40분** (구현 약 12분, 스모크 측정 약 3분, 본 실행 약 20분, 집계·보고 약 5분)
- 이번 지시서의 범위
  - `cape/generators.py`, `cape/train_proto.py`, `cape/run_expB.sh`, `cape/summarize.py` 작성
  - Exp B 실행
  - G1–G3 판정 근거를 하나로 집계 (원래 T04에서 하려던 일을 여기에 합친다)
- 범위 밖
  - scorer 재학습이나 보정 수정
  - 하이퍼파라미터 탐색
  - 다중 seed
  - TEG

## 0. CLAUDE.md 대비 이번 지시서의 변경 (지시서 우선)

1. **arm 구성**
   - `sup`을 뺀다. 게이트 판정에 쓰이지 않기 때문이다.
   - `lr_only`를 추가한다. T02에서 `lr` 순위는 최선 단일 단서와 거의 같았고(Δ ≤ 1.6%p), 앵커 선택(π̂)만 뚜렷한 효과를 보였다. 그래서 두 요소의 효과를 분리해 본다.
   - 최종 arm: `naq_feat`, `naq_diff`, `lr_only`, `cape_lite`, `oracle` (데이터셋당 5개, 총 10회)
2. **학습 에피소드 수**: 16,000 → **4,000**. 검증 간격은 400 → **250**. 시간 예산 때문이며, 모든 arm에 똑같이 적용한다.
3. **R\***: `results/expA/{ds}.json`의 best single cue를 쓴다. 예상값은 `Amazon_eletronics` = `raw`, `dblp` = `sgc`이다.

## 1. `cape/generators.py`

모든 생성기는 `gen(rng) -> (id_spt[5], id_qry[50])`를 반환한다. 레이블은 `y_spt = arange(5)`, `y_qry = repeat(arange(5), 10)`이다.
입력은 `pool.npz`(`nbr`, `cue`)와 `scores.npz`(`s_lr`, `pi_hat`)이다. **생성 과정에서 레이블을 쓰는 arm은 `oracle`뿐이다** (`CLAUDE.md` §3).

| arm | 앵커 | query (way당 10개) | 기타 |
|---|---|---|---|
| `naq_feat` | 전체 노드에서 5개 비복원 균등 | 풀 안 `cos_raw` top-10 | way 간 query 중복 허용 (NaQ 원형) |
| `naq_diff` | 같음 | 풀 안 `log_ppr` top-10 | 같음 |
| `lr_only` | 같음 | 풀 안 `s_lr` top-10 | 같음. 기각 규칙 없음 |
| `cape_lite` | `CLAUDE.md` §8 규칙 그대로 (π̂ 하위 30% 제외, 순차 채택 조건 i–iii) | `s_lr` top-10 | way 간 중복 금지 |
| `oracle` | 서로 다른 레이블 5개 (전체 노드에서) | 풀 안 같은 레이블 노드를 R* 순으로 top-10, 부족하면 같은 레이블 무작위로 보충 | 분석용 상한 |

- `cape_lite`가 1,000회 시도 안에 에피소드를 완성하지 못하면 오류로 기록하고 중단한다.

**에피소드 통계** (학습 에피소드 4,000개 전부에 대해 계산, 분석용)
- `query_purity`: query 중 앵커와 같은 레이블인 비율
- `way_collision`: 앵커 레이블이 같은 에피소드의 다른 앵커와 겹치는 way의 비율
- `query_overlap`: 둘 이상의 way에 등장하는 query 슬롯의 비율
- `anchor_unique_frac`: 서로 다른 앵커 수 / 전체 앵커 슬롯 수
- `anchor_class_entropy`: 앵커 레이블 분포의 정규화 엔트로피 (log C로 나눈다)
- `anchor_group_frac`: 앵커가 base / valid / test 클래스에 속하는 비율

## 2. `cape/train_proto.py`

실행: `python -m cape.train_proto --dataset {ds} --arm {arm} --episodes 4000 --eval_every 250 --seed 0`

- 학습 에피소드 4,000개를 학습 시작 전에 seed로 미리 생성한다. 에피소드 통계도 이 집합으로 계산한다.
- 모델·손실·최적화는 `CLAUDE.md` §8을 따른다.
  - GCN d→512→256, ReLU, xavier, dropout 없음
  - squared Euclidean ProtoNet, Adam lr 1e-4, step당 에피소드 1개
- 검증
  - 250 에피소드마다(첫 측정은 학습 전, 0 에피소드) 고정 valid 태스크 50개에서 prototype 분류 정확도를 잰다.
  - 최고 checkpoint를 CPU에 복사해 둔다. 동률이면 앞선 것을 택한다.
- 테스트
  - 최고 checkpoint로 임베딩을 한 번 계산한다.
  - test1(5w1s), test5(5w5s) 각 500개 태스크에 대해 `LogisticRegression(max_iter=1000, C=1e4)` 정확도와 prototype 정확도를 잰다.
  - LR probing은 `joblib`으로 병렬화해도 된다 (`n_jobs ≤ 8`).
- 출력: `results/expB/{ds}_{arm}_s0.json`
  - 필드: config, 에피소드 통계, valid 곡선, best 에피소드, test 정확도(LR/proto × 1s/5s, 평균과 95% CI), 단계별 시간

## 3. 스모크 측정 → 본 실행

1. **스모크**: `Amazon_eletronics`의 `cape_lite`를 `--episodes 400 --eval_every 200`으로 한 번 돌린다.
   - 측정할 것: step당 시간, 생성기 시간, test 평가(LR probing 1,000회) 시간
   - 이 결과로 본 실행 전체의 벽시계 시간을 추정한다.
2. **본 실행 조건**
   - 추정 벽시계 시간(두 프로세스 병렬)이 **25분 이하**면 4,000 에피소드로 진행한다.
   - 25분을 넘으면 3,000 에피소드로 낮춰 다시 추정한다.
   - 그래도 넘으면 실행하지 말고 추정치를 보고한다.
3. **`cape/run_expB.sh`**
   - 데이터셋별로 백그라운드 프로세스 하나씩, 총 두 개를 띄운다.
   - 각 프로세스는 arm 5개를 순서대로 돈다: `naq_feat`, `naq_diff`, `lr_only`, `cape_lite`, `oracle`
   - 로그는 `logs/expB_{ds}.log`에 남긴다. `CLAUDE.md` §10의 nohup 규칙을 따른다.
4. 진행 상황을 주기적으로 확인한다. 한 run이 추정치의 2배를 넘기면 원인을 확인하고 보고한다.

## 4. `cape/summarize.py` — 최종 집계

실행: `python -m cape.summarize` → `results/summary.md`, `results/expB/purity_vs_acc.png`

**(a) Exp B 표** (dataset × arm)
- query_purity, way_collision, query_overlap, anchor_unique_frac, anchor_class_entropy
- test 1s LR ± CI, test 5s LR ± CI, test 1s proto
- best 에피소드 / 전체 에피소드, 실행 시간

**(b) 게이트 표** (`CLAUDE.md` §9 정의)
- G1: `results/expA`의 값을 그대로 옮긴다 (Δ_lr, Spearman, 통과 여부).
- G2: 데이터셋별 및 평균의 [`oracle` − NaQ-best], NaQ-best = max(`naq_feat`, `naq_diff`), 1s LR 기준
- G3
  - (`cape_lite` − NaQ-best)를 (`oracle` − NaQ-best)/3과 비교한다 (평균).
  - 데이터셋별로 `cape_lite` ≥ NaQ-best − 0.5%p인지 확인한다.
  - 데이터셋별 판정과 평균 판정을 모두 적는다.
- 보조 비교 (게이트 아님)
  - `lr_only` − NaQ-best: 순위 학습만의 효과
  - `cape_lite` − `lr_only`: 앵커 기각·중복 금지의 효과
- 1.5%p 미만의 차이에는 "잡음 범위"라고 표시한다.

**(c) 재현 점검** (`CLAUDE.md` §8 참고 수치와 비교)
- `naq_feat`과 `naq_diff`의 1s/5s LR 값을 NaQ 논문값과 나란히 적는다.
- ±5%p를 넘으면 표시한다.

**(d) 그림**
- x축 query_purity, y축 1s LR 정확도
- 점 = arm, 데이터셋별 마커
- matplotlib 기본 스타일, 축 이름과 범례 포함

**(e) 과소학습 점검**
- best 에피소드가 마지막 20%(3,200 이후)에 있는 run의 수를 적는다.
- 과반이면 "학습 부족 가능성"으로 표시한다.

## 5. 완료 기준

- [ ] 모듈 3개와 셸 스크립트 1개 작성
- [ ] 스모크 측정과 실행 조건 판단을 기록
- [ ] 10회 실행 결과 JSON 10개
- [ ] `results/summary.md`, `results/expB/purity_vs_acc.png`
- [ ] `reports/T03_report.md` 작성, 커밋 (`T03: Exp B downstream check and gate summary`)

## 6. 중단 조건

- 스모크 기준으로 3,000 에피소드에서도 추정 벽시계 시간이 25분을 넘는다.
- `cape_lite`가 에피소드를 완성하지 못한다 (1,000회 시도 초과).
- 학습 중 loss가 NaN이 된다.
- GPU OOM이 발생한다. 이 경우 병렬을 끄고 순차 실행으로 한 번만 재시도한다.
- 레이블 규칙(`CLAUDE.md` §3)을 지키면서 구현할 방법이 불분명하다.

## 7. 보고에 반드시 포함할 항목

- 스모크 측정값과 실행 조건 판단 (에피소드 수)
- `results/summary.md`의 (a) Exp B 표, (b) 게이트 표, (c) 재현 점검, (e) 과소학습 점검
- 에피소드 통계에서 눈에 띄는 점
  - 예: `cape_lite`의 anchor_class_entropy나 anchor_group_frac이 다른 arm과 크게 다른가
- 계획 대비 차이와 임의로 내린 결정
- 해석은 짧게, 사실 위주로 적는다. 게이트 판정은 사람이 한다.
