# T05 — 메커니즘 진단(A)과 원인 분리 실험(B), 무인 실행용

- 선행 문서: `CLAUDE.md`, `reports/T04_report.md`, `results/T04_summary.md`
- **사용자는 약 40분간 자리를 비운다.** 질문하지 말고, 이 지시서의 규칙에 따라 판단하고 진행한다. 판단 내용은 보고서에 기록한다.
- 시간 예산: **총 42분 이내**
  - 구현 약 10분
  - A 실행 약 3분
  - 스모크 약 1분
  - B 실행 약 25분
  - 집계·보고 약 3분
- 목적: T03·T04의 "`cape_lite` < NaQ" 원인이 무엇인지 가른다.
  - 후보: 커버리지, 난이도, 하드 기각 자체
  - 다음 연구 방향을 정할 근거를 만든다.

## 0. 이번 지시서의 예외 (CLAUDE.md보다 우선)

1. **`naq_unl` arm은 "어떤 노드가 레이블되어 있는가"를 생성기에 사용한다.**
   - 이 벤치마크에서 레이블된 노드는 base 클래스 노드와 정확히 같다.
   - 따라서 `CLAUDE.md` §3의 "unlabeled = 비-base 가정 금지"에 대한 명시적 예외다.
   - 분석용 arm이며, 결과 해석 시 이 점을 표에 표시한다.
2. 새 arm 4개를 추가한다 (§2).
3. 무인 실행이므로 중단 대신 축소 규칙(§3)을 먼저 적용한다. 중단은 §6의 경우에만 한다.

## 1. A — 메커니즘 진단 (CPU, 학습 없음)

- 스크립트: `cape/diag_pihat.py`
- 실행: `python -m cape.diag_pihat` → `results/T05A_diag.md`, `results/T05A_diag.json`
- 입력: 기존 `cache/{ds}/pool.npz`, `scores.npz`, split
- 대상: 세 데이터셋 모두 (`Amazon_clothing`, `Amazon_eletronics`, `dblp`)
- 레이블은 분석용으로만 쓴다.

**노드 속성**
- `deg`: degree
- `boundary`: 이웃 중 레이블이 다른 노드의 비율. degree 0이면 NaN
- `class_size`: 노드가 속한 클래스의 크기
- `group`: base / valid / test
- `tail`: 노드가 속한 클래스가 해당 그룹 안에서 크기 하위 20%인가

**계산할 것**
1. Spearman ρ(π̂, 속성): `deg`, `boundary`, `class_size` 각각, 전체 노드 기준
2. π̂ 십분위표: 십분위별 평균 deg, 평균 boundary, test 그룹 비율, tail 비율, 실제 purity@10(lr)
3. 기각 집합 비교: π̂ 하위 30%(`cape_lite`가 버리는 노드)와 나머지 70%
   - 위 속성들의 평균
   - 그룹별 비율
4. 클래스 커버리지
   - 클래스별로 "기각된 노드 비율"을 구한다.
   - 그 분포의 지니 계수와 최댓값을 적는다.
   - 기각 비율이 50%를 넘는 클래스의 수를 적는다 (그룹별로).

## 2. B — 원인 분리 arm 4개

기존 `cape/generators.py`와 `cape/train_proto.py`를 확장한다. 모델, 손실 기본형, 평가, 에피소드 수(4,000), `eval_every`(250)는 T03과 같다. NaQ-best ranker는 eletronics = `raw`, dblp = `ppr`이다.

| arm | 바꾸는 요인 | 정의 |
|---|---|---|
| `naq_unl` | 커버리지만 | NaQ-best와 같다. 다만 앵커를 **레이블 없는 노드**(valid ∪ test 클래스 노드)에서만 비복원 균등으로 뽑는다. query는 NaQ-best ranker의 풀 안 top-10이며 모든 노드가 후보다. way 간 중복을 허용한다 |
| `lr_soft` | 기각 대신 소프트 가중 | 에피소드는 `lr_only`와 같다 (기각 없음). 손실은 query별 NLL에 w_q = s_lr(anchor, q) / (그 에피소드 50개 query의 s_lr 평균)을 곱해 평균한다 |
| `naq_hard` | 난이도 (어렵게) | 매 step NaQ-best 규칙으로 후보 에피소드 4개를 만든다. 그 step의 임베딩(학습용 forward 1회)으로 4개 각각의 손실을 계산하고, **손실이 가장 큰** 것으로만 backward한다 |
| `naq_easy` | 난이도 (쉽게) | `naq_hard`와 같고, **손실이 가장 작은** 것을 고른다 |

**구현 규칙**
- `naq_hard`·`naq_easy`의 후보 4개 × 4,000 step은 seed로 미리 생성한다.
- 후보 손실 계산은 같은 forward 결과로 하고, backward는 선택한 손실 하나에만 한다.
- 에피소드 통계(T03과 같은 항목)는 **실제로 학습에 쓰인 에피소드**로 계산한다.
- 추가로 `naq_hard`·`naq_easy`에는 다음을 기록한다.
  - 선택된 손실의 평균 / 후보 4개 손실의 평균 (전체 step 평균)
  - 선택된 에피소드의 query_purity와 후보 전체의 query_purity
- `lr_soft`에는 w_q의 평균, 표준편차, 최솟값, 최댓값을 기록한다.

**실행 목록** (총 12 run)

| 우선순위 | arm | dataset | seed |
|---|---|---|---|
| 1 | `naq_unl` | eletronics, dblp | 0, 1 |
| 2 | `lr_soft` | eletronics, dblp | 0, 1 |
| 3 | `naq_hard`, `naq_easy` | eletronics, dblp | 0 |

- 비교 대상인 NaQ-best와 `lr_only`의 seed 0, 1 결과는 T03/T04 JSON을 재사용한다.
- 출력: `results/expB/{ds}_{arm}_s{seed}.json`
- 실행 스크립트: `cape/run_T05.sh`. 프로세스 2개로 GPU 시간이 비슷하도록 배분하고, 우선순위 순서로 돌린다.
- 로그: `logs/T05_*.log`

## 3. 시간 규칙 (무인 실행)

1. A를 먼저 실행한다. 실패해도 B는 진행하고, A 실패 사유를 보고한다.
2. **스모크**: `naq_hard`(eletronics, seed 0)를 `--episodes 300 --eval_every 150`으로 한 번 돌려 step 시간을 잰다.
   - 후보 손실 계산 때문에 step 시간이 T03(27ms)보다 2배 넘게 길면, `naq_hard`·`naq_easy`의 후보 수를 4에서 3으로 줄인다.
3. **B 시작 시점의 축소 규칙**
   - 남은 예산 기준으로 B의 추정 벽시계 시간이 28분을 넘으면, 우선순위 3을 seed 0의 eletronics만 남긴다.
   - 그래도 넘으면 우선순위 2의 seed 1을 뺀다.
   - 축소 내용은 보고서에 적는다.
4. 실행 중 진행 확인은 5분 간격 정도로 충분하다. 확인할 때 불필요하게 많은 출력을 읽지 않는다.

## 4. 집계 — `cape/summarize_t05.py`

실행: `python -m cape.summarize_t05` → `results/T05_summary.md`

1. **A 요약**: ρ 표, 기각 집합 비교 표, 클래스 커버리지 지표 (3개 데이터셋)
2. **B 결과 표** (dataset × arm × seed)
   - 1s/5s LR, query_purity, anchor_unique_frac, anchor_class_entropy, anchor_group_frac, best 에피소드
3. **짝지은 차이** (같은 seed끼리, 1s/5s)
   - `naq_unl` − NaQ-best
   - `lr_soft` − `lr_only`
   - `naq_hard` − NaQ-best, `naq_easy` − NaQ-best, `naq_hard` − `naq_easy`
   - seed가 2개인 항목은 seed별 값과 평균을 적는다.
   - 1.0%p 미만에는 "잡음 범위 가능"이라고 표시한다 (dblp는 1.5%p 미만).
4. **사전 등록 대조표**: §5의 각 예측에 해당하는 계산값만 나란히 적는다. 적중 여부는 판정하지 않는다.

## 5. 사전 등록 (실행 전 이 파일을 그대로 커밋한다. 판정은 사람이 한다)

| ID | 예측 | 기준 | 확신도 |
|---|---|---|---|
| Q1 | π̂는 경계 노드에서 낮다 | 세 데이터셋 모두 ρ(π̂, boundary) ≤ −0.3 | 0.70 |
| Q2 | 기각 집합에 test 그룹 노드가 과대 표집된다 | 기각 30%의 test 비율 > 전체 test 비율 (3개 중 2개 이상) | 0.60 |
| Q3 | 기각이 일부 클래스를 크게 덜어낸다 | 기각 비율 > 50%인 클래스가 데이터셋마다 1개 이상 | 0.65 |
| Q4 | target 영역 커버리지는 이득이다 | eletronics에서 `naq_unl` − NaQ-best ≥ +0.5 (seed 평균) | 0.55 |
| Q5 | 소프트 가중은 거의 중립이다 | 두 데이터셋 모두 \|`lr_soft` − `lr_only`\| ≤ 0.7 | 0.60 |
| Q6 | 어려운 에피소드가 쉬운 것보다 낫다 | eletronics에서 `naq_hard` − `naq_easy` ≥ +1.0 | 0.55 |
| Q7 | 쉬운 에피소드는 NaQ보다 나쁘다 | 두 데이터셋 모두 `naq_easy` − NaQ-best < 0 | 0.65 |

## 6. 중단 조건 (이 경우에만 중단)

- NaN 또는 OOM이 발생했고, 해당 run을 한 번 다시 돌려도 반복된다. 이 경우 해당 run만 건너뛰고 나머지는 계속한다. 같은 문제가 2개 run 이상에서 생기면 중단한다.
- 레이블 규칙 위반이 불가피하다 (§0의 예외는 제외).
- 총 경과 시간이 55분을 넘는다. 이 경우 진행 중인 run이 끝나면 멈추고, 완료된 것만 집계·보고한다.

## 7. 완료 기준과 보고

- [ ] 이 파일을 실행 전에 그대로 커밋 (`T05: instruction and preregistration`)
- [ ] `cape/diag_pihat.py`, `cape/run_T05.sh`, `cape/summarize_t05.py` 작성, generators·train_proto 확장
- [ ] `results/T05A_diag.md`, `results/T05_summary.md`, JSON 최대 12개
- [ ] `reports/T05_report.md` 작성, 커밋 (`T05: mechanism diagnosis and disentangling arms`)

**보고 형식**: `CLAUDE.md` §12. 다만 핵심 수치에 다음을 포함한다.
- A: ρ 표, 기각 집합 비교 표
- B: 짝지은 차이 표
- 사전 등록 대조표
- 축소 규칙 적용 여부
- 실제 소요 시간

해석은 짧게, 사실 위주로 적는다.
