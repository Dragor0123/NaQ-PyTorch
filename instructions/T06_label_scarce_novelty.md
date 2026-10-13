# T06 — base 레이블 부족 설정에서 novelty 기반 앵커링 검증 (무인 실행 가능)

- 선행 문서: `CLAUDE.md` (**갱신판**: §1.1, §3 split 누수 금지, §6.1), `reports/T05_report.md`, `results/T05_summary.md`
- 무인 실행을 전제로 한다. 질문하지 말고 이 지시서의 규칙에 따라 판단하고, 판단 내용은 보고서에 기록한다.
- 시간 예산: **총 50분 이내**
  - 구현 약 10분
  - 진단(A) 약 2분
  - 실행(B) 약 33분
  - 집계·보고 약 5분
- 목적
  - T05 `naq_unl`의 이득(elec +2.8%p)은 "레이블 없음 = novel"이라는 벤치마크 구조에서 나왔다.
  - 이 구조가 성립하지 않는 설정, 즉 base 레이블 부족 설정(r = 0.5)에서 다음 두 가지를 확인한다.
    - (a) 레이블 없는 노드에 앵커를 두는 것만으로 이득이 얼마나 남는가
    - (b) base 레이블로 추정한 novelty로 앵커를 고르면 이득을 얼마나 회수하는가

## 0. 이번 지시서의 범위와 예외

- §6.1의 레이블 마스크를 새로 만든다 (r = 0.5, 0.2).
- 새 arm 2종을 추가한다: `unl_uni_r{r}`, `nov_r{r}`
- 기존 `naq_unl`(r = 1.0)은 †분석용 상한으로만 추가 실행한다. `CLAUDE.md` §3이 허용하는 범위다.
- scorer(`s_lr`)와 π̂는 이번에 쓰지 않는다.
- 범위 밖: TEG, scorer 재학습, 하이퍼파라미터 탐색, 새 novelty 정의 추가

## 1. 구현

### 1.1 `cape/labelmask.py`
- `make_mask(ds, r, seed=0)` → `cache/{ds}/labelmask_r{r}.npz` (`L_idx`, `y_L`)
  - base 클래스별로 round(r · n_c)개를 층화 추출하며, 클래스당 최소 1개를 남긴다.
  - 이미 있으면 재사용한다.
- 이 모듈만 split을 읽는다. 출력에는 `L_idx`, `y_L`만 담는다.

### 1.2 `cape/novelty.py`
- `novelty_scores(Xs, L_idx, y_L) -> np.ndarray[N]`
  - Xs = 행 L2 정규화한 SGC feature (Â² X). `cape.graph_ops`를 재사용한다.
  - μ_c = normalize(mean_{v ∈ L, y_v = c} Xs[v])
  - novelty(u) = 1 − max_c cos(Xs[u], μ_c)
- 함수 인자는 위 세 개로 제한한다 (`CLAUDE.md` §3). split을 import하지 않는다.
- 분석용으로 raw feature 기반의 같은 점수(`novelty_raw`)도 계산할 수 있게 한다. 이 값은 생성기에 쓰지 않는다.

### 1.3 `cape/generators.py`에 arm 추가

모든 arm의 query는 NaQ-best ranker(eletronics = `raw`, dblp = `ppr`)로 풀 안 top-10을 고른다. 모든 노드가 후보이고, way 간 중복을 허용한다. **arm 사이에 다른 것은 앵커 분포뿐이다.**

| arm | 앵커 후보 집합 | 샘플링 |
|---|---|---|
| `unl_uni_r{r}` | U = 전체 − L | U에서 5개 비복원 균등 |
| `nov_r{r}` | A_nov = U 중 novelty 상위 50% (값 고정, 튜닝 금지) | A_nov에서 5개 비복원 균등 |
| `naq_unl` (†, 기존) | valid ∪ test 클래스 노드 | 기존 구현 그대로 |

- `unl_uni`·`nov` 생성기는 `(pool, L_idx, y_L, novelty)`만 받는다.
- 에피소드 통계에 **`anchor_novel_frac`**를 추가한다. 앵커 중 valid·test 클래스 노드의 비율이며, 분석용으로 집계 시점에 계산한다.
  - 기존 arm의 JSON에도 이 값을 사후 계산해 집계표에 넣는다. 앵커 목록이 없으면 `anchor_group_frac`의 valid + test로 대신한다.

### 1.4 사전 점검 (실패하면 중단)
1. r = 0.5 마스크에서 클래스별 |L_c| / n_c가 0.5 ± 1/n_c이다.
2. r = 1.0일 때 U가 valid ∪ test 노드와 정확히 같다. 이 점검은 분석 코드에서만 한다.
3. `unl_uni`·`nov` 생성기 소스에서 split 관련 이름(`split`, `base`, `valid`, `test`)을 import하거나 참조하지 않는다. grep 결과를 보고서에 붙인다.

## 2. A — novelty 진단 (CPU)

- 실행: `python -m cape.diag_novelty` → `results/T06A_novelty.md`, `.json`
- 대상: eletronics, dblp × r ∈ {0.5, 0.2}

**계산할 것** (레이블은 분석용으로만 쓴다)
1. |L|, |U|, U의 구성 비율 (숨겨진 base / valid / test)
2. novelty AUROC: U 안에서 novel(valid ∪ test)과 숨겨진 base를 가르는 성능. `novelty`(SGC, 사용하는 점수)와 `novelty_raw`(분석용)를 모두 계산한다.
3. A_nov의 구성 비율과 `anchor_novel_frac` 기대값. U 균등 추출, NaQ(전체 균등), naq_unl(1.0)과 나란히 적는다.
4. A_nov의 novel 클래스 커버리지
   - A_nov에 노드가 1개 이상 포함된 novel 클래스의 비율
   - 클래스별 포함 비율의 지니 계수

## 3. B — 실행

**재사용** (새로 돌리지 않는다)
- NaQ-best seed 0, 1, 2: eletronics `naq_feat`, dblp `naq_diff` (T03/T04)
- `naq_unl` seed 0, 1 (T05)

**새 run** (우선순위 순)

| 우선순위 | arm | dataset | seed | run 수 |
|---|---|---|---|---|
| P1 (필수) | `unl_uni_r0.5`, `nov_r0.5` | eletronics, dblp | 0, 1, 2 | 12 |
| P2 | `naq_unl` (†) | eletronics, dblp | 2 | 2 |
| P3a | `unl_uni_r0.2`, `nov_r0.2` | eletronics | 0 | 2 |
| P3b | `unl_uni_r0.2`, `nov_r0.2` | dblp | 0 | 2 |

**공통 설정**: 4,000 에피소드, `eval_every` 250, 학습기와 평가는 T03과 같다.

**출력**: `results/expB/{ds}_{arm}_s{seed}.json` (arm 이름에 r을 포함한다)

**실행 방식**: `cape/run_T06.sh`. 프로세스 2개로 우선순위 순서대로 돌리고, 로그는 `logs/T06_*.log`에 남긴다.

**시간 규칙**
- 첫 run 2개가 끝나면 B의 남은 시간을 추정한다.
- P2, P3a, P3b를 순서대로 포함하되, B의 총 추정 벽시계 시간이 36분 이하인 데까지만 포함한다.
- 제외한 항목은 보고서에 적는다.

## 4. 집계 — `cape/summarize_t06.py`

`results/T06_summary.md`와 그림 `results/T06_dose_response.png`를 만든다.

1. **A 요약 표**: dataset × r. U 구성, AUROC(sgc / raw), A_nov의 novel 비율, 클래스 커버리지
2. **결과 표**: dataset × arm × seed
   - 1s/5s LR, query_purity, anchor_novel_frac, anchor_class_entropy, best 에피소드
3. **짝지은 차이** (같은 seed, 1s/5s)
   - `unl_uni_r0.5` − NaQ-best
   - `nov_r0.5` − NaQ-best
   - `nov_r0.5` − `unl_uni_r0.5`
   - `naq_unl` − NaQ-best (†)
   - seed별 값, 평균, 표준편차, 음수/양수 개수
4. **회수율**: R = (`nov_r0.5` − NaQ-best) / (`naq_unl` − NaQ-best). seed 평균으로 계산하며, dataset별로 적는다.
5. **그림** (dose-response)
   - x축: anchor_novel_frac
   - y축: 1s LR (seed 평균, 오차막대는 표준편차)
   - 점: NaQ-best, `unl_uni`, `nov`, `naq_unl` (+ r = 0.2가 있으면 포함)
   - 데이터셋별로 패널을 나눈다.
6. **사전 등록 대조표** (§5): 계산값만 적는다.

## 5. 사전 등록 (실행 전 이 파일을 그대로 커밋한다. 판정은 사람이 한다)

참고 계산 (CLAUDE.md §6 비율 기준, r = 0.5): U의 novel 비율은 eletronics ≈ .61, dblp ≈ .60이다. NaQ의 앵커 novel 비율은 ≈ .44 / .42, `naq_unl`은 1.0이다.

**게이트 G6** (r = 0.5, 1s LR, seed 3개)
- eletronics: mean(`nov` − NaQ-best) ≥ +1.0, 그리고 seed 3개 모두 양수
- dblp: mean(`nov` − NaQ-best) ≥ −0.5

| ID | 예측 | 기준 | 확신도 |
|---|---|---|---|
| R1 | SGC centroid novelty가 novel과 숨겨진 base를 어느 정도 가른다 | AUROC ≥ .65 (두 데이터셋, r = 0.5) | 0.60 |
| R2 | A_nov는 U보다 novel 비율이 높다 | eletronics A_nov novel 비율 ≥ .70 | 0.55 |
| R3 | U 균등 추출만으로도 일부 이득이 남는다 | eletronics `unl_uni_r0.5` − NaQ-best ∈ [+0.2, +1.6] | 0.60 |
| R4 | novelty 앵커링이 게이트를 통과한다 (eletronics) | G6 eletronics 조건 | 0.50 |
| R5 | novelty가 균등 추출보다 낫다 | eletronics mean(`nov` − `unl_uni`) > 0 | 0.65 |
| R6 | dblp에서 손해를 보지 않는다 | G6 dblp 조건 | 0.70 |
| R7 | 앵커 novel 비율에 대한 용량-반응 관계가 있다 | eletronics에서 NaQ < `unl_uni` < `nov` < `naq_unl` 순서 (seed 평균) | 0.55 |
| R8 | 레이블이 적을수록 균등 추출의 이득이 줄어든다 | eletronics s0: (`unl_uni_r0.2` − NaQ) < (`unl_uni_r0.5` − NaQ). P3a를 실행한 경우만 | 0.60 |

## 6. 중단 조건 (이 경우에만)

- §1.4 사전 점검이 실패한다.
- NaN 또는 OOM이 발생해 한 번 다시 돌려도 반복된다.
  - 해당 run만 건너뛰고 계속한다.
  - 2개 run 이상에서 반복되면 중단한다.
- 총 경과 시간이 60분을 넘는다. 진행 중인 run이 끝나면 멈추고, 완료된 것만 집계·보고한다.

## 7. 완료 기준과 보고

- [ ] **실행 전에** 이 지시서와 갱신된 `CLAUDE.md`를 수정 없이 커밋한다 (`T06: instruction, preregistration, CLAUDE.md update`).
- [ ] `cape/labelmask.py`, `cape/novelty.py`, `cape/diag_novelty.py`, `cape/run_T06.sh`, `cape/summarize_t06.py` 작성, generators 확장
- [ ] `results/T06A_novelty.md`, `results/T06_summary.md`, `results/T06_dose_response.png`
- [ ] `reports/T06_report.md` 작성, 커밋 (`T06: label-scarce setting and novelty anchoring`)

**보고 형식**: `CLAUDE.md` §12. 핵심 수치에 다음을 포함한다.
- A 요약 표
- 짝지은 차이 표
- 회수율
- G6 계산값
- 사전 등록 대조표
- 실행 범위(P2·P3 포함 여부)와 실제 소요 시간

해석은 짧게, 사실 위주로 적는다.
