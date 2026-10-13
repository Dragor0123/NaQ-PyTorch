# T07 — PU 기반 test-영역 추정기와 비례 앵커 샘플링 (무인 실행 가능)

- 선행 문서: `CLAUDE.md` (§1.1, §3, §6.1), `reports/T06_report.md`, `results/T06_summary.md`, `instructions/T07_preregistration.md`
- 무인 실행을 전제로 한다. 질문하지 말고 이 지시서의 규칙에 따라 판단하고, 판단 내용은 보고서에 기록한다.
- 시간 예산: **총 80분 이내** (구현 약 15분, A 약 5분, B 약 50분, 집계·보고 약 5분). 시간 규칙은 §4.
- 목적
  1. novelty 추정을 휴리스틱(centroid 거리 상위 50%)에서 **PU learning 기반 보정 확률 p(novel|u)** 로 바꾸고, 추정 품질(AUROC)이 얼마나 오르는지 본다. 특히 dblp.
  2. 하드 컷 대신 **p(novel|u)^τ에 비례한 앵커 샘플링**으로 바꿔 이득이 오르는지 본다.
  3. base 지도 에피소드와의 **혼합**, 세 번째 데이터셋(Amazon_clothing), r ∈ {1.0, 0.2} 확장을 확인한다.
  4. **메커니즘 증거**: 학습된 임베딩에서 test 클래스 영역의 군집 품질이 arm에 따라 달라지는지 본다.

## 0. 범위와 규칙

- 레이블 규칙은 `CLAUDE.md` §3을 따른다. 추정기와 생성기는 (feature/구조, `L_idx`, `y_L`)만 받는다. split 정보는 분석·집계 코드에서만 읽는다. §1.4 방식의 소스 검사를 다시 한다.
- `mix_alt` arm은 L의 레이블로 지도 에피소드를 만든다. 이는 §3 표의 "Sup arm"에 해당하며 허용된다.
- r = 1.0 run은 "표준 설정에서 추정기가 자명해진다"는 사실을 확인하는 용도다. 결과 표에 그 점을 표시한다.
- 범위 밖: TEG, VISION 생성기 재구현, task-negative 손실, 하이퍼파라미터 탐색(τ는 아래 명시한 값만)

## 1. A — 추정기 비교 (`cape/novelty2.py`, `cape/diag_novelty2.py`)

대상: `Amazon_clothing`, `Amazon_eletronics`, `dblp` × r ∈ {0.5, 0.2}. 마스크는 `cape/labelmask.py`로 만든다 (clothing은 새로 생성).

**입력 채널** (모두 L과 그래프만 사용)
- `Xs`: 행 정규화 SGC feature (Â²X), T06과 같다.
- `Xr`: 행 정규화 raw feature
- `known_struct(u)`: L에서 출발한 확산 질량의 비율
  - m_L = Σ_{k=1..K} (1−α)^k Â^k 1_L, m_V = Σ_{k=1..K} (1−α)^k Â^k 1_V, α = 0.1, K = 50
  - known_struct = m_L / (m_V + 1e−12). 1_L은 L의 지시 벡터, 1_V는 전체 1 벡터.
- `logdeg`: log(1 + degree)

**추정기 6종** (출력은 모두 novelty 점수, 높을수록 novel)

| 이름 | 정의 |
|---|---|
| `cent_sgc` | T06의 추정기 (1 − max_c cos(Xs[u], μ_c)) |
| `cent_raw` | 같은 식을 Xr로 |
| `msp` | L로 학습한 base 분류기 `LogisticRegression(max_iter=2000)`(입력 Xs)의 1 − max softmax |
| `pu_feat` | Elkan–Noto PU: g = LR(L→1, U→0, 입력 Xs). c = L에 대한 5-fold out-of-fold g의 평균. p_known = min(1, g/c). novelty = 1 − p_known |
| `pu_struct` | 같은 PU 절차, 입력 [known_struct, logdeg] (표준화) |
| `pu_both` | 같은 PU 절차, 입력 [Xs, known_struct, logdeg] (표준화) |

- PU의 g는 U 전체와 L 전체로 학습한다. 클래스 불균형은 `class_weight="balanced"`로 보정한다.
- 각 추정기의 출력을 `cache/{ds}/novelty2_r{r}.npz`에 추정기 이름을 키로 저장한다.

**계산할 것** (레이블은 분석용)
1. AUROC: U 안에서 novel(valid ∪ test) vs 숨겨진 base. 6종 × 3 데이터셋 × 2 r.
2. 보정: `pu_*`에 대해 U에서의 mean p(novel) vs 실제 novel 비율. 절대 오차를 적는다.
3. 추정기 선택 규칙 (사전 명시, 데이터셋별 선택 금지)
   - **E\*** = r = 0.5에서 세 데이터셋 AUROC 평균이 가장 높은 추정기 하나
   - 보고서에 E\*와 데이터셋별 최고 추정기를 모두 적는다.
4. 비례 샘플링의 기대 앵커 novel 비율: 가중치 w(u) = p(novel|u)^τ (τ ∈ {1, 2}, U 위에서), E[novel frac] = Σ_{u∈novel} w / Σ_{u∈U} w. 참고로 T06의 하드 컷 값과 나란히 적는다.

출력: `results/T07A_estimators.md`, `.json`

## 2. B — arm 정의

query는 모든 arm에서 NaQ-best ranker(clothing·eletronics = `raw`, dblp = `ppr`)의 풀 안 top-10, way 간 중복 허용. 학습기·평가는 T03과 같다 (4,000 에피소드, `eval_every` 250).

| arm | 앵커 분포 | 비고 |
|---|---|---|
| `nov2_t1_r{r}` | U에서 w(u) ∝ p_E\*(novel\|u)^1 로 비복원 가중 추출 | 제안 방법, τ = 1 |
| `nov2_t2_r{r}` | 같음, τ = 2 | 더 강한 집중 |
| `mix_alt_r{r}` | 에피소드의 70%는 `nov2_t1`, 30%는 L에서 만든 5-way 1-shot 지도 에피소드 (클래스 5개, 각 1 support + 10 query, 모두 L에서) | 혼합. 비율 0.3 고정, seed로 교대 순서 결정 |
| `naq_feat` / `naq_diff` | 기존 | clothing은 새로 실행, 나머지는 재사용 |

- p(novel|u)가 0인 노드는 앵커로 뽑히지 않는다. U의 가중치 합이 0이면 오류로 보고한다.
- `train_proto.py`에 `--save_emb` 옵션을 추가한다. best checkpoint의 전체 노드 임베딩을 `cache/emb/{ds}_{arm}_s{seed}.npy`(float16)로 저장한다. T07의 모든 run은 이 옵션을 켠다.

## 3. B — 실행 목록 (우선순위 순)

| 우선순위 | run | 수 |
|---|---|---|
| P1 | `nov2_t1_r0.5`, `nov2_t2_r0.5` × {eletronics, dblp} × seed {0,1,2} | 12 |
| P2 | `mix_alt_r0.5` × {eletronics, dblp} × seed {0,1} | 4 |
| P3 | 임베딩 확보용 재실행: eletronics `naq_feat` s0, dblp `naq_diff` s0, eletronics `naq_unl` s0 (†) | 3 |
| P4 | `nov2_t1_r0.2` × {eletronics, dblp} × seed {0,1} | 4 |
| P5 | clothing: `naq_feat` × seed {0,1}, `nov2_t1_r0.5` × seed {0,1} | 4 |
| P6 | `nov2_t1_r1.0` × {eletronics, dblp} × seed 0 | 2 |

- 출력: `results/expB/{ds}_{arm}_s{seed}.json`. P3의 재실행은 기존 파일을 덮어쓰지 말고 `_emb` 접미사로 저장한다.
- 실행 스크립트 `cape/run_T07.sh`: T06과 같은 대기열 방식, 작업자 2개.

## 4. 시간 규칙

1. A를 먼저 끝낸다. A가 실패하면 E\* = `cent_sgc`로 두고 B를 진행하며, 사유를 보고한다.
2. 첫 run 2개가 끝나면 run당 시간을 재고, 남은 예산에서 **B의 총 추정 벽시계 시간이 55분 이하**가 되는 우선순위까지만 포함한다. 제외한 항목은 보고서에 적는다.
3. 다른 사용자의 프로세스가 GPU를 쓰고 있으면 그 사실과 `nvidia-smi` 요약을 보고서에 적는다. 실행은 계속한다.
4. 총 경과 시간이 85분을 넘으면 진행 중인 run이 끝난 뒤 멈추고, 완료된 것만 집계한다.

## 5. 집계 — `cape/summarize_t07.py` → `results/T07_summary.md`

1. **A 표**: 추정기 × 데이터셋 × r의 AUROC, 보정 오차, E\*, 기대 novel 비율
2. **B 결과 표**: dataset × arm × seed — 1s/5s LR, query_purity, anchor_novel_frac, anchor_class_entropy, best 에피소드
3. **짝지은 차이** (같은 seed, 1s/5s, 평균·표준편차·부호 개수)
   - `nov2_t1_r0.5` − NaQ-best, `nov2_t2_r0.5` − NaQ-best, `nov2_t1` − `nov2_t2`
   - `nov2_t1_r0.5` − T06 `nov_r0.5` (하드 컷 대비)
   - `mix_alt_r0.5` − `nov2_t1_r0.5`, `mix_alt_r0.5` − NaQ-best
   - `nov2_t1_r0.2` − NaQ-best; `nov2_t1_r1.0` − NaQ-best 와 `naq_unl` − NaQ-best 나란히
   - clothing: `nov2_t1_r0.5` − `naq_feat`
4. **회수율**: (nov2_t1_r0.5 − NaQ) / (naq_unl − NaQ), 데이터셋별
5. **용량-반응 그림** 갱신: x = anchor_novel_frac, y = 1s LR (seed 평균). NaQ, T06 unl_uni, T06 nov, nov2_t1, nov2_t2, naq_unl. 데이터셋별 패널. `results/T07_dose_response.png`
6. **AUROC–이득 그림**: x = E\*의 AUROC(r=0.5), y = nov2_t1 − NaQ (1s). 데이터셋 3개 점. `results/T07_auroc_vs_gain.png`
7. **메커니즘 분석** (저장된 임베딩, 레이블은 분석용)
   - 대상 run: P1의 eletronics·dblp seed 0, P3 세 개, P2 seed 0
   - test 클래스 노드만 모아 k-means(k = test 클래스 수, n_init = 3, seed 0) → 실제 레이블과의 NMI
   - base 클래스 노드도 같은 방식 (k = base 클래스 수)
   - 표: arm × {NMI_test, NMI_base}, 그리고 NaQ 대비 차이
8. **게이트 G7 계산값** (판정은 사람)
   - G7-a: E\*의 dblp AUROC(r=0.5) ≥ 0.65
   - G7-b: `nov2_t1_r0.5` − NaQ-best ≥ +1.0 이고 seed 3개 모두 양수인 데이터셋 수 (eletronics, dblp, clothing 중)
9. **사전 등록 대조표**: `T07_preregistration.md`의 S1–S10 계산값만

## 6. 중단 조건 (이 경우에만)

- 소스 검사(생성기·추정기에 split 참조)가 실패한다.
- NaN·OOM이 한 번 재시도 후에도 반복된다. 해당 run만 건너뛰고, 2개 이상이면 중단한다.
- 총 경과 시간이 85분을 넘는다 (§4-4).

## 7. 완료 기준과 보고

- [ ] 실행 전에 이 지시서와 `instructions/T07_preregistration.md`를 수정 없이 커밋 (`T07: instruction and preregistration`)
- [ ] `cape/novelty2.py`, `cape/diag_novelty2.py`, `cape/run_T07.sh`, `cape/summarize_t07.py`, generators·train_proto 확장
- [ ] `results/T07A_estimators.{md,json}`, `results/T07_summary.md`, 그림 2개, 메커니즘 표
- [ ] `reports/T07_report.md` 작성, 커밋 (`T07: PU novelty estimator, proportional anchoring, mixing`)

보고 형식은 `CLAUDE.md` §12. 핵심 수치에 A 표, 짝지은 차이 표, 회수율, 메커니즘 표, G7 계산값, 사전 등록 대조표, 실행 범위와 실제 소요 시간을 포함한다. 해석은 짧게, 사실 위주로 적는다.
