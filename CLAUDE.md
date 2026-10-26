# CLAUDE.md — CaPE 검증 실험 (NaQ-PyTorch 기반)

개정 2026-10-26 (T09): §1.1–1.2 현재 단계·확정 사실 갱신, §5 git(push·브랜치), §8 Exp B 대상에 Amazon_clothing 추가, §12 커밋 줄.

이 문서는 이 저장소에서 작업하는 Claude Code Agent의 상위 규약이다.
작업은 `instructions/T{nn}_{title}.md` 지시서 단위로 진행하며, 지시서와 이 문서가 충돌하면 **지시서가 우선**한다.
단, §3(레이블 사용 규칙)과 §10(기술 규칙)은 지시서가 명시적으로 예외를 선언하지 않는 한 항상 지킨다.

---

## 1. 연구 맥락 (요약)

- 과제: Graph Few-Shot Node Classification(FSNC)의 메타학습 에피소드 생성.
- 기존 방법 NaQ(ICML'24, 이 저장소)는 레이블 없이 에피소드를 만든다.
  - 앵커(=1-shot support)를 무작위로 뽑는다.
  - 앵커와 유사한 노드 top-Q를 같은 pseudo-class의 query로 쓴다.
  - 유사도는 NaQ-Feat이면 raw feature cosine, NaQ-Diff이면 PPR diffusion이다.
- 제안 아이디어 CaPE(Calibrated Pseudo-Episodes)의 핵심 주장은 "base 클래스 레이블은 학습기보다 **에피소드 생성 규칙**을 가르치는 데 쓰는 것이 낫다"이다.
  - 클래스와 무관한 쌍 단서(cue) 여러 개를 base 레이블로 학습한 로지스틱 회귀(scorer)로 결합한다.
  - 그 결과 "두 노드가 같은 클래스일 확률" s(u,v)를 얻는다.
  - s로 query를 고르고, 신뢰도 낮은 앵커를 기각한다.
- 이번 단계의 목적은 CaPE 전체 구현이 아니다. 아래 세 가설을 **80–90분 안에** 가장 싸게 확인하는 것이다.
  - **H1 (전이)**: base 레이블로 학습한 scorer가 **test 클래스 앵커**에서 가장 좋은 단일 단서보다 query 순도가 높다.
  - **H2 (병목)**: 에피소드 순도가 하위 FSNC 성능의 병목이다. 순도 100% Oracle 에피소드가 NaQ보다 확실히 좋아야 한다.
  - **H3 (회복)**: CaPE-lite 에피소드가 NaQ와 Oracle 사이 격차의 의미 있는 부분을 회복한다.
- 판정 기준(게이트)은 §9에 있다. 게이트 판정은 사람이 최종 결정한다. Agent는 수치와 판정 근거만 보고한다.

### 1.1 현재 단계 (T06부터)
- T01–T05에서 확인된 사실
  - H1·H3는 실패했다. 순도를 높이는 것은 병목이 아니었다.
  - 앵커를 novel(비-base) 영역에 두면 성능이 크게 올랐다 (T05 `naq_unl`, elec +2.8%p).
  - 단, `naq_unl`은 "레이블 없는 노드 = novel"이라는 벤치마크 구조를 이용한 분석용 상한이다.
- 현재 질문: **base 레이블만으로 novel 영역을 추정해서 그 이득을 정당하게 회수할 수 있는가.**
- 이를 위해 주 실험 설정을 **base 레이블 부족 설정**(§6.1)으로 바꾼다.

### 1.2 확정 사실 (T06–T08, r = 0.5, 1-shot LR, NaQ-best 대비; Agent가 전제로 삼아도 된다)
- 방법 후보는 `mix_alt_r0.5`로 고정한다: 에피소드의 70%는 앵커 ∝ p_novel(`pu_feat`, T07 E*, τ = 1)인 pseudo-episode, 30%는 L의 지도 에피소드.
  - T07·T08: Electronics +1.67 (2 seed), dblp +2.65 (2 seed).
- **novel 앵커의 독립 몫(T08 G8-a)**: Electronics `mix_alt − naq_mix` = +1.56 (s0 +1.46, s1 +1.66). dblp는 −0.35.
- **지도 혼합의 몫**: dblp `naq_mix − NaQ` = +2.99, Electronics +0.11. 지도 단독(`sup_only`)은 Electronics −5.73, dblp +0.72.
- **보정은 해롭다**: 보정된 p(`nov2c_t1`, E2* = `pu_both2`)는 Electronics에서 `nov2_t1`보다 −1.10(3/3). 보정 경로(E2*, `nov2c_*`)는 T09부터 쓰지 않는다.
- Electronics의 이득은 pseudo-episode 앵커 중 novel 노드 비율과 함께 오른다: .43 → 0, .67 → +0.52, .88 → +1.62, 1.00(`naq_unl`†) → +2.59. dblp는 상한 자체가 작다(`naq_unl`† +0.44).
- seed: Electronics·dblp의 mix 계열은 s0–1, `nov2_t1`·NaQ-best는 s0–2. Amazon_clothing은 Exp B 학습이 한 번도 없다(PU 추정기 캐시만 있음, T07 A).
- 현재 단계(T09): 새 방법 없이 논문 표에 필요한 수치를 확정한다 — seed 5개로 확장(Electronics·dblp), 세 번째 데이터셋(Amazon_clothing) 추가.
- 다른 브랜치: `token_proto`(TP01–TP02, 종결)는 이 트랙과 무관하다. 읽거나 수정하지 않는다.

## 2. 작업 방식

1. 사용자는 매 턴 `instructions/T{nn}_*.md` 하나를 지정해 작업을 지시한다.
2. Agent는 이 문서와 해당 지시서를 읽고, **지시서 범위 안의 일만** 수행한다. 다음 지시서의 작업을 미리 하지 않는다.
3. 지시서에 근거가 없는 결정이 필요하면 아래처럼 처리한다.
   - 결과에 영향이 작은 구현 세부(변수명, 배치 크기 등): 가장 단순한 선택을 하고 보고서에 적는다.
   - 프로토콜·split·지표 정의·게이트·하이퍼파라미터처럼 **결과를 바꿀 수 있는 결정**: 임의로 바꾸지 않는다. 중단하고 보고한다.
4. 아래 경우에는 작업을 중단하고 보고한다.
   - 지시서의 "중단 조건"에 해당할 때
   - 같은 오류를 3회 시도해도 해결하지 못할 때
   - 예상 실행 시간이 지시서의 시간 예산을 50% 넘게 초과할 것으로 추정될 때. 이 경우 실행하지 말고 추정치를 보고한다.
5. 종료 시(성공·부분 성공·실패 모두) 다음 순서로 마무리한다.
   1. `reports/T{nn}_report.md`를 쓴다.
   2. git commit을 한다.
   3. `git push origin cape-validation`을 한다(T09부터, §5).
   4. §12 형식의 메시지로 보고한다.

## 3. 레이블 사용 규칙 (누수 방지 — 항상 적용)

| 용도 | base 레이블 | valid 레이블 | test 레이블 |
|---|---|---|---|
| scorer(LR/GBDT) 학습 | O | X | X |
| CaPE-lite의 임계값·앵커 기각·query 선택 | X (s만 사용) | X | X |
| Sup arm 에피소드 | O | X | X |
| checkpoint 선택 | X | O (고정 valid 태스크로만) | X |
| Oracle arm 에피소드 | 전체 레이블 사용. **분석용 상한이며 방법이 아니다** | | |
| 지표 계산, 에피소드 통계 로그 | 분석용으로 모두 사용 가능 | | |

- 분석용 레이블 정보(지표, 에피소드 통계)를 학습·선택·임계값으로 되먹이지 않는다.
- valid 클래스 노드와 test 클래스 노드는 생성기 입장에서 "unlabeled"다.
- "unlabeled 노드 = 비-base"라는 가정을 생성기에 쓰지 않는다.

**split 누수 금지 (T06부터 강화)**
- 생성기, novelty 추정, scorer는 다음 두 가지 외의 레이블·split 정보를 받지 않는다.
  - 레이블된 노드 집합 L
  - L의 레이블
- 어떤 노드가 base / valid / test 클래스에 속하는지(split membership)는 생성기 코드에 전달하지 않는다.
  - 구현 시 생성기 함수의 인자를 `(features/pool, L_idx, y_L)`로 제한한다.
  - split 정보는 분석용 집계 코드에서만 읽는다.
- **표준 설정(base 전부 레이블, r = 1.0)에서는 "레이블 없음"이 곧 "novel"이다.**
  - 따라서 레이블 유무로 앵커 영역을 정하는 arm은 분석용(†)으로만 실행·보고한다.
  - 방법으로 주장하는 결과는 §6.1의 레이블 부족 설정에서 얻은 것만 쓴다.

## 4. 환경

- 서버: Ubuntu, RTX 4080 16GB (CUDA 12.4 driver), Ryzen 9 7900 (12C/24T), RAM 61GB
- conda env: **`taskns`** (`conda activate taskns`)
  - torch 2.4.1+cu121, scipy 1.13.1, scikit-learn 1.5.2
- 새 코드는 **torch, numpy, scipy, scikit-learn만** 쓴다.
  - torch_geometric을 쓰지 않는다. 설치되어 있더라도 쓰지 않는다.
- 패키지를 설치하거나 업그레이드하지 않는다. 필요한 패키지가 없으면 보고한다.
- 모든 명령은 저장소 루트(`~/wkspace/NaQ-PyTorch`)에서 `python -m cape.<module> ...` 형태로 실행한다.

## 5. 저장소 구조와 수정 규칙

```
NaQ-PyTorch/
├── CLAUDE.md                 # 이 문서 (사용자가 관리)
├── instructions/             # T{nn}_*.md 지시서 (사용자가 넣음, Agent는 수정 금지)
├── reports/                  # T{nn}_report.md (Agent가 작성)
├── cape/                     # 새 코드 전부 여기
│   ├── __init__.py
│   ├── data.py               # 그래프·feature·레이블 로드, split, 고정 태스크
│   ├── graph_ops.py          # Â, truncated PPR, SGC feature, 배치 top-k
│   ├── pool.py               # 후보 풀 + 단서 계산
│   ├── scorer.py             # LR/GBDT 학습 및 전체 풀 점수화
│   ├── expA.py               # 생성기 진단
│   ├── generators.py         # 에피소드 생성기 5종
│   ├── train_proto.py        # Exp B 학습·평가
│   └── summarize.py          # 집계·게이트 판정
├── cache/{dataset}/          # split.json, tasks.npz, pool.npz, scores.npz 등 (gitignore)
├── results/expA/, results/expB/   # 작은 JSON/MD 결과 (git 추적)
├── logs/                     # 실행 로그 (gitignore)
└── (NaQ 원본 파일들)
```

- 위 파일 구성은 기본안이다. 지시서가 다르게 지정하면 지시서를 따른다.
- **NaQ 원본 파일은 읽기 전용이다.**
  - 대상: `arg.py data.py meta.py model.py naq_train.py sup_protonet.py sup_train.py task_generator.py unsup_protonet.py utils.py data/ save/ images/ README.md`
  - NaQ 모듈을 import하지 않는다. NaQ 모듈은 torch_geometric을 import하고 N×N dense 행렬을 만든다. 프로토콜 참고용으로 읽기만 한다.
- git
  - 작업 브랜치는 `cape-validation`이다 (T01에서 생성). 원격 `origin`(Dragor0123/NaQ-PyTorch)에 같은 이름으로 있다.
  - 각 지시서는 시작 시 `T{nn}: instruction and preregistration` 커밋, 종료 시 `T{nn}: <요약>` 커밋을 만든다.
  - **T09부터 지시서 종료 커밋 뒤 `git push origin cape-validation`을 한다.** push가 실패하면 보고서 "문제·미해결"에 적고 계속한다. force push 금지.
  - `main`과 `token_proto` 브랜치에는 커밋하지 않는다. 다른 브랜치로 전환하지 않는다(지시서가 요구할 때만).
  - `data/`, `dataset/`, `cache/`, `logs/`는 커밋하지 않는다(.gitignore).

## 6. 데이터와 프로토콜 (사전 확인된 사실)

| dataset (파일명 그대로) | N | 무방향 edge | d | X nnz/행 | 전체 / base / valid / test 클래스 | 최소 클래스 크기 | Σp² | 5-way / 20-way way 오염률 | edge homophily | 평균 degree |
|---|---|---|---|---|---|---|---|---|---|---|
| `Amazon_clothing` | 24,919 | 91,680 | 9,034 | 31.74 | 77 / 40 / 17 / 20 | 101 | 0.0202 | 0.078 / 0.321 | 0.676 | 7.36 |
| `Amazon_eletronics` (철자 그대로) | 42,318 | 43,556 | 8,669 | 40.26 | 167 / 90 / 37 / 40 | 101 | 0.0081 | 0.032 / 0.144 | 0.373 | 2.06 |
| `dblp` | 40,672 | 288,270 | 7,202 | 46.57 | 137 / 80 / 27 / 30 | 101 | 0.0100 | 0.039 / 0.174 | 0.258 | 14.18 |

- X nnz/행은 전체 노드 기준이다 (T01 측정값).
- self-loop 0개, degree 0인 노드 0개이다. train.mat과 test.mat의 Index는 겹치지 않는다.
- edge homophily는 NaQ Table 7(0.62 / 0.38 / 0.29)과 조금 다르다. 정의 차이로 보이며 문제로 취급하지 않는다.

- 파일
  - `data/{ds}_network`: 탭으로 구분된 edge 목록이다. **양방향이 모두 들어 있다** (행 수 = 무방향 edge × 2).
  - `data/{ds}_train.mat`, `data/{ds}_test.mat`: 키는 `Index`, `Label`, `Attributes`(scipy sparse)이다.
- 모든 노드에 레이블이 있다. 레이블 값은 0..C−1로 연속이다.
- test 클래스 = `test.mat`의 레이블 집합이다. train 클래스 = `train.mat`의 레이블 집합이다.
- **valid 클래스는 NaQ가 저장해 둔 valid 태스크에서 추출한다.**
  - 대상 파일: `save/downstream_tasks/{ds}/*.pkl` 전체
  - 각 파일 valid pool의 실제 레이블을 모두 모아 합집합을 만든다.
  - 파일 하나(50개 태스크)만 쓰면 클래스가 많은 `Amazon_eletronics`에서 일부 valid 클래스가 빠질 수 있어 합집합을 쓴다.
  - 이렇게 하면 NaQ 논문과 split이 정확히 같아진다.
  - base 클래스 = train 클래스 − valid 클래스이다.
  - 합집합 크기가 17/37/27과 다르면 중단하고, 파일별 클래스 수를 보고한다.
- **평가 태스크도 NaQ의 저장본을 그대로 쓴다.**
  - `save/downstream_tasks/{ds}/5way_1shot.pkl`, `5way_5shot.pkl`
  - 내용은 `(valid_pool, test_pool)`이다. 각 원소는 `(id_spt: np.ndarray, y_spt: LongTensor, id_query: np.ndarray, y_query: LongTensor)`이다.
  - valid 태스크 50개, test 태스크 500개, query는 클래스당 8개다.
  - T01에서 `cache/{ds}/tasks.npz`로 변환해 두고, 이후에는 npz만 쓴다.
- 그래프 정규화: Â = D^−1/2 (A + I) D^−1/2이다. A는 대칭 이진 행렬이며 중복과 self-loop를 제거한 것이다. NaQ의 `normalize_adj(adj + I)`와 같다.

### 6.1 base 레이블 부족 설정 (T06부터 주 설정)

**레이블 마스크**
- base 클래스마다 노드의 비율 r만 레이블된 것으로 둔다.
  - 클래스별 층화 추출이며, 클래스당 최소 1개를 남긴다.
- 주 설정은 r = 0.5, 보조 설정은 r = 0.2이다.
- 마스크 seed는 0으로 고정한다. 학습 seed와 독립이며, 모든 arm이 같은 마스크를 쓴다.
- 저장 위치: `cache/{ds}/labelmask_r{r}.npz` (키: `L_idx`, `y_L`)

**집합 정의**
- L = 레이블된 base 노드
- U = 나머지 전부 (숨겨진 base, valid, test 노드)

**설정이 바꾸는 것과 바꾸지 않는 것**
- 평가(test 태스크), checkpoint 선택(valid 태스크), split은 바뀌지 않는다.
- NaQ arm은 레이블을 쓰지 않으므로 결과가 r과 무관하다. 기존 결과를 재사용한다.
- 레이블을 쓰는 모든 구성 요소(scorer, novelty 추정, sup 에피소드)는 L만 쓴다.

**참고 문헌**: NaQ Figure 1(a)의 Class%/Label% 실험, TEG·X-FNC의 레이블 부족 실험

## 7. 정의

**그래프 연산**
- `X_raw`: 원 BoW feature (scipy CSR, float32). GCN 입력으로 쓴다.
- `X_sgc` = Â² X_raw. dense float32를 허용한다 (약 1.2–1.5GB).
- PPR: Π = α Σ_{k=0..K} (1−α)^k Â^k, α = 0.1, K = 50 (truncated).
  - 대칭 행렬이다.
  - 배치 열 단위 power iteration으로 계산한다.
  - NaQ-Diff의 정확한 역행렬 해와의 차이는 (0.9)^51 ≈ 0.5% 이하이며, 보고서에 deviation으로 적는다.
- 모든 이웃 목록에서 자기 자신은 제외한다.

**후보 풀 (pool)**
- 노드 u마다 다음 세 목록의 합집합이다 (최대 P = 150, 부족분은 −1로 패딩).
  - top-50 by `cos_raw`
  - top-50 by `ppr`
  - top-50 by `cos_sgc`
- 각 ranker의 전역 top-Q(Q ≤ 50)는 풀 안의 순위와 같다.

**단서 (cue, 순서 고정, 6개)**

| idx | 이름 | 정의 |
|---|---|---|
| 0 | `cos_raw` | X_raw 행 간 cosine |
| 1 | `cos_sgc` | X_sgc 행 간 cosine |
| 2 | `log_ppr` | log(Π[u,v] + 1e−6) |
| 3 | `adj` | (u,v) ∈ E이면 1 |
| 4 | `mknn_raw` | v ∈ top10_raw(u) 이고 u ∈ top10_raw(v)이면 1 |
| 5 | `neg_euclid_raw` | −‖x_u − x_v‖₂ (raw, 정규화 전; 노름과 cosine으로 계산) |

**scorer**
- 학습 쌍: u ∈ V_base (최대 10,000개, seed 0으로 샘플), v ∈ pool(u) ∩ V_base. 레이블 y = 1[y_u = y_v].
- 모델
  - `StandardScaler` + `LogisticRegression(C=1.0, max_iter=2000)`
  - 참고용: `HistGradientBoostingClassifier(max_iter=200, random_state=0)`
- 출력: 모든 노드의 모든 풀 쌍에 대한 `s_lr[N,P]`, `s_gbdt[N,P]`.

**ranker**
- `raw`, `ppr`, `sgc`, `lr`, `gbdt` 다섯 가지다.
- 풀 안에서 각 점수로 정렬한 top-Q를 쓴다.
- `raw` = NaQ-Feat, `ppr` = NaQ-Diff에 대응한다.

**추정 순도**: π̂(u) = `lr` ranker top-10에 대한 s_lr의 평균.

**지표** (분석용)
- purity@Q(u; R) = (1/Q) Σ_{v ∈ topQ_R(u)} 1[y_v = y_u]
  - 앵커 그룹(base / valid / test)별 평균, Q ∈ {5, 10, 20}
- class-level similarity (NaQ 부록 A.1.2 정의)
  - c_k = 클래스 k 노드들의 raw feature 평균
  - 두 노드의 값은 cos(c_{y_u}, c_{y_v})이고, top-10에 대해 평균한다.
- AUC, ECE (10개 등간격 bin): test 그룹 앵커의 모든 풀 쌍에 대해 계산한다. base 그룹도 함께 계산한다.
- Spearman ρ(π̂, purity@10_lr): test 그룹
- 선택적 순도: test 앵커를 π̂ 내림차순으로 정렬해 상위 f ∈ {1.0, 0.7, 0.5}만 남겼을 때의 purity@10_lr
- way 오염률: 1 − (1 − Σp²)^{N−1}, N ∈ {5, 10, 20}

## 8. 실험 사양

### Exp A — 생성기 진단 (학습 없음)

- 대상: 데이터셋 3개 모두
- 출력: `results/expA/{ds}.json`, `results/expA/summary.md`
- 내용
  - ranker × 앵커 그룹 × Q별 purity
  - `raw`·`ppr`의 class-level similarity와 정확 순도 비교
  - LR 계수 (표준화 기준)
  - AUC, ECE, Spearman, 선택적 순도
  - tail 클래스 purity: test 클래스 중 크기 하위 20%
  - way 오염률
- **best single cue** R* = test 그룹 purity@10이 가장 높은 단일 단서 ranker (`raw`/`ppr`/`sgc` 중). Exp B의 Oracle에 쓴다.

### Exp B — 하위 성능 확인

- 대상 데이터셋: `Amazon_eletronics`, `dblp` (NaQ-Feat이 이기는 곳 / NaQ-Diff가 이기는 곳). **T09부터 `Amazon_clothing` 추가**(NaQ-best = `naq_feat`, `generators.NAQ_BEST_COL` 그대로).
- T04 이후 실행 설정: 학습 에피소드 4,000개, valid 평가 250 에피소드마다(`--episodes 4000 --eval_every 250`). T09도 같다.
- 에피소드 공통: 5-way, 1-shot support, way당 query 10개, 레이블은 `y_spt = arange(5)`, `y_qry = repeat(arange(5), 10)`

**arm 5개**

| arm | 생성 규칙 |
|---|---|
| `sup` | base 클래스 5개를 무작위로 뽑고, 클래스마다 노드 1+10개를 무작위로 뽑는다 |
| `naq_feat` | 전체 노드에서 앵커 5개를 비복원 균등 샘플하고, 각 앵커의 query는 `raw` top-10이다. way 간 query 중복을 허용한다 (NaQ 원형) |
| `naq_diff` | `naq_feat`과 같고 ranker만 `ppr`이다 |
| `cape_lite` | 아래 별도 설명 |
| `oracle` | 아래 별도 설명 (분석용) |

`cape_lite`
- 적격 앵커 집합 E = {u : π̂(u) ≥ π̂의 전체 노드 30번째 백분위수}. 레이블을 쓰지 않는다.
- 앵커를 E에서 하나씩 균등하게 뽑고, 다음을 모두 만족할 때만 채택한다.
  - (i) 이미 채택된 way들의 {앵커 ∪ query} 집합에 포함되지 않는다.
  - (ii) 자신의 `lr` top-10이 그 집합과 겹치지 않는다.
  - (iii) 채택된 각 앵커 b에 대해, b ∈ pool(a)라면 s_lr(a,b) < 0.5이다. 풀 밖이면 통과로 본다.
- 시도가 1,000회를 넘으면 오류로 기록한다.
- query = `lr` top-10

`oracle`
- 서로 다른 레이블을 가진 앵커 5개를 전체 노드에서 뽑는다.
- query = 풀 안에서 같은 레이블인 노드를 R* 순으로 정렬한 top-10이다.
- 10개가 안 되면 같은 레이블 노드를 무작위로 보충한다.

**에피소드 통계** (학습 전, arm마다 2,000개 생성, 분석용)
- query 순도
- 앵커 레이블 충돌률 (way 단위)
- way 간 query 중복률

**학습기** (NaQ ProtoNet 설정을 따른다)
- 인코더: 2-layer GCN
  - H1 = ReLU(Â(X W1) + b1), H2 = ReLU(Â(H1 W2) + b2)
  - 차원 d → 512 → 256, xavier_uniform 초기화, bias는 0, dropout 없음
  - X는 sparse tensor로 GPU에 올린다.
- 손실: prototype은 support 임베딩(K=1)이다. logits = −‖q − p‖², NLL을 쓴다.
- 최적화: Adam lr 1e-4, 한 step에 에피소드 1개, 총 16,000개 (NaQ 기본값)
  - T03의 시간 측정 결과에 따라 지시서가 줄일 수 있다.
- 검증
  - 400 에피소드마다 고정 valid 태스크 50개(5w1s)로 prototype 분류 정확도를 잰다.
  - 최고 checkpoint를 보관한다. 동률이면 앞선 것을 택한다.
  - NaQ는 160 에피소드마다 LR probing으로 검증한다. 이 차이는 시간 절약을 위한 deviation이다.
- 테스트
  - 최고 checkpoint의 임베딩을 쓴다.
  - 고정 test 태스크 500개(5w1s, 5w5s)마다 `LogisticRegression(max_iter=1000, C=1e4)`를 support로 학습하고 query 정확도를 잰다.
  - 평균과 95% CI(1.96·std/√500)를 보고한다.
  - prototype 분류 정확도도 보조 지표로 함께 기록한다.
- seed 0. split과 태스크는 seed와 무관하게 고정이다.
- 출력: `results/expB/{ds}_{arm}_s{seed}.json`
  - 필드: config, 에피소드 통계, valid 곡선, best 에피소드, test 정확도(LR / proto × 1-shot / 5-shot), 실행 시간

### 참고 수치 (NaQ 논문 Table 15/16, ProtoNet, 5-way, LR probing)

| | Sup 1s / 5s | NaQ-Feat 1s / 5s | NaQ-Diff 1s / 5s |
|---|---|---|---|
| Amazon_eletronics | 69.48 / 84.81 | 76.46 / 88.72 | 68.56 / 84.88 |
| dblp | 73.80 / 81.33 | 71.38 / 82.34 | 73.89 / 82.24 |

- 우리 재현값이 이 값에서 ±5%p 안에 있고, Feat/Diff의 우열 방향이 같으면 정상으로 본다. 게이트는 아니다.
- 우리 `sup`은 1-shot으로 학습한다. NaQ의 Sup 5-shot과는 설정이 다르다.
- NaQ가 보고한 class-level similarity (top-10, 전체 노드)

| | raw | ppr |
|---|---|---|
| Amazon_clothing | 81.6% | 78.8% |
| Amazon_eletronics | 81.9% | 52.7% |
| dblp | 73.5% | 83.0% |

## 9. 게이트 (판정 근거 계산은 Agent, 최종 판정은 사람)

- **G1 (H1)**
  - 조건: test 그룹에서 purity@10(`lr`) − max(`raw`, `ppr`, `sgc`) ≥ +0.03 (3%p)인 데이터셋이 3개 중 2개 이상
  - 보조: Spearman ρ ≥ 0.3
- **G2 (H2)**
  - 조건: 두 데이터셋 평균으로 [ `oracle` − max(`naq_feat`, `naq_diff`) ] ≥ +2.0%p
  - 기준 지표: 5-way 1-shot LR test 정확도
- **G3 (H3)**
  - 두 조건을 모두 만족해야 한다.
    - 두 데이터셋 평균으로 (`cape_lite` − NaQ-best) ≥ (`oracle` − NaQ-best) / 3
    - 각 데이터셋에서 `cape_lite` ≥ NaQ-best − 0.5%p
- 주의
  - test 태스크 500개의 95% CI는 대략 ±1.1%p이다.
  - seed가 1개이므로 1.5%p 미만의 차이는 잡음일 수 있다고 보고서에 명시한다.

## 10. 기술 규칙

- **[N, N] 크기의 dense tensor/array를 만들지 않는다.**
  - 블록 크기는 [N, 2048] 이하로 한다.
  - top-k는 블록 단위로 계산하고 바로 버린다.
- GPU 메모리 목표
  - 단일 프로세스: 12GB 이하
  - 두 프로세스를 동시에 돌릴 때(T03): 프로세스당 7GB 이하
- 결정론
  - `random`, `numpy`, `torch` seed를 모두 고정한다.
  - cache 파일은 한 번만 만들고, 있으면 재사용한다 (`--force`일 때만 다시 만든다).
- 2분이 넘는 실행은 `nohup ... > logs/<name>.log 2>&1 &`로 돌리고 진행 상황을 확인한다.
- 각 스크립트는 단계별 소요 시간을 로그에 남긴다.
- 코드 스타일
  - 단순한 함수와 argparse만 쓴다. 새 프레임워크, 노트북, 불필요한 추상화를 쓰지 않는다.
  - 같은 결과를 내는 방법이 여럿이면 더 단순한 쪽을 택한다.
- 수치 결과는 JSON으로 저장하고, 사람이 읽을 요약은 Markdown 표로 만든다.

## 11. 범위 밖 (지시서가 요구하지 않는 한 하지 않는다)

- s̃ (레이블 영역의 정확한 affinity 주입)
- K>1 support
- task-negative
- RWR 난이도 조절
- TEG 백본
- 다중 seed (지시서가 요구할 때만)
- 하이퍼파라미터 탐색

## 12. 보고 형식

`reports/T{nn}_report.md`에 아래 내용을 쓰고, 같은 내용을 메시지로도 보낸다. 메시지는 40줄 이내로 한다.

```
[T{nn} 보고] 상태: SUCCESS | PARTIAL | FAILED
소요 시간: 약 N분 (구현 a분 / 실행 b분)
수행 내용:
- ...
핵심 수치:
- (표 또는 key=value, 지시서가 요구한 항목 전부)
계획 대비 차이·임의 결정:
- (없으면 "없음")
문제·미해결:
- (없으면 "없음")
변경·생성 파일:
- ...
커밋: <short hash> (branch cape-validation), push: 성공 | 실패(사유)
```
