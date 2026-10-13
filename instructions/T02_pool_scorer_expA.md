# T02 — 후보 풀·단서·scorer 구축과 Exp A (생성기 진단)

- 선행 문서: `CLAUDE.md` (특히 §3, §7, §8 Exp A, §9 G1, §10), `reports/T01_report.md`
- 시간 예산: **20분** (구현 약 12분, 실행 약 3분, 보고 약 5분)
- 이번 지시서의 범위
  - `cape/graph_ops.py`, `cape/pool.py`, `cape/scorer.py`, `cape/expA.py` 작성
  - 세 데이터셋(`Amazon_clothing`, `Amazon_eletronics`, `dblp`)에 대해 풀·점수·Exp A 실행
- 범위 밖: 에피소드 생성기, GCN 학습, Exp B (T03)
- T01에서 측정한 시간(블록 B=2048 기준, 데이터셋당 풀 계산 약 15초)으로 보아 실행은 몇 분 안에 끝나야 한다.

## 1. `cape/graph_ops.py`

- `adj_hat_torch(A, device)`: Â = D^−1/2 (A+I) D^−1/2를 torch sparse로 만든다. `cape.data.norm_adj`를 재사용한다.
- `ppr_columns(Ahat, idx, alpha=0.1, K=50) -> Tensor[N, len(idx)]`
  - Π[:, idx] = α Σ_{k=0..K} (1−α)^k Â^k E_idx
  - 배치 power iteration으로 계산한다. Â가 대칭이므로 Π도 대칭이고, 이 열 블록은 곧 행 블록이다.
- `sgc_features(Ahat, X) -> dense Tensor[N, d]` = Â² X, 이어서 행 L2 정규화
- `row_normalize_dense(X) -> dense Tensor[N, d]`: raw X를 dense로 바꾸고 행 L2 정규화한다.
- T01 보고대로 `X_sgc` 피크 메모리가 4–5.5GB이므로, 생성 직후 불필요한 중간 텐서를 해제한다.

## 2. `cape/pool.py` — 후보 풀과 단서

실행: `python -m cape.pool --dataset {ds}` → `cache/{ds}/pool.npz` (있으면 재사용, `--force`면 다시 만든다)

**블록 루프** (앵커 블록 b, 크기 B=2048, 한 번의 순회로 처리)
1. `S_raw` = Xn @ Xn[b]ᵀ, `S_sgc` = Xs @ Xs[b]ᵀ, `P` = `ppr_columns(b)`. 각각 [N, B]이다.
2. 세 행렬 모두 자기 자신 위치를 −inf로 두고 `topk(50)`을 한다. 결과는 앵커당 후보 150개다.
3. 앵커별로 합집합을 만들어 중복을 제거한다. 최대 P=150이며, 부족분은 −1로 패딩한다.
4. 아직 블록이 메모리에 있을 때, 합집합의 각 쌍 (u, v)에 대해 다음 값을 gather한다.
   - `cos_raw` ← S_raw
   - `cos_sgc` ← S_sgc
   - `log_ppr` ← log(P + 1e−6)
5. 블록 텐서를 해제한다.

**블록 루프 이후 단서**
- `adj`: (u, v) ∈ E 여부. 쌍을 u·N+v 정수로 해시한 뒤 `np.isin`으로 판정한다.
- `mknn_raw`
  - top10_raw(u) = 풀 안에서 `cos_raw` 상위 10개
  - v ∈ top10_raw(u) 이고 u ∈ top10_raw(v)이면 1 (해시 방식으로 판정)
- `neg_euclid_raw` = −sqrt(max(‖x_u‖² + ‖x_v‖² − 2‖x_u‖‖x_v‖·cos_raw, 0)). raw(정규화 전) 노름을 쓴다.

**저장** (`pool.npz`)
- `nbr`: int32 [N, 150], 패딩은 −1
- `cue`: float32 [N, 150, 6], 패딩 위치는 NaN, 열 순서는 `CLAUDE.md` §7과 같다
- `pool_size`: int32 [N]

**검사** (실패하면 중단)
1. 무작위 앵커 100개에 대해 brute force로 전체 행 cosine(raw, sgc)을 구한다. 그 top-10이 풀 안의 `cos_raw` / `cos_sgc` 상위 10개와 일치하는지 본다. 동률로 인한 차이 외에는 100% 일치해야 한다.
2. 무작위 앵커 50개에 대해 K=100으로 다시 계산한 PPR의 top-10과 비교한다. K=50 결과와의 평균 겹침이 0.95 이상이어야 한다.
3. 대칭성: 무작위 쌍 1,000개에서 Π[u,v]와 Π[v,u]의 상대 오차 중앙값이 1e−3 이하여야 한다. 풀 안에 양방향이 모두 있는 쌍만 쓴다.

**기록할 통계**: 풀 크기(평균, 중앙값, 최소), 세 ranker top-50 간 쌍별 겹침 비율, 단계별 시간

## 3. `cape/scorer.py` — 쌍 scorer

실행: `python -m cape.scorer --dataset {ds}` → `cache/{ds}/scores.npz`, `results/expA/{ds}_scorer.json`

**학습 데이터** (`CLAUDE.md` §3을 지킨다. base 레이블만 쓴다)
- 앵커 u: V_base에서 seed 0으로 min(10,000, |V_base|)개를 샘플한다.
- 후보 v: pool(u) ∩ V_base, 패딩 제외
- 레이블 y = 1[y_u = y_v]

**모델**
- `StandardScaler` + `LogisticRegression(C=1.0, max_iter=2000)`
- `HistGradientBoostingClassifier(max_iter=200, random_state=0)`. 표준화하지 않은 원 단서를 쓴다.

**출력**
- `s_lr`, `s_gbdt`: float32 [N, 150], 모든 노드의 모든 풀 쌍에 대한 점수. 패딩은 NaN.
- `pi_hat`: float32 [N]. `lr` top-10에 대한 `s_lr`의 평균.
- `{ds}_scorer.json`
  - 학습 쌍 수, 양성 비율
  - LR 표준화 계수 6개(단서 이름 포함)와 절편
  - 학습 AUC (LR, GBDT)
  - 학습·추론 시간

## 4. `cape/expA.py` — 진단 지표

실행
- `python -m cape.expA --dataset {ds}` → `results/expA/{ds}.json`
- `python -m cape.expA --summarize` → `results/expA/summary.md`

**공통**
- 앵커 그룹: `base`, `valid`, `test`, `all` (레이블은 분석용으로 쓴다)
- ranker
  - `raw` (cue 0), `sgc` (cue 1), `ppr` (cue 2), `lr` (`s_lr`), `gbdt` (`s_gbdt`)
  - 각 점수로 풀 안을 내림차순 정렬한 top-Q를 쓰며, 패딩은 제외한다.
- 지표 정의는 `CLAUDE.md` §7을 그대로 따른다.

**계산할 항목**
1. purity@Q: Q ∈ {5, 10, 20} × ranker 5종 × 그룹 4종
2. class-level similarity (NaQ 정의, centroid = 클래스별 raw feature 평균)
   - ranker `raw`, `ppr`, `sgc`, `lr`의 top-10 기준
   - 그룹 `all`, `test`
   - 같은 셀에 정확 순도(purity@10)를 나란히 둔다.
3. 판별력과 보정 (그룹 `base`, `test`, 앵커의 모든 풀 쌍 기준)
   - `s_lr`, `s_gbdt`의 AUC
   - ECE (10개 등간격 bin)
4. 보정 격차: 그룹 `base`, `valid`, `test` 각각에 대해 mean(π̂) − mean(purity@10_lr)
5. Spearman ρ(π̂, purity@10_lr): 그룹 `test`, `valid`
6. 선택적 순도: `test` 앵커를 π̂ 내림차순으로 정렬해 상위 f ∈ {1.0, 0.7, 0.5}만 남긴 purity@10_lr
   - 비교용으로 같은 f에서 무작위로 남긴 경우의 값도 함께 적는다. f=1.0과 거의 같아야 한다.
7. tail: test 클래스를 크기 순으로 정렬해 하위 20%(올림)에 속한 앵커들의 purity@10을 ranker별로 구한다.
8. best single cue R*: `test` 그룹 purity@10이 가장 높은 ranker (`raw`/`ppr`/`sgc` 중)
9. G1 판정 근거
   - Δ_lr = purity@10_lr − purity@10_{R*} (`test` 그룹), 통과 기준 Δ_lr ≥ 0.03
   - 참고: Δ_gbdt, 그리고 `valid` 그룹에서의 같은 Δ
   - Spearman 통과 기준 ρ ≥ 0.3

**`summary.md` 구성** (Markdown 표)
- (a) G1 표: dataset, R*, purity@10(R*), purity@10(lr), Δ_lr, Δ_gbdt, Δ_lr(valid), Spearman(test), G1 통과 여부, 그리고 "3개 중 몇 개 통과"
- (b) purity@10 표: dataset × 그룹 × ranker (Q=5, 20은 JSON에만 둔다)
- (c) class-level similarity와 정확 순도 비교, NaQ 보고값(`CLAUDE.md` §8 참고 수치) 열 포함
- (d) 보정 표: AUC, ECE (base / test), 보정 격차 (base / valid / test)
- (e) 선택적 순도 표 (f별, 무작위 대조 포함)
- (f) LR 표준화 계수 표 (dataset × 단서 6개)
- (g) tail purity 표

## 5. 완료 기준

- [ ] 모듈 4개 작성
- [ ] 세 데이터셋의 `pool.npz`, `scores.npz` 생성, §2 검사 1–3 통과
- [ ] `results/expA/{ds}.json` 3개, `{ds}_scorer.json` 3개, `summary.md` 생성
- [ ] `reports/T02_report.md` 작성, 커밋 (`T02: candidate pools, pair scorer, Exp A diagnostics`)

## 6. 중단 조건

- §2 검사 1–3 중 하나라도 실패한다.
- GPU OOM이 발생한다. 블록 크기를 1024로 한 번만 줄여 재시도하고, 그래도 실패하면 중단한다.
- LR 학습이 수렴하지 않는다 (`ConvergenceWarning`이 `max_iter=5000`에서도 반복될 때).
- 레이블 규칙(`CLAUDE.md` §3)을 지키면서 구현할 방법이 불분명하다.

## 7. 보고에 반드시 포함할 항목

- `summary.md`의 (a) G1 표 전체
- (c) class-level similarity vs 정확 순도 표 (NaQ 보고값 비교 포함)
- (d) 보정 표, (e) 선택적 순도 표, (f) LR 계수 표
- 풀 통계 (크기, ranker 간 겹침), §2 검사 결과
- 데이터셋별 단계 시간 (풀 / scorer / 지표)
- 계획 대비 차이와 임의로 내린 결정
- 결과를 보며 눈에 띈 이상치 (예: 특정 단서의 계수 부호가 직관과 반대, AUC는 높은데 ECE가 큰 경우 등). 해석은 짧게, 사실 위주로 적는다.
