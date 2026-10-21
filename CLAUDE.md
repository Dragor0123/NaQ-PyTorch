# CLAUDE.md — token_proto 검증 실험 (NaQ-PyTorch 기반)

이 문서는 브랜치 `token_proto`에서 작업하는 Claude Code Agent의 상위 규약이다.
작업은 `instructions/TP{nn}_{title}.md` 지시서 단위로 진행한다. 지시서와 이 문서가 충돌하면 **지시서가 우선**한다.
단, §3(레이블 사용 규칙)과 §8(기술 규칙)은 지시서가 명시적으로 예외를 선언하지 않는 한 항상 지킨다.

이 브랜치는 `cape-validation`(마지막 커밋 `8f92e39`)에서 갈라진다. CaPE 트랙(T01–T08)의 코드·결과·보고서는 그대로 남아 있으며, 이 트랙에서는 **읽기 전용**이다.

---

## 1. 연구 맥락

**과제**: Graph Few-Shot Node Classification(FSNC)의 비지도 메타학습. 기준 방법은 NaQ-Feat(ICML'24)이다.
- NaQ-Feat 에피소드: 전체 노드에서 앵커 N=5개를 균등 비복원 추출해 각각 1-shot support(=pseudo-class)로 삼는다. 각 앵커의 raw feature cosine top-10을 같은 pseudo-label의 query로 쓴다.
- NaQ 손실(ProtoNet): prototype = 앵커 임베딩 하나. logit = −‖h_q − h_a‖².

**제안 아이디어 token_proto (TP)**: CrossTransformers(Doersch et al., NeurIPS 2020)의 query-aligned prototype을 NaQ 에피소드에 옮긴다.
- 앵커 하나를 **토큰 집합**으로 펼친다. 토큰 = 앵커 자신 + 앵커와 유사한 노드 m개(직접 이웃일 필요 없음).
- query마다 토큰 위에 attention을 걸어 **그 query만을 위한 prototype**을 만든다. 같은 pseudo-class라도 query가 다르면 prototype이 다르다.
- 기대 효과: 앵커 하나로는 담기지 않는 클래스 내 변이를 prototype에 넣어, 학습된 인코더가 novel 클래스를 더 잘 가르게 한다.
- 이 아이디어는 NaQ가 한계로 남긴 "K>1 support 생성"(NaQ §6.2)을 query-조건부 soft support로 푸는 것으로 볼 수도 있다.

**바뀌는 것은 학습 손실 하나다.** 에피소드(앵커·query), 인코더, 최적화, 평가는 NaQ-Feat 기준선과 같다.

### 1.1 현재 단계 (TP02부터)
- TP01에서 확인된 사실 (`reports/TP01_report.md`, `results/TP01_summary.md`)
  - TP 학습기는 λ = 0에서 NaQ-Feat 기준선과 같다 (Gb, G0 통과).
  - seed 0 격자에서 `tp_l0` 대비 1s LR 차이는 −1.23 ~ +0.48%p였다. valid로 고른 `tp_a2_pprrow_l50_m8`은 3 seed에서 clothing −1.06, eletronics −0.70, dblp +0.21이었다.
  - attention은 거의 균등했다 (정규화 엔트로피 0.95–0.98, 같은 레이블 토큰 질량 ≈ 균등 기대값).
  - sgc 토큰 순도는 clothing·eletronics에서 NaQ query 순도보다 낮고, dblp에서만 높았다. 일관된 양의 차이는 dblp에서만 보였다.
- TP02의 질문
  - seed 0에서 세 데이터셋 모두 양수였던 `tp_a2_sgc_l25_m8`의 이득이 seed 3개에서도 남는가.
  - 토큰이 레이블상 순수하면(분석용 상한) 이득이 생기는가. 즉 토큰 순도가 병목인가.
  - query별 attention이 균등 평균보다 나은가. attention을 날카롭게 하면 달라지는가.

## 2. 작업 방식

1. 사용자는 매 턴 `instructions/TP{nn}_*.md` 하나를 지정해 작업을 지시한다.
2. Agent는 이 문서와 해당 지시서를 읽고, **지시서 범위 안의 일만** 수행한다. 다음 지시서의 작업을 미리 하지 않는다.
3. 지시서에 근거가 없는 결정이 필요하면 아래처럼 처리한다.
   - 결과에 영향이 작은 구현 세부(변수명, 블록 크기 등): 가장 단순한 선택을 하고 보고서에 적는다.
   - 프로토콜·지표 정의·게이트·하이퍼파라미터·선택 규칙처럼 **결과를 바꿀 수 있는 결정**: 임의로 바꾸지 않는다. 중단하고 보고한다.
4. 아래 경우에는 작업을 중단하고 보고한다.
   - 지시서의 "중단 조건"에 해당할 때
   - 같은 오류를 3회 시도해도 해결하지 못할 때
   - 예상 실행 시간이 지시서의 시간 예산을 50% 넘게 초과할 것으로 추정될 때. 이 경우 실행하지 말고 추정치를 보고한다.
5. **사전 등록**: 지시서의 사전 등록 표(TPP 번호)는 실행 전에 지시서와 함께 커밋한다. 예측은 틀려도 수정하지 않는다. Agent는 예측과 결과의 산술 대조만 적고, 적중·판정은 사람이 한다.
6. 보고서에는 사실만 쓴다. 해석·추측·권고를 쓰지 않는다.
7. 종료 시(성공·부분 성공·실패 모두) 다음 순서로 마무리한다.
   1. `reports/TP{nn}_report.md`를 쓴다.
   2. git commit을 한다.
   3. §10 형식의 메시지로 보고한다.

## 3. 레이블 사용 규칙 (누수 방지 — 항상 적용)

| 용도 | base 레이블 | valid 레이블 | test 레이블 |
|---|---|---|---|
| 에피소드 생성 (앵커·query) | X | X | X |
| 토큰 선정, attention, prototype, 손실 | X | X | X |
| checkpoint 선택, arm·설정 선택 | X | O (고정 valid 태스크로만) | X |
| 지표 계산, 진단 통계 로그 | 분석용으로 모두 사용 가능 | | |

- TP와 NaQ-Feat은 **레이블을 전혀 쓰지 않는 방법**이다. 학습 경로의 어떤 코드도 `y`나 split을 읽지 않는다.
- 분석용 레이블 정보(토큰 순도, attention 질량 등)를 학습·선택·하이퍼파라미터로 되먹이지 않는다.
- 설정 선택은 **valid 정확도로만** 한다. test 정확도로 고르지 않는다.
- 레이블 없는 노드 사용 경계: 토큰과 query 후보는 **전체 노드**에서 고른다. "base가 아님" 같은 split 정보로 후보를 좁히지 않는다.

## 4. 환경

- 서버: Ubuntu, RTX 4080 16GB (CUDA 12.4 driver), Ryzen 9 7900 (12C/24T), RAM 61GB
- conda env: **`taskns`** (`conda activate taskns`)
  - torch 2.4.1+cu121, scipy 1.13.1, scikit-learn 1.5.2
- 새 코드는 **torch, numpy, scipy, scikit-learn, joblib만** 쓴다. torch_geometric을 쓰지 않는다.
- 패키지를 설치하거나 업그레이드하지 않는다. 필요한 패키지가 없으면 보고한다.
- 모든 명령은 저장소 루트(`~/wkspace/NaQ-PyTorch`)에서 `python -m tokproto.<module> ...` 형태로 실행한다.

## 5. 저장소 구조와 수정 규칙

```
NaQ-PyTorch/
├── CLAUDE.md            # 이 문서 (사용자가 관리, Agent 수정 금지)
├── instructions/        # 지시서 (사용자가 넣음, Agent 수정 금지)
├── reports/             # TP{nn}_report.md (Agent가 작성)
├── tokproto/            # TP 트랙의 새 코드 전부 여기
├── cape/                # CaPE 트랙 코드 — 읽기 전용, import는 허용
├── cache/{dataset}/     # pool.npz, tasks.npz, split.json 등 (gitignore, 서버에 이미 있음)
├── results/tp/          # TP run JSON (git 추적)
├── results/TP{nn}_*     # TP 요약 MD/JSON (git 추적)
├── logs/                # 실행 로그 (gitignore)
└── (NaQ 원본 파일들)     # 읽기 전용
```

- **읽기 전용(수정 금지)**
  - NaQ 원본: `arg.py data.py meta.py model.py naq_train.py sup_protonet.py sup_train.py task_generator.py unsup_protonet.py utils.py data/ save/ images/ README.md`
    - NaQ 원본 모듈은 import하지 않는다 (torch_geometric을 import하고 N×N dense 행렬을 만든다).
  - `cape/` 전체, `results/expA/`, `results/expB/`, 기존 `results/T0*`, `reports/T0*`, `instructions/T0*`
    - `cape`의 함수는 import해서 쓴다. 고쳐야 할 것이 있으면 `tokproto/`에 복사본을 만든다.
    - `cape` 스크립트를 실행해 추적 파일이 바뀌면 `git checkout -- <파일>`로 되돌린다.
- git
  - 작업 브랜치는 `token_proto`다. TP01 시작 시 Agent가 `cape-validation`에서 만든다.
  - 커밋 메시지는 `TP{nn}: <요약>`이다.
  - **push하지 않는다.** push는 사람이 한다.

## 6. 데이터와 프로토콜 (CaPE 트랙에서 확인된 사실)

| dataset (파일명 그대로) | N | 무방향 edge | d | 전체 / base / valid / test 클래스 | 평균 degree |
|---|---|---|---|---|---|
| `Amazon_clothing` | 24,919 | 91,680 | 9,034 | 77 / 40 / 17 / 20 | 7.36 |
| `Amazon_eletronics` (철자 그대로) | 42,318 | 43,556 | 8,669 | 167 / 90 / 37 / 40 | 2.06 |
| `dblp` | 40,672 | 288,270 | 7,202 | 137 / 80 / 27 / 30 | 14.18 |

- 그래프·feature·split·평가 태스크 로더: `cape.data` (`load_graph`, `load_split`, `load_tasks`, `norm_adj`, `to_torch_sparse`)
  - split과 평가 태스크는 NaQ 저장본(`save/downstream_tasks/{ds}/5way_{1,5}shot.pkl`)과 같다.
  - valid 태스크 50개(5w1s), test 태스크 500개(5w1s, 5w5s), query는 클래스당 8개.
- Â = D^−1/2 (A + I) D^−1/2.
- 후보 풀: `cache/{ds}/pool.npz` (키 `nbr [N,150]`, `cue [N,150,6]`)
  - 노드마다 top-50 by `cos_raw` ∪ top-50 by `ppr` ∪ top-50 by `cos_sgc`. 자기 자신 제외, 부족분은 −1.
  - cue 열: 0 `cos_raw`, 1 `cos_sgc`, 2 `log_ppr`, 3 `adj`, 4 `mknn_raw`, 5 `neg_euclid_raw`.
  - 각 ranker의 전역 top-k(k ≤ 50)는 풀 안에서 그 cue로 정렬한 top-k와 같다 (`cape.generators.top_in_pool`).
  - `cos_sgc` = Â²X_raw 행 간 cosine. PPR = α Σ_{k=0..50} (1−α)^k Â^k, α = 0.1 (truncated, `cape.graph_ops.ppr_columns`).

### 6.1 NaQ-Feat 기준선 (= TP의 비교 대상)

- 이 트랙에서 "NaQ 원본"은 CaPE 트랙에서 검증한 **NaQ-Feat 재구현**(`cape.train_proto`, arm `naq_feat`)을 뜻한다.
  - NaQ 원본 코드(`unsup_protonet.py`)와의 차이: valid를 LR probing 대신 prototype 정확도로 250 ep마다 잰다, 학습 4,000 ep. 나머지(GCN d→512→256, Adam 1e-4, 에피소드 5-way·1-shot·query 10, LR probing test)는 같다.
  - 기존 결과 (seed 0, 4000 ep, 1s LR / 5s LR): eletronics 74.58 / 87.44, dblp 71.72 / 81.35. eletronics는 seed 1, 2도 있다 (`results/expB/*_naq_feat_s*.json`).
  - CUDA sparse mm의 비결정성으로 같은 seed도 run 간 약 0.15%p 차이가 난다 (T04 측정).
- **TP 학습기**는 `cape.train_proto`의 학습 루프를 그대로 따르고 손실만 바꾼다.
  - 같은 seed면 에피소드(앵커·query) 열과 모델 초기화가 `naq_feat`과 같아야 한다.
    - seed 고정 순서: `random`, `numpy`, `torch`, `torch.cuda` → 에피소드 일괄 생성(`np.random.default_rng(seed)`, `make_topk_gen(top_in_pool(nbr, cue[:,:,0], 10))`) → 그래프 로드 → GCN 생성.
    - 토큰 선정은 결정적이어야 하며 rng를 소비하지 않는다.
  - λ = 0이면 TP 손실은 NaQ 손실과 수치적으로 같다. 이것을 회귀 게이트로 쓴다 (`tp_l0`).
- 평가: best checkpoint(valid prototype 정확도 최대, 동률이면 앞선 것)의 임베딩 H에 대해, test 태스크마다 `LogisticRegression(max_iter=1000, C=1e4)`를 support로 학습하고 query 정확도를 잰다. 1-shot, 5-shot 모두. 평균과 95% CI(1.96·std/√500). prototype 정확도는 보조 지표. (`cape.train_proto`의 `proto_acc`, `lr_task`, `ci`를 그대로 쓴다.)

## 7. 정의

**표기**: H = GCN(X, Â) ∈ R^{N×256}. 에피소드의 앵커 a_1..a_5, query q (pseudo-label y_q ∈ {0..4}), Q_ep = 에피소드 query 50개 집합.

**토큰 ranker** (토큰 후보를 정렬하는 기준, 레이블 없음)
- `sgc`: cos_sgc (풀 cue 1). 앵커의 Â²X 코사인 top-50.
- `pprrow`: PPR **행 벡터** 간 cosine, cos(Π_u, Π_v). "u에서 본 전체 노드의 PPR 분포"와 "v에서 본 분포"가 비슷한 정도다. 직접 연결이 없는 노드도 뽑힐 수 있다.
  - Π는 대칭이므로 Π_u·Π_v = (Π²)_{uv}, ‖Π_u‖² = (Π²)_{uu}.
  - 노드마다 top-50(자기 자신 제외)을 블록 단위로 계산해 `cache/{ds}/pprrow_top50.npz`(키 `nbr [N,50]`, `score [N,50]`)에 저장한다. [N,N] dense를 만들지 않는다.
  - 참고: PPR **값** Π_{uv}(풀 cue 2, `ppr`)와는 다른 양이다.

**토큰 집합** T(a) (앵커 a, 크기 m)
- 후보: ranker top-50을 점수 내림차순으로 본다.
- 제외: 그 에피소드의 query 50개(Q_ep 전체)와 다른 앵커 4개. 앵커 자신은 원래 후보에 없다.
- 남은 후보 중 앞에서 m개. m개가 안 되면 있는 만큼 쓰고, 0개면 아래 prototype이 앵커 임베딩으로 떨어진다. 부족 횟수를 로그에 남긴다.
- 제외 이유: query가 자기 클래스 토큰에 들어 있으면 attention이 자기 자신에 몰려 거리가 0이 된다. 다른 클래스 토큰에 들어 있으면 그 클래스의 prototype이 query와 같아진다.

**query별 prototype** (클래스 c, query q)
- attention: α^c_{q,j} = softmax_{j ∈ T(a_c)} ( cos(h_q, h_j) / τ_att ). 마스크된 토큰은 제외한다.
- prototype: t^c_q = (1 − λ) · h_{a_c} + λ · Σ_j α^c_{q,j} h_j
- λ = 0이면 t^c_q = h_{a_c} (NaQ와 같다). 앵커 몫 (1 − λ)는 항상 보장된다.
- key/query/value 변환은 두지 않는다 (항등). 학습 파라미터는 GCN뿐이다.
- 그래디언트는 h_q, h_{a_c}, 토큰 h_j, attention 가중치 모두를 통해 흐른다 (stop-grad 없음).

**logit과 손실** (네거티브 구성 두 가지)
- `a2` (CTX식): 모든 클래스 c에 대해 ℓ_c(q) = −‖h_q − t^c_q‖². query를 다섯 클래스의 토큰 집합에 각각 걸어 클래스별 prototype을 만든다.
- `b` (positive만 attention): ℓ_c(q) = −‖h_q − t^c_q‖² (c = y_q), ℓ_c(q) = −‖h_q − h_{a_c}‖² (c ≠ y_q).
- 손실 = 50개 query의 cross-entropy 평균. λ = 0이면 두 구성 모두 NaQ 손실과 같다.

**arm 이름 규칙**: `tp_{neg}_{tok}_l{100λ}_m{m}`. 예: `tp_a2_sgc_l50_m8`. λ = 0은 `tp_l0` 하나로 쓴다.
- 고정값: τ_att = 0.1, 에피소드 4,000개, eval 250 ep마다.

**진단 지표** (분석용, 레이블 사용 가능)
- 토큰 순도: 토큰 중 앵커와 실제 레이블이 같은 비율. query 순도(NaQ query 중 앵커와 같은 레이블 비율)와 나란히 적는다.
- 토큰 부족률: 크기가 m보다 작은 토큰 집합의 비율.
- hop 거리: 토큰과 앵커 사이 최단 경로 길이의 분포 (1, 2, ≥3, 연결 안 됨).
- pprrow–ppr 겹침: 앵커별로 pprrow top-m과 ppr(값) top-m의 교집합 크기 / m.
- attention 진단 (학습 끝, best checkpoint): 정규화 엔트로피 H(α)/log|T|, 같은 레이블 토큰에 실린 attention 질량.

## 8. 기술 규칙

- **[N, N] 크기의 dense tensor/array를 만들지 않는다.** 블록 크기는 [N, 2048] 이하로 한다. top-k는 블록 단위로 계산하고 바로 버린다.
- GPU 메모리: 단일 프로세스 12GB 이하, 두 프로세스 동시 실행 시 프로세스당 7GB 이하.
- 결정론: `random`, `numpy`, `torch` seed를 모두 고정한다. cache 파일은 한 번만 만들고, 있으면 재사용한다 (`--force`일 때만 다시 만든다).
- 2분이 넘는 실행은 `nohup ... > logs/<name>.log 2>&1 &`로 돌리고 진행 상황을 확인한다.
- 각 스크립트는 단계별 소요 시간을 로그에 남긴다.
- 코드 스타일: 단순한 함수와 argparse만 쓴다. 새 프레임워크, 노트북, 불필요한 추상화를 쓰지 않는다. 같은 결과를 내는 방법이 여럿이면 더 단순한 쪽을 택한다.
- 수치 결과는 JSON으로 저장하고, 사람이 읽을 요약은 Markdown 표로 만든다.
- 표준편차는 표본 표준편차(ddof=1)다.

## 9. 범위 밖 (지시서가 요구하지 않는 한 하지 않는다)

- 학습 가능한 key/query/value head, query 쪽 토큰화(query도 토큰 집합으로 펼치기)
- NaQ-Diff 기준선, TEG 백본, 레이블 부족 설정(CaPE §6.1)
- 10-way·20-way 평가, 에피소드 수 변경
- 지시서에 없는 하이퍼파라미터 탐색

## 10. 보고 형식

`reports/TP{nn}_report.md`에 아래 내용을 쓰고, 같은 내용을 메시지로도 보낸다. 메시지는 40줄 이내로 한다.

```
[TP{nn} 보고] 상태: SUCCESS | PARTIAL | FAILED
소요 시간: 약 N분 (구현 a분 / 실행 b분)
수행 내용:
- ...
핵심 수치:
- (표 또는 key=value, 지시서가 요구한 항목 전부)
사전 등록 대조:
- (TPP 번호별 예측 기준과 실제 값의 산술 비교. 적중 판정은 쓰지 않는다)
계획 대비 차이·임의 결정:
- (없으면 "없음")
문제·미해결:
- (없으면 "없음")
변경·생성 파일:
- ...
커밋: <short hash> (branch token_proto)
```

## 11. 번호 장부

| 항목 | 규칙 | 다음 번호 |
|---|---|---|
| 지시서 | `instructions/TP{nn}_*.md` (TP01 완료, TP02 진행) | TP03 |
| 사전 등록 | TPP### (지시서 안의 표). TP01: TPP001–010, TP02: TPP011–016 | TPP017 |
| 결과 | `results/tp/{ds}_{arm}_s{seed}.json` | — |
| 캐시 | `cache/{ds}/pprrow_top50.npz` (TP01에서 생성) | — |
| arm 접미사 (TP02) | `_t{100τ_att}` (τ_att ≠ 0.1일 때, 예 `_t02`), `_mean` (균등 α), tok `sgcora` (레이블 필터 토큰, 분석용 †) | — |