# TP01 — token_proto 첫 검증: 구현, 회귀 게이트, seed 0 격자, 3 seed 확인

- 선행 문서: `CLAUDE.md` (전부, 특히 §6.1 기준선과 §7 정의)
- **무인 실행 전제.** 질문하지 말고 이 지시서의 규칙에 따라 판단하고, 판단 근거를 보고서에 적는다.
- 시간 예산: **120분** (구현 25 / A 진단 10 / B1 45 / B2 30 / 집계·보고 10). 시간 규칙은 §7.
- 목적
  1. TP 학습기를 구현하고, λ = 0에서 NaQ-Feat 기준선과 같은지 확인한다 (회귀 게이트).
  2. 학습 없이 토큰 집합의 성질(순도, 거리, pprrow의 국소성)을 잰다.
  3. seed 0에서 TP 설정 6개를 NaQ-Feat(`tp_l0`)과 비교한다.
  4. valid로 고른 설정 하나를 seed 3개로 다시 비교한다.

## 0. 범위
- 데이터셋: `Amazon_clothing`, `Amazon_eletronics`, `dblp`
- 평가: 5-way 1-shot, 5-way 5-shot (같은 학습 run에서 둘 다 잰다)
- 범위 밖: `CLAUDE.md` §9 전부, 그리고 이 지시서의 arm 표에 없는 설정

## 1. 시작 절차 (git)

1. 현재 상태를 확인한다. 하나라도 다르면 **아무것도 바꾸지 말고** 중단하고 출력을 그대로 보고한다.
   - `git branch --show-current` → `cape-validation`
   - `git rev-parse HEAD` → `8f92e396901239a07996da763a894af067afa7a3`
   - `git status --porcelain`
     - 추적 파일의 변경은 ` M CLAUDE.md` 하나뿐이어야 한다.
     - `?? instructions/TP01_token_proto_first.md`가 있어야 한다.
     - 그 밖의 untracked 파일은 허용한다. 건드리지 않고, 커밋에 넣지 않고, 목록을 보고서에 적는다.
2. `git checkout -b token_proto`
3. `git add CLAUDE.md instructions/TP01_token_proto_first.md`로 두 파일만 커밋한다: `TP01: instruction, preregistration, CLAUDE.md`
   - 이 커밋은 코드 작성·실행 **전에** 한다. 사전 등록(§9)을 고정하기 위해서다.
4. 이후 모든 작업은 `token_proto` 브랜치에서 한다. push하지 않는다.

## 2. 사전 확인 게이트 (학습 전)

**Ga — 입력 존재 확인**
- 세 데이터셋 모두: `data/{ds}_network`, `data/{ds}_train.mat`, `data/{ds}_test.mat`, `cache/{ds}/pool.npz`, `save/downstream_tasks/{ds}/5way_1shot.pkl`, `5way_5shot.pkl`
- `cache/{ds}/tasks.npz`, `cache/{ds}/split.json`이 없으면 `cape.data.load_tasks` / `load_split`가 만든다 (cache에만 쓰므로 허용).
- `cache/Amazon_clothing/pool.npz`가 없으면 `python -m cape.pool --dataset Amazon_clothing`으로 만들고, 바로 `git checkout -- results/expA/`로 추적 파일 변경을 되돌린다. 다른 데이터셋의 pool이 없으면 중단하고 보고한다.

**Gb — 손실 동치 단위 검사** (`tokproto/test_equiv.py`, CPU 가능)
1. 합성 입력: seed 0, N = 300, D = 16, H = ReLU(randn). 에피소드 5개(앵커 5, query 50, 토큰 m = 8 무작위, 일부 way는 토큰 0개).
   - L_naq = `cross_entropy(-cdist(Q, P)**2, y)` (`cape.train_proto`와 같은 식)
   - L_tp(λ = 0, `a2`), L_tp(λ = 0, `b`)
   - 조건: |L_tp − L_naq| ≤ 1e-5 (두 구성 모두, 에피소드 5개 모두)
2. 토큰을 전부 앵커 임베딩의 복사본으로 두고 λ = 1: L_tp = L_naq (≤ 1e-5)
3. 토큰 0개인 way: prototype이 앵커 임베딩과 같고 NaN이 없다.
4. 실제 입력: `Amazon_eletronics`, seed 0의 첫 에피소드, 초기화 직후 GCN의 H에서 L_tp(λ = 0, `a2`)와 L_naq 비교 (≤ 1e-5).
- 하나라도 실패하면 B를 실행하지 않는다. 원인을 보고하고 중단한다.

## 3. 구현 (`tokproto/`)

| 파일 | 내용 |
|---|---|
| `tokproto/__init__.py` | 빈 파일 |
| `tokproto/pprrow.py` | pprrow top-50 캐시 생성 (아래) |
| `tokproto/tokens.py` | 토큰 후보 목록 로드(`sgc`, `pprrow`), 에피소드별 토큰 집합 [E, 5, m] + 마스크 생성 (`CLAUDE.md` §7 규칙) |
| `tokproto/loss.py` | TP 손실 (`a2`, `b`), attention 진단 함수 |
| `tokproto/test_equiv.py` | Gb |
| `tokproto/diag_tokens.py` | A 진단 |
| `tokproto/train_tp.py` | 학습·평가 (아래) |
| `tokproto/run_TP01.sh` | 대기열 실행 (작업자 2개) |
| `tokproto/summarize_tp01.py` | 집계 |

**`pprrow.py`**
- Π의 열 블록(크기 ≤ 2048)은 `cape.graph_ops.ppr_columns`로 구한다. Π는 대칭이므로 열 = 행이다.
- 1단계: 모든 블록에서 노름 ‖Π_u‖를 구한다.
- 2단계: 블록마다 G = Π · Π[:, B] (같은 급수를 dense 블록에 적용하는 `ppr_apply`를 `tokproto`에 만든다)를 구한다. cos = G / (‖Π_v‖ ‖Π_u‖)로 나누고, 자기 자신을 −inf로 둔 뒤 열마다 top-50을 구한다.
- 저장: `cache/{ds}/pprrow_top50.npz` (`nbr` int64 [N,50], `score` float32 [N,50])
- 검증: 노드 200개를 무작위로 골라 반환된 top-50 중 5개의 점수를 Π 열 두 개로 직접 다시 계산해 비교한다 (차이 ≤ 1e-4). 무작위 노드 1,000개와의 cosine이 50번째 점수를 넘지 않는지도 확인한다. 실패하면 중단한다.

**`train_tp.py`**
- 인자: `--dataset --neg {a2,b} --tok {sgc,pprrow} --lam --m --tau_att 0.1 --seed --episodes 4000 --eval_every 250 --n_jobs 8 --out`
- 학습 루프는 `cape.train_proto.main`을 따른다. seed 순서, 에피소드 생성, GCN 초기화, valid 평가, best 선택, test 평가가 모두 같아야 한다 (`CLAUDE.md` §6.1).
  - `cape.train_proto`의 `GCN`, `proto_acc`, `lr_task`, `ci`를 import해서 쓴다.
  - 에피소드: `make_topk_gen(top_in_pool(nbr, cue[:, :, 0], 10))`, rng = `np.random.default_rng(seed)`, 4,000개를 미리 생성.
- 토큰: 학습 전에 4,000개 에피소드 전체의 토큰 집합을 한 번에 만든다. rng를 쓰지 않는다.
- 손실: `CLAUDE.md` §7. 한 step = 에피소드 1개. 전체 그래프 GCN forward 후 앵커·query·토큰 임베딩을 H에서 뽑는다.
- λ = 0이면 토큰을 계산하지 않고 NaQ 손실 경로와 같은 계산을 한다 (arm 이름 `tp_l0`).
- JSON 출력 `results/tp/{ds}_{arm}_s{seed}.json` 필드
  - `config`, `test`(1s/5s × LR/proto, mean·ci95), `valid_curve`, `best_episode`, `best_valid`, `time`, `peak_gpu_GB`
  - `episode_stats`: `cape.generators.episode_stats`(분석용)
  - `token_stats` (λ > 0): 토큰 부족률, 평균 토큰 수, 토큰 순도(분석용)
  - `attn_diag` (λ > 0, best checkpoint, 학습 에피소드 앞 200개, 각 query의 자기 클래스 토큰에 대해): 정규화 엔트로피 평균, 같은 레이블 토큰 attention 질량 평균, 같은 레이블 토큰 비율(균등 attention일 때의 기대 질량)
- NaN/inf 손실이면 즉시 중단하고 그 run을 실패로 기록한다.

## 4. A — 학습 없는 토큰 진단 (`diag_tokens.py`)

- 대상: 세 데이터셋, seed 0의 학습 에피소드 앞 2,000개 (B와 같은 에피소드)
- 기록 (토큰 ranker `sgc`, `pprrow` × m ∈ {4, 8})
  - 토큰 순도, NaQ query 순도(같은 에피소드)
  - 토큰 부족률
  - hop 거리 분포 (앵커 1,000개까지): 토큰(m = 8)과 NaQ query 각각에 대해 1 / 2 / ≥3 / 연결 안 됨 비율과 중앙값. `scipy.sparse.csgraph.shortest_path(..., unweighted=True, indices=...)`
  - pprrow–ppr 겹침@8 (앵커 1,000개 평균)
- 출력: `results/TP01A_diag.json`, `results/TP01A_diag.md`

## 5. B1 — seed 0 격자

| 순서 | arm | neg | tok | λ | m |
|---|---|---|---|---|---|
| 1 | `tp_l0` | — | — | 0 | — |
| 2 | `tp_a2_sgc_l25_m4` | a2 | sgc | 0.25 | 4 |
| 3 | `tp_a2_sgc_l25_m8` | a2 | sgc | 0.25 | 8 |
| 4 | `tp_a2_sgc_l50_m4` | a2 | sgc | 0.5 | 4 |
| 5 | `tp_a2_sgc_l50_m8` | a2 | sgc | 0.5 | 8 |
| 6 | `tp_b_sgc_l50_m8` | b | sgc | 0.5 | 8 |
| 7 | `tp_a2_pprrow_l50_m8` | a2 | pprrow | 0.5 | 8 |

- 7 arm × 3 데이터셋 = 21 run, seed 0.
- 대기열의 맨 앞은 `tp_l0`의 `Amazon_eletronics`, `dblp` 두 run이다. 둘이 끝나면 **G0**를 계산한다.

**G0 — 학습 경로 회귀 게이트**
- 비교 대상: `results/expB/{ds}_naq_feat_s0.json` (eletronics 1s 74.58 / 5s 87.44, dblp 71.72 / 81.35)
- 조건: `tp_l0` s0과의 차이가 1s LR, 5s LR 모두 |Δ| ≤ 0.7%p (두 데이터셋 모두)
- 실패하면 나머지 B를 실행하지 않는다. 두 run의 valid 곡선을 기존 JSON과 나란히 보고하고 중단한다.

## 6. B2 — 설정 선택과 3 seed 확인

**선택 규칙 (사전 명시, valid만 사용)**
- C\* = B1의 arm 2–7 중 `best_valid`의 세 데이터셋 평균이 가장 높은 arm. 동률이면 §5 표의 순서가 앞선 arm.
- test 정확도로 고르지 않는다.

**실행**
- `tp_l0`, C\* × 세 데이터셋 × seed {1, 2} = 12 run
- B1의 seed 0 결과와 합쳐 seed 3개로 비교한다.

## 7. 시간 규칙

1. B1의 첫 run 2개가 끝나면 run당 시간을 재고, B1 + B2의 남은 벽시계 시간을 추정한다.
2. 추정이 **75분 이하**이면 전부 실행한다.
3. 75분을 넘으면 B2를 seed 1만으로 줄인다 (6 run). 그래도 넘으면 B2를 건너뛴다. B1은 줄이지 않는다.
4. A 진단이 15분을 넘기면 hop 거리 계산의 앵커를 300개로 줄인다.
5. 총 경과가 135분을 넘으면 진행 중인 run만 마치고 집계한다.
- 줄이거나 건너뛴 내용과 추정치를 보고서에 적는다.

## 8. 집계 — `summarize_tp01.py` → `results/TP01_summary.md`, `results/TP01_summary.json`

1. Ga, Gb, G0 결과
2. A 표: 데이터셋 × ranker × m의 토큰 순도, query 순도, 부족률, hop 분포, pprrow–ppr 겹침
3. B1 표 (데이터셋 × arm): best_valid, 1s LR ± CI, 5s LR ± CI, 1s proto, best ep, 토큰 부족률, attention 엔트로피, 같은 레이블 질량 / 균등 기대값
4. B1 차이 표: 각 arm − `tp_l0` (1s LR, 5s LR), 데이터셋별
5. 선택: arm별 세 데이터셋 평균 best_valid, C\*
6. B2 표 (데이터셋별): `tp_l0`, C\*의 seed별 1s/5s LR, 평균 ± sd, 짝지은 차이(같은 seed)의 seed별 값·평균·sd·부호 개수
7. 참고: `tp_l0` vs 기존 `naq_feat`(eletronics s0–2, dblp s0)
8. 사전 등록 대조표 (§9). 각 TPP의 기준식에 실제 값을 넣은 산술 비교만 적는다.

## 9. 사전 등록 (실행 전 커밋, 판정은 사람)

| ID | 예측 | 기준 | 확신도 |
|---|---|---|---|
| TPP001 | λ = 0 손실은 NaQ 손실과 같다 | Gb 1–4 모두 ≤ 1e-5 | 0.95 |
| TPP002 | 학습 경로가 기준선과 같다 | G0 통과 (eletronics, dblp의 1s·5s 모두 \|Δ\| ≤ 0.7) | 0.85 |
| TPP003 | sgc 토큰 순도는 데이터셋마다 query 순도와 반대로 움직인다 | m = 8에서 eletronics: 토큰 순도 < query 순도, dblp: 토큰 순도 > query 순도 | 0.80 |
| TPP004 | pprrow 토큰은 국소적이다 | 세 데이터셋 모두 pprrow–ppr 겹침@8 ≥ 0.5 이고, pprrow 토큰의 hop ≤ 2 비율 ≥ 0.7 | 0.65 |
| TPP005 | (주 가설, seed 0) TP가 NaQ-Feat보다 낫다 | A2-sgc 4개 중 best_valid 최고 설정의 1s LR − `tp_l0` ≥ +0.5인 데이터셋이 2개 이상 | 0.35 |
| TPP006 | 이득은 query 순도가 낮은 곳에서 크다 | `tp_a2_sgc_l50_m8` − `tp_l0` (1s)가 dblp에서 가장 크고 eletronics에서 가장 작다 | 0.45 |
| TPP007 | 네거티브 쪽 attention은 도움이 되지 않는다 | `tp_b_sgc_l50_m8` ≥ `tp_a2_sgc_l50_m8` (1s)인 데이터셋이 2개 이상 | 0.50 |
| TPP008 | sgc 토큰이 pprrow 토큰보다 낫다 | `tp_a2_sgc_l50_m8` ≥ `tp_a2_pprrow_l50_m8` (1s)인 데이터셋이 2개 이상 | 0.60 |
| TPP009 | attention은 균등하지 않다 | `tp_a2_sgc_l50_m8`의 정규화 엔트로피 평균 ≤ 0.9 (세 데이터셋 모두) | 0.50 |
| TPP010 | (3 seed) 선택 설정이 NaQ-Feat보다 낫다 | C\* − `tp_l0`의 seed 평균(1s) ≥ +0.5 이고 3 seed 중 양수 ≥ 2인 데이터셋이 2개 이상 | 0.30 |

해석 규칙 (사전 명시, 판정은 사람)
- TPP010 충족 → 다음 지시서에서 학습 가능한 head, 학습 길이, 10-way 평가를 검토한다.
- TPP005, TPP010 모두 불충족이고 모든 arm이 `tp_l0` 대비 −0.5 ~ +0.5 안 → 효과 없음. 종결 후보로 본다.
- 한 데이터셋에서만 이득 → 그 데이터셋의 토큰 순도(TPP003)와 함께 검토한다.
- 참고: test 태스크 500개의 95% CI는 약 ±1.1%p다. seed 0 단독의 1.0%p 미만 차이는 잡음일 수 있다.

## 10. 완료 기준과 보고

- [ ] §1의 커밋 (`TP01: instruction, preregistration, CLAUDE.md`) — 실행 전
- [ ] Ga, Gb 통과 기록
- [ ] `cache/{ds}/pprrow_top50.npz` 3개, `results/TP01A_diag.*`
- [ ] `results/tp/*.json`, `results/TP01_summary.*`
- [ ] `reports/TP01_report.md`, 커밋 (`TP01: token_proto first validation`)
- 보고 형식: `CLAUDE.md` §10. 핵심 수치에는 Ga/Gb/G0, A 표 요약, B1 차이 표, C\*와 선택 근거, B2 짝지은 차이, 사전 등록 대조, 실행 범위와 소요 시간을 넣는다.
