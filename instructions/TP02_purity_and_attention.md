# TP02 — 토큰 순도 상한, attention 통제, 최선 설정의 3 seed 확인

- 선행 문서: `CLAUDE.md` (특히 §1.1, §7), `reports/TP01_report.md`, `results/TP01_summary.md`
- **무인 실행 전제.** 질문하지 말고 이 지시서의 규칙에 따라 판단하고, 판단 근거를 보고서에 적는다.
- 시간 예산: **75분** (구현 10 / 실행 50 / 집계·보고 10 / 여유 5). 시간 규칙은 §6.
- 목적
  1. seed 0에서 세 데이터셋 모두 양수였던 `tp_a2_sgc_l25_m8`을 seed 3개로 확인한다.
  2. **토큰 순도 상한**: 토큰을 앵커와 같은 실제 레이블로 거른 arm(분석용 †)으로, 토큰이 순수하면 이득이 생기는지 잰다.
  3. **attention 통제**: 같은 토큰에 균등 가중치(평균)를 준 arm과 비교해, query별 attention의 몫을 잰다.
  4. attention을 날카롭게(τ_att = 0.02) 했을 때 엔트로피와 정확도가 어떻게 바뀌는지 잰다.
- 이 지시서에는 **설정 선택 단계가 없다.** 모든 arm은 아래 표에 고정되어 있다.

## 0. 범위와 레이블 규칙 예외
- 데이터셋 3개, 평가 5-way 1-shot / 5-shot, 학습 4,000 ep, eval 250 ep (TP01과 같음)
- **레이블 규칙 예외 (이 지시서에서만, `sgcora` arm에만 적용)**
  - `sgcora` arm은 토큰 후보를 고를 때 노드의 실제 레이블 `y`를 읽는다. **분석용 상한이며 방법이 아니다.**
  - 레이블은 토큰 필터에만 쓴다. 에피소드(앵커·query), 손실, checkpoint 선택에는 쓰지 않는다.
  - 결과 JSON에 `"analysis_only": true`를 넣고, 요약 표에서는 arm 이름 뒤에 †를 붙인다.
  - 레이블을 읽는 코드는 `tokproto/tokens.py`의 `oracle_filter` 한 함수에만 둔다. 다른 arm의 경로에서 이 함수가 호출되지 않는지 보고서에 적는다 (grep 결과).
- 범위 밖: 학습 가능한 head, 새 토큰 ranker, 에피소드 수 변경, 이 지시서 표에 없는 설정

## 1. 시작 절차 (git)

1. 상태 확인. 다르면 아무것도 바꾸지 말고 git 출력을 그대로 보고하고 멈춘다.
   - `git branch --show-current` → `token_proto`
   - `git log -1 --format=%h` → `c3f6ea7` (TP01 보고서 해시 커밋)
   - 추적 파일의 변경은 ` M CLAUDE.md` 하나뿐, `?? instructions/TP02_purity_and_attention.md`가 있어야 한다. 그 밖의 untracked 파일은 건드리지 않고 목록만 보고서에 적는다.
2. `git add CLAUDE.md instructions/TP02_purity_and_attention.md` → 커밋 `TP02: instruction, preregistration, CLAUDE.md` (코드 작성 전)

## 2. 구현 (`tokproto/` 확장, 기존 함수의 기본 동작은 바꾸지 않는다)

| 변경 | 내용 |
|---|---|
| `train_tp.py --attn {softmax,mean}` | 기본 `softmax` (TP01과 같음). `mean`이면 α를 유효 토큰 위 균등 분포로 둔다 (τ_att 무시) |
| `train_tp.py --tok sgcora` | 토큰 후보 = sgc top-50 중 **앵커와 실제 레이블이 같은 노드만**. 그 뒤 제외 규칙(에피소드 query 50개, 다른 앵커 4개)과 앞에서 m개 규칙은 `CLAUDE.md` §7과 같다. m개가 안 되면 있는 만큼 쓴다 |
| arm 이름 | `CLAUDE.md` §11 접미사 규칙. 예: `tp_a2_sgc_l25_m8_t02`, `tp_a2_sgc_l25_m8_mean`, `tp_a2_sgcora_l50_m8` |
| JSON | TP01 필드 + `attn` 모드, `analysis_only`. `token_stats`에 토큰 수 분포(0, 1–3, 4–7, m개 비율)를 추가 |

**회귀 확인 (실행 전)**
- `--attn softmax --tau_att 0.1 --tok sgc`의 기본 경로가 TP01과 같은지: `tp_a2_sgc_l25_m8` seed 0의 첫 에피소드 손실을, 초기화 직후 GCN에서 TP01 코드 경로와 새 코드 경로로 각각 계산해 |Δ| ≤ 1e-6인지 본다. 실패하면 중단한다.
- `--attn mean`에서 토큰이 모두 같은 벡터이면 softmax 결과와 같은지 (≤ 1e-6).

## 3. 실행 목록 (우선순위 순)

| 우선순위 | arm | 설정 | seed | run 수 |
|---|---|---|---|---|
| P1 | `tp_a2_sgc_l25_m8` | TP01과 같음 | 1, 2 | 6 |
| P2 | `tp_a2_sgcora_l50_m8` † | 레이블 필터 토큰, λ = 0.5, m = 8 | 0 | 3 |
| P2 | `tp_a2_sgcora_l25_m8` † | 레이블 필터 토큰, λ = 0.25, m = 8 | 0 | 3 |
| P3 | `tp_a2_sgc_l25_m8_mean` | 균등 α | 0, 1, 2 | 9 |
| P4 | `tp_a2_sgc_l25_m8_t02` | τ_att = 0.02 | 0 | 3 |

- 합계 24 run. 대기열(작업자 2개)은 위 순서대로 넣는다.
- **재사용** (다시 돌리지 않는다): `tp_l0` seed 0–2 (세 데이터셋 모두 TP01에 있음), `tp_a2_sgc_l25_m8` seed 0, `tp_a2_sgc_l50_m8` seed 0.
- 짝지은 비교는 같은 seed끼리 한다 (같은 seed = 같은 에피소드 열, 같은 초기화).

## 4. 진단 (학습 없음, 실행 전 또는 실행 중 CPU에서)
- `sgcora` 토큰의 부족률과 토큰 수 분포 (seed 0 에피소드 2,000개, m = 8). 데이터셋별.
- 같은 에피소드에서 `sgc` 토큰 순도(참고, TP01 값과 같아야 함).
- 출력: `results/TP02A_oracle_tokens.{json,md}`

## 5. 집계 — `tokproto/summarize_tp02.py` → `results/TP02_summary.{md,json}`

1. 회귀 확인 결과, `oracle_filter` 호출 경로 grep 결과
2. §4 진단 표
3. 3 seed 표 (데이터셋별): `tp_l0`, `tp_a2_sgc_l25_m8`, `tp_a2_sgc_l25_m8_mean`의 seed별 1s/5s LR, 평균 ± sd
4. 짝지은 차이 (seed별 값, 평균, sd, 양수 개수; 1s·5s)
   - `tp_a2_sgc_l25_m8` − `tp_l0`
   - `tp_a2_sgc_l25_m8_mean` − `tp_l0`
   - `tp_a2_sgc_l25_m8` − `tp_a2_sgc_l25_m8_mean` (**attention의 몫**)
5. seed 0 표: `tp_l0`, `sgc_l25_m8`, `sgc_l50_m8`, `sgcora_l25_m8`†, `sgcora_l50_m8`†, `sgc_l25_m8_t02`의 1s/5s LR, best ep, 토큰 순도, 엔트로피, 같은 레이블 질량 / 균등 기대값
   - 차이: `sgcora_lXX_m8` − `sgc_lXX_m8` (**순도의 몫**, λ별), `sgcora_lXX_m8` − `tp_l0`, `t02` − τ 0.1
6. 사전 등록 대조표 (§7)

## 6. 시간 규칙
1. 첫 run 2개가 끝나면 남은 벽시계 시간을 추정한다.
2. 추정이 **50분 이하**이면 전부 실행한다.
3. 넘으면 P4를 뺀다. 그래도 넘으면 P3을 seed 0만으로 줄인다. P1, P2는 줄이지 않는다.
4. 총 경과가 90분을 넘으면 진행 중인 run만 마치고 집계한다.

## 7. 사전 등록 (실행 전 커밋, 판정은 사람)

| ID | 예측 | 기준 | 확신도 |
|---|---|---|---|
| TPP011 | seed 0의 작은 양의 차이는 3 seed에서 대부분 사라진다 | `tp_a2_sgc_l25_m8` − `tp_l0`의 3 seed 평균(1s)이 +0.3 이상이고 양수가 2개 이상인 데이터셋이 **1개 이하** | 0.65 |
| TPP012 | query별 attention의 몫은 없다 | 세 데이터셋 모두 \|`sgc_l25_m8` − `sgc_l25_m8_mean`\| (3 seed 평균, 1s) ≤ 0.3 | 0.75 |
| TPP013 | τ_att = 0.02는 attention을 비균등하게 만든다 | `t02`의 정규화 엔트로피 ≤ 0.8 (세 데이터셋 모두) | 0.70 |
| TPP014 | 날카로운 attention은 정확도를 올리지 않는다 | `t02` − `sgc_l25_m8` (seed 0, 1s) ≤ +0.3인 데이터셋이 2개 이상 | 0.55 |
| TPP015 | 토큰이 순수해도 이득은 작다 | `sgcora_l50_m8`† − `tp_l0` (seed 0, 1s) ≥ +1.0인 데이터셋이 **1개 이하** | 0.60 |
| TPP016 | 순도를 올리면 순도가 가장 낮은 곳에서 가장 크게 바뀐다 | `sgcora_l50_m8`† − `sgc_l50_m8` (seed 0, 1s)가 eletronics에서 가장 크다 | 0.45 |

해석 규칙 (사전 명시, 판정은 사람)
- TPP015 적중 (순수 토큰도 +1.0 미만) 이고 TPP011 적중 → 토큰 prototype은 순도와 무관하게 효과가 작다. **TP 트랙 종결 후보.**
- TPP015 빗나감 (순수 토큰이 2개 이상 데이터셋에서 +1.0 이상) → 병목은 토큰 선택이다. 다음 지시서에서 레이블 없이 토큰 순도를 올리는 방법을 검토한다 (attention 개선보다 우선).
- TPP012 적중 → "query별 prototype" 주장은 근거가 없다. 남는 효과가 있다면 "K>1 soft support"로 설명해야 한다.
- TPP014 빗나감 (날카로운 attention이 2개 이상 데이터셋에서 +0.3 초과) → 학습 가능한 head를 다음 후보로 둔다.
- 참고: test 태스크 500개의 95% CI는 약 ±1.1%p, seed 간 sd는 TP01에서 0.15–0.45%p였다.

## 8. 완료 기준과 보고
- [ ] §1의 커밋 (실행 전)
- [ ] 회귀 확인 통과, grep 결과
- [ ] `results/TP02A_oracle_tokens.*`, `results/tp/*.json` (새 24개), `results/TP02_summary.*`
- [ ] `reports/TP02_report.md`, 커밋 (`TP02: purity ceiling and attention controls`)
- 보고 형식: `CLAUDE.md` §10. 핵심 수치에 §5의 3, 4, 5번 표와 사전 등록 대조, 실행 범위와 소요 시간을 넣는다.
