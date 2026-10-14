# T08 — 혼합의 통제 실험과 PU 보정 수정 (1시간 단위, 무인 실행 가능)

- 선행 문서: `CLAUDE.md`, `reports/T07_report.md`, `results/T07_summary.md`, `results/T07A_estimators.md`
- 무인 실행 전제. 질문하지 말고 규칙에 따라 판단하고 보고서에 기록한다.
- 시간 예산: **60분** (구현 10 / A′ 5 / B 40 / 집계·보고 5). 시간 규칙은 §4.
- **사람 확인 사항 처리**: T07의 소스 검사기 수정(`.split(` 메서드 호출 제외)은 사람이 확인했고 타당하다. 같은 검사기를 그대로 쓴다.
- 목적
  1. **통제 실험**: T07 `mix_alt`의 이득 중 "novel 앵커링"의 몫과 "지도 에피소드"의 몫을 분리한다.
  2. **PU 보정 수정**: p(novel|u)의 과소추정을 고치고, 추정기 적합 시간을 수 분 이내로 줄인다.
  3. 보정된 p로 비례 샘플링(τ=1)을 다시 돌린다.

## 0. 범위와 규칙
- 레이블 규칙 `CLAUDE.md` §3. 지도 에피소드(`sup_*`, `*_mix*`)는 L의 레이블만 쓴다. 소스 검사를 다시 한다.
- 범위 밖: TEG, clothing, r ≠ 0.5, τ ≠ 1, 새 추정기 유형 추가

## 1. A′ — PU 보정 수정 (`cape/novelty2.py` 확장, `cape/diag_novelty3.py`)

대상: eletronics, dblp (clothing은 캐시가 있으면 함께, 없으면 생략), r = 0.5

**추정기 변형** (T07의 pu_* 와 비교)
| 이름 | 변경점 |
|---|---|
| `pu_feat2` | 입력 = PCA(256)로 축소한 Xs (PCA는 전체 노드에서 비지도 적합). `LogisticRegression(C=1.0, max_iter=2000)`, **class_weight 없음** |
| `pu_both2` | 입력 = [PCA256(Xs), known_struct, logdeg] 표준화, 같은 LR 설정 |
| `pu_struct` | T07과 같음 (참고용) |
- c는 T07과 같은 OOF 방식. p_known = min(1, g/c), p_novel = 1 − p_known.
- 기록: AUROC, 보정 절대 오차(mean p_novel − 실제 novel 비율), 적합 시간, 기대 앵커 novel 비율(τ=1)
- **E2\* 선택 규칙 (사전 명시)**: 두 데이터셋 모두 보정 오차 ≤ 0.15인 변형 중 평균 AUROC 최고. 해당 변형이 없으면 보정 오차 평균이 가장 작은 변형.
- 저장: `cache/{ds}/novelty3_r0.5.npz`. 출력: `results/T08A_calibration.md`, `.json`

## 2. B — arm 정의 (r = 0.5, 학습기·평가는 T03과 같음)

| arm | 정의 |
|---|---|
| `naq_mix` | 에피소드의 70%는 NaQ-best(전체 노드 균등 앵커), 30%는 L 지도 에피소드. Bernoulli(0.3)는 `mix_alt`와 같은 방식 |
| `sup_only` | 100% L 지도 에피소드 (5-way 1-shot, 클래스당 query 10, 모두 L에서) |
| `mix_alt_p10`, `mix_alt_p50` | `mix_alt`에서 지도 비율만 0.1 / 0.5 |
| `nov2c_t1` | 앵커 ∝ p_{E2\*}(novel\|u), τ = 1 (보정된 p) |
| 비교용 기존 | `naq_feat`/`naq_diff` (s0–2), `nov2_t1_r0.5` (s0–2), `mix_alt_r0.5` (s0, s1) — 재사용 |

## 3. B — 실행 목록 (우선순위 순)

| 우선순위 | run | 수 |
|---|---|---|
| P1 | `naq_mix` × {eletronics, dblp} × seed {0, 1} | 4 |
| P2 | `nov2c_t1` × {eletronics, dblp} × seed {0, 1, 2} | 6 |
| P3 | `sup_only` × {eletronics, dblp} × seed {0, 1} | 4 |
| P4 | `mix_alt_p10`, `mix_alt_p50` × {eletronics, dblp} × seed 0 | 4 |

- 모든 run은 `--save_emb`를 켠다. 출력 `results/expB/{ds}_{arm}_r0.5_s{seed}.json`
- `cape/run_T08.sh`: 대기열 방식, 작업자 2개

## 4. 시간 규칙
1. A′가 10분을 넘기면 `pu_both2`를 건너뛰고 `pu_feat2`만으로 E2\*를 정한다. A′가 실패하면 E2\* = `pu_struct`로 두고 진행하며 사유를 적는다.
2. 첫 run 2개 후 추정 B 벽시계 시간이 **40분 이하**가 되는 우선순위까지만 포함한다. 제외분은 보고서에 적는다.
3. 총 경과 65분을 넘으면 진행 중 run만 마치고 집계한다.

## 5. 집계 — `cape/summarize_t08.py` → `results/T08_summary.md`
1. A′ 표: 변형 × 데이터셋의 AUROC, 보정 오차, 적합 시간, E2\*
2. 결과 표 (dataset × arm × seed): 1s/5s LR, query_purity, anchor_novel_frac, anchor_ent, best ep
3. 짝지은 차이 (1s/5s, seed별·평균·sd·부호 개수)
   - `naq_mix` − NaQ-best, `mix_alt` − `naq_mix` (**novel 앵커링의 몫**), `naq_mix` − `nov2_t1`
   - `sup_only` − NaQ-best, `mix_alt` − `sup_only`
   - `mix_alt_p10` / `p30(=mix_alt)` / `p50` − NaQ-best (seed 0)
   - `nov2c_t1` − NaQ-best, `nov2c_t1` − `nov2_t1`
4. 분해 표 (1s, seed 평균): 데이터셋별로 NaQ → +지도 30%(naq_mix) → +novel 앵커(mix_alt) 의 증분, 그리고 NaQ → +novel 앵커(nov2_t1) → +지도 30%(mix_alt) 의 증분. 두 경로의 합이 같은지 적는다.
5. NMI (seed 0): `naq_mix`, `sup_only`, `nov2c_t1`을 T07 표에 추가
6. 게이트 G8 계산값 (판정은 사람)
   - G8-a: eletronics에서 (`mix_alt` − `naq_mix`) ≥ +0.7 (seed 평균, 2 seed 모두 양수)
   - G8-b: dblp에서 `mix_alt` − `sup_only` ≥ +0.7
   - G8-c: E2\*의 보정 오차가 두 데이터셋 모두 ≤ 0.15
7. 사전 등록 대조표 (§6)

## 6. 사전 등록 (실행 전 커밋, 판정은 사람)
| ID | 예측 | 기준 | 확신도 |
|---|---|---|---|
| U1 | class_weight 제거와 PCA로 보정이 좋아진다 | `pu_feat2` 보정 오차 ≤ 0.15 (두 데이터셋), AUROC 하락 ≤ 0.03 (T07 pu_feat 대비) | 0.60 |
| U2 | 보정된 p로도 이득이 유지된다 | eletronics `nov2c_t1` − NaQ ≥ +1.3 | 0.60 |
| U3 | novel 앵커링은 eletronics에서 독립적 몫이 있다 | eletronics `mix_alt` − `naq_mix` ≥ +0.7, 2 seed 모두 양수 | 0.60 |
| U4 | dblp에서는 지도 에피소드가 이득의 원천이다 | dblp \|`mix_alt` − `naq_mix`\| ≤ 1.0 | 0.60 |
| U5 | 지도 단독은 부족하다 | eletronics `sup_only` − NaQ ≤ −2.0; dblp `sup_only` − NaQ ≤ +1.0 | 0.65 |
| U6 | 지도 비율은 dblp에서 민감, eletronics에서 둔감 | dblp p50 > p30 > p10 (1s, seed 0); eletronics 세 값의 범위 ≤ 1.0 | 0.50 |
| U7 | 분해의 두 경로가 대략 일치한다 | 두 경로 총합의 차이 ≤ 0.7 (eletronics) | 0.55 |

해석 규칙 (사전 명시)
- U3 적중 → "지도 + novel 앵커링" 두 구성 요소 모두 기여. 방법을 혼합으로 확정.
- U3 빗나감(eletronics에서 `naq_mix` ≈ `mix_alt`) → novel 앵커링의 독립 기여 없음. 방법은 "지도 + NaQ 혼합"으로 축소되며 방향 재검토.

## 7. 완료 기준과 보고
- [ ] 실행 전 이 파일을 수정 없이 커밋 (`T08: instruction and preregistration`)
- [ ] 코드, `results/T08A_calibration.*`, `results/T08_summary.md`, JSON
- [ ] `reports/T08_report.md`, 커밋 (`T08: mixing controls and PU calibration`)
- 보고 형식 `CLAUDE.md` §12. 핵심 수치에 A′ 표, 짝지은 차이, 분해 표, G8, 사전 등록 대조, 실행 범위·소요 시간.
