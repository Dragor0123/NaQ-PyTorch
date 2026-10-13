# T04 — T03 결론의 견고성 확인 (seed 추가, 학습 길이 점검)

- 선행 문서: `CLAUDE.md` (§3, §8, §10, §12), `reports/T03_report.md`, `results/summary.md`
- 시간 예산: **40분** (구현 약 5분, 실행 약 30분, 집계·보고 약 5분)
- 목적: T03의 다음 세 관찰이 seed 잡음이나 학습 부족 때문이 아닌지 확인한다.
  - O1. `cape_lite`가 NaQ-best보다 낮다 (−1.32 / −1.18)
  - O2. `lr_only`가 NaQ-best와 같다 (−0.47 / +0.49)
  - O3. Electronics에서 학습이 부족했을 수 있다 (best 에피소드가 대부분 마지막 구간)
- 범위 밖
  - 새 arm, 새 생성기 규칙
  - 하이퍼파라미터 변경 (에피소드 수는 아래 명시분만)
  - scorer 재학습, TEG

## 1. 실행 목록 (총 14 run)

T03 코드(`cape/train_proto.py`, `cape/generators.py`)를 그대로 쓴다. 생성기와 모델은 바꾸지 않는다.

**A. seed 추가** (12 run, 각 4,000 에피소드, `--eval_every 250`)

| dataset | arm | seed |
|---|---|---|
| `Amazon_eletronics` | `naq_feat`, `lr_only`, `cape_lite` | 1, 2 |
| `dblp` | `naq_diff`, `lr_only`, `cape_lite` | 1, 2 |

- NaQ-best는 T03 seed 0에서 정해진 arm을 그대로 쓴다 (eletronics = `naq_feat`, dblp = `naq_diff`).
- seed 0 결과는 T03의 JSON을 재사용한다.
- seed는 학습 에피소드 생성과 모델 초기화에 쓰인다. split과 test 태스크는 고정이다.

**B. 학습 길이 점검** (2 run, `Amazon_eletronics`, seed 0)

| arm | 에피소드 | eval_every |
|---|---|---|
| `naq_feat` | 8,000 | 250 |
| `cape_lite` | 8,000 | 250 |

- 출력 파일 이름은 `results/expB/{ds}_{arm}_s{seed}_e8000.json`으로 한다. T03 결과를 덮어쓰지 않는다.
- 이를 위해 `train_proto.py`에서 에피소드 수가 4,000이 아니면 파일 이름에 `_e{episodes}`를 붙이도록 최소한으로 수정한다.

**실행 방식**
- T03에서 두 프로세스를 병렬로 돌렸을 때 step 시간이 27ms에서 60ms로 늘었다. 병렬 이득이 거의 없었다는 뜻이다.
- 두 프로세스로 나누되, 각 프로세스의 총 GPU 시간이 비슷하도록 run을 배분한다.
  - 예: 프로세스 1 = eletronics A 6개 + B 1개, 프로세스 2 = dblp A 6개 + B 1개
- 스크립트는 `cape/run_T04.sh`, 로그는 `logs/T04_*.log`에 남긴다.
- 시작 후 첫 run 2개가 끝나면 남은 시간을 추정한다. 추정 총 벽시계 시간이 45분을 넘으면 B를 빼고 진행하고, 보고서에 기록한다.

## 2. 집계 — `cape/summarize_t04.py`

실행: `python -m cape.summarize_t04` → `results/T04_summary.md`

**(a) seed별 표** (dataset × arm × seed 0/1/2)
- test 1s LR, test 5s LR, query_purity, best 에피소드

**(b) 요약 표** (dataset × arm)
- 1s LR의 평균 ± 표준편차 (seed 3개)
- 5s LR의 평균 ± 표준편차

**(c) 짝지은 차이** (같은 seed끼리 차를 구한다)
- `cape_lite` − NaQ-best
- `lr_only` − NaQ-best
- 각각 seed별 값 3개, 평균, 표준편차, 같은 부호의 개수를 1s와 5s로 적는다.

**(d) 학습 길이 점검** (B)
- `naq_feat`, `cape_lite`의 4,000 대비 8,000 에피소드 결과: 1s/5s LR, best 에피소드
- 두 학습 길이에서의 (`cape_lite` − `naq_feat`) 차이
- 8,000 run의 valid 곡선: 250 에피소드마다의 값을 짧은 표로 적거나, 그림 `results/T04_valid_curves.png`로 남긴다.

**(e) 판정 근거** (아래 기준을 계산만 하고, 판정은 사람이 한다)

| ID | 확인 대상 | 기준 (사전 명시) |
|---|---|---|
| C1 | O1이 견고한가 | 두 데이터셋 모두에서 (`cape_lite` − NaQ-best)의 seed 평균 ≤ −0.5 이고, 6개 (데이터셋 × seed) 쌍 중 5개 이상이 음수 |
| C2 | O2가 견고한가 | 두 데이터셋 모두에서 \|`lr_only` − NaQ-best\|의 seed 평균 ≤ 1.0 |
| C3 | 학습 부족이 결론을 바꾸는가 | 8,000 에피소드에서 (`cape_lite` − `naq_feat`) ≥ 0 이면 "결론 바뀔 수 있음", 아니면 "유지" |
| C4 | 학습 부족의 크기 | `naq_feat`의 8,000 − 4,000 차이 (1s LR). ≥ +1.5이면 이후 실험 기본 에피소드 수를 다시 정해야 함 |

## 3. 완료 기준

- [ ] `train_proto.py` 최소 수정 (파일 이름 규칙), `run_T04.sh`, `summarize_t04.py`
- [ ] 결과 JSON 12개(A)와 2개(B), 또는 B를 뺀 경우 그 사유
- [ ] `results/T04_summary.md`
- [ ] `reports/T04_report.md` 작성, 커밋 (`T04: seed and training-length robustness check`)

## 4. 중단 조건

- 같은 seed와 설정으로 T03 seed 0 run 하나를 다시 돌렸을 때 결과가 크게 다르다 (아래 확인 사항 참고).
- NaN, OOM이 발생한다.
- 첫 2 run 이후 추정 총 시간이 B를 빼고도 45분을 넘는다.

**재현성 확인** (실행 전 1회, 짧게)
- `Amazon_eletronics`의 `naq_feat` seed 0을 `--episodes 500 --eval_every 250`으로 두 번 돌린다.
- 두 결과의 valid 정확도가 소수 셋째 자리까지 같은지 본다.
- 다르면 비결정성의 원인(cuSPARSE 등)을 한 줄로 적고 계속 진행한다. 다만 차이가 1%p를 넘으면 중단한다.

## 5. 보고에 반드시 포함할 항목

- `results/T04_summary.md`의 (b), (c), (d), (e) 표
- 재현성 확인 결과
- 실제 소요 시간과 B 실행 여부
- 계획 대비 차이와 임의로 내린 결정
- 해석은 짧게, 사실 위주로 적는다.
