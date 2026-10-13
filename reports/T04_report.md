[T04 보고] 상태: SUCCESS (A 12 run + B 2 run 완료. C1–C4는 계산만 했고 판정은 사람이 한다)
소요 시간: 약 45분 (구현 5분 / 재현성 확인 1분 / 실행 38분 / 집계·보고 2분)
수행 내용:
- train_proto.py 최소 수정: episodes ≠ 4000이면 출력 파일 이름에 `_e{episodes}`를 붙인다 (2줄). 생성기와 모델은 T03 그대로다.
- cape/run_T04.sh: 프로세스 2개.
  - p1 = eletronics A 6개 + B naq_feat e8000
  - p2 = dblp A 6개 + B cape_lite e8000 (데이터셋은 eletronics)
  - B는 각 프로세스의 마지막에 두었고, logs/T04_skipB 파일이 있으면 건너뛰게 했다.
- cape/summarize_t04.py → results/T04_summary.md, results/T04_valid_curves.png
- 재현성 확인 (eletronics naq_feat s0, 500 ep, 2회)
  - valid: ep0 .416/.416, ep250 .635/.635, ep500 .6775/.676 (차이 0.15%p)
  - test 1s LR: 66.25 / 66.30
  - 1%p 이하라서 계속 진행했다. 원인 추정: CUDA sparse mm(cuSPARSE)과 backward의 atomic 누적 순서가 비결정적이다.
- B 실행 여부 판단
  - 첫 2 run은 각각 약 255s (병렬 step 60–62ms)
  - 추정 총 시간: 6×256s + 8,000 ep run 약 510s ≈ 34분 ≤ 45분 → B 포함
  - 실측: 11:21:57–11:59:36, 37.6분. NaN·OOM 없음.
핵심 수치:
- (b) seed 3개, 평균 ± 표준편차 (1s LR / 5s LR)
| ds | arm | 1s LR | 5s LR |
|---|---|---|---|
| elec | naq_feat | 74.59 ± 0.09 | 87.42 ± 0.02 |
| elec | lr_only | 74.31 ± 0.18 | 87.17 ± 0.28 |
| elec | cape_lite | 72.88 ± 0.53 | 86.56 ± 0.21 |
| dblp | naq_diff | 73.73 ± 0.44 | 82.80 ± 0.40 |
| dblp | lr_only | 73.59 ± 0.25 | 82.43 ± 0.09 |
| dblp | cape_lite | 72.97 ± 0.99 | 82.17 ± 0.50 |
- (c) 짝지은 차이 (같은 seed). 열 순서: s0, s1, s2, 평균 (표준편차), 음수/양수 개수
| 비교 | shot | s0 | s1 | s2 | 평균 (sd) | −/+ |
|---|---|---|---|---|---|---|
| elec cape_lite−feat | 1s | −1.32 | −1.57 | −2.23 | −1.71 (0.47) | 3/0 |
| elec cape_lite−feat | 5s | −0.71 | −0.81 | −1.08 | −0.87 (0.19) | 3/0 |
| elec lr_only−feat | 1s | −0.47 | −0.23 | −0.15 | −0.28 (0.16) | 3/0 |
| elec lr_only−feat | 5s | −0.17 | −0.05 | −0.54 | −0.25 (0.26) | 3/0 |
| dblp cape_lite−diff | 1s | −1.18 | −1.39 | +0.29 | −0.76 (0.91) | 2/1 |
| dblp cape_lite−diff | 5s | +0.31 | −1.23 | −0.95 | −0.63 (0.82) | 2/1 |
| dblp lr_only−diff | 1s | +0.49 | −0.85 | −0.06 | −0.14 (0.67) | 2/1 |
| dblp lr_only−diff | 5s | +0.05 | −0.37 | −0.78 | −0.37 (0.42) | 2/1 |
- (d) 학습 길이 (eletronics, seed 0). 순서: 1s LR / 5s LR / best ep
  - naq_feat: 4000 ep 74.58 / 87.44 / 4000, 8000 ep 75.43 / 88.18 / 7750
  - cape_lite: 4000 ep 73.26 / 86.72 / 3250, 8000 ep 74.38 / 87.15 / 7500
  - cape_lite − naq_feat (1s / 5s): 4000 ep −1.32 / −0.71, 8000 ep −1.06 / −1.03
  - valid 곡선 (1,000 ep 간격, ep 0→8000)
    - naq_feat: .416 .716 .742 .746 .762 .767 .772 .779 .784
    - cape_lite: .416 .690 .704 .746 .746 .756 .753 .763 .766
  - 두 곡선 모두 8,000까지 완만하게 오르며 아직 포화 전이다.
- (e) 판정 근거 (계산값, 기준과의 산술 비교만)
  - C1: 평균(cape_lite − best, 1s) = elec −1.71, dblp −0.76. 음수 5/6. 기준(둘 다 ≤ −0.5, 음수 ≥ 5/6)과 산술상 일치한다.
  - C2: 평균 |lr_only − best| (1s) = elec 0.28, dblp 0.47. 기준(≤ 1.0)과 산술상 일치한다.
  - C3: 8000 ep에서 cape_lite − naq_feat (1s) = −1.06. 값이 0보다 작다.
  - C4: naq_feat 8000 − 4000 (1s) = +0.86. 값이 +1.5보다 작다.
계획 대비 차이·임의 결정:
- 재현성 확인 run은 results/expB를 오염시키지 않도록 --out logs/T04_repro/로 저장했다.
- B의 두 run(모두 eletronics)을 서로 다른 프로세스에 하나씩 배치했다. 지시서 예시와 같다.
- seed 0 8000 run은 rng가 같아서 처음 4,000개 에피소드가 4000 run과 동일하다. 학습 기간 4,000 ep 연장 비교에 해당한다.
  - 단, 비결정성 때문에 4000 시점의 모델이 비트 단위로 같지는 않다.
- (e)에는 기준과의 산술 비교만 적었다. 결론 판정은 하지 않았다.
- std는 표본 표준편차(ddof=1)로 계산했다.
문제·미해결:
- dblp의 cape_lite 1s는 seed별 편차가 크다 (sd 0.99, s2는 +0.29로 부호가 반대).
- dblp best ep이 이른 run이 있다 (cape_lite s1 = 500, lr_only s2 = 1000). valid 50태스크 기준 선택의 잡음일 수 있다.
- results/T03_summary_message.md(이전 요청으로 저장한 파일)는 커밋에 포함하지 않았다.
변경·생성 파일:
- cape/train_proto.py (2줄), cape/run_T04.sh, cape/summarize_t04.py
- results/expB/{ds}_{arm}_s{1,2}.json ×12, results/expB/Amazon_eletronics_{naq_feat,cape_lite}_s0_e8000.json
- results/T04_summary.md, results/T04_valid_curves.png
- reports/T04_report.md, instructions/T04_confirm_seeds.md
커밋: f5d0854 (branch cape-validation)
