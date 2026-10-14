[T08 보고] 상태: SUCCESS (A′ 완료, B P1–P4 18 run 전부 완료. G8·U1–U7은 계산값만 적었다)
소요 시간: 약 43분 (01:16:06 시작 – 01:59 종료; 구현 약 1분 / A′ 25초 / B 39분 / 집계·보고 약 3분)
수행 내용:
- 지시서를 수정 없이 먼저 커밋했다 (f168cc7).
- novelty2.py 확장: pu_novelty2(class_weight 없는 LR), compute_estimators3 (PCA256 + pu_feat2 / pu_both2 / pu_struct)
- 새 파일: cape/diag_novelty3.py, cape/run_T08.sh, cape/summarize_t08.py
- generators.py 확장
  - make_mix(일반 혼합)를 추가하고 make_mix_alt를 이것으로 재구성했다. rng 소비 순서가 같아서 기존 arm 동작은 바뀌지 않는다.
  - 새 arm: naq_mix_r, sup_only_r, mix_alt_p{10,50}_r, nov2c_t{τ}_r
- 소스 검사: T07과 같은 검사기 (`.split(` 메서드 호출 제외)로 통과했다 (results/T08_source_check.json).
- A′ (eletronics, dblp, clothing; 캐시가 있어 clothing 포함)
  - 데이터셋당 12–14초 (PCA 9–13초, PU 적합 0.1–1.3초). T07 pu_both는 약 10분이었다.
  - 10분 규칙 미발동. pu_both2를 포함했다.
- E2* = **pu_both2**
  - 두 데이터셋 모두 보정 오차 ≤ 0.15인 변형이 없었다 (pu_both2 elec +0.156, pu_struct dblp −0.232).
  - 그래서 대체 규칙 "평균 보정 오차 최소"를 적용했다: pu_both2 .113 < pu_struct .149 < pu_feat2 .317.
- 시간 규칙: 첫 2 run은 각 255s. 18 run이면 9쌍 × 4.3분 ≈ 38.6분 ≤ 40분이라 P1–P4 전부 포함했다.
  - 실측 B: 01:17:34–01:56:38 (39분). 다른 사용자 GPU 프로세스 없음. NaN·OOM·재시도 없음.
핵심 수치:
- A′ (r=0.5)
| ds | 변형 | AUROC | 보정 오차 (mean p − 실제) | 적합 s |
|---|---|---|---|---|
| elec | pu_feat2 | .831 | −.260 | 1.0 |
| elec | pu_both2 | .730 | +.156 | 0.4 |
| elec | pu_struct | .753 | +.067 | 0.1 |
| dblp | pu_feat2 | .724 | −.375 | 1.2 |
| dblp | pu_both2 | .731 | −.069 | 0.7 |
| dblp | pu_struct | .764 | −.232 | 0.1 |
| clothing | pu_feat2 / pu_both2 / pu_struct | .895 / .836 / .870 | −.193 / +.070 / −.060 | |
  - T07 pu_feat (참고): AUROC elec .867, dblp .749; 보정 오차 .263 / .396
  - class_weight를 없애도 pu_feat2의 과소추정은 남는다.
- 짝지은 차이 (1s, seed별 → 평균; ‡는 L 레이블 지도 에피소드 포함)
| 비교 | eletronics | dblp |
|---|---|---|
| naq_mix‡ − best | +.33 / −.10 → +0.11 | +3.41 / +2.57 → **+2.99** |
| mix_alt‡ − naq_mix‡ (novel 앵커 몫) | +1.46 / +1.66 → **+1.56** | +.17 / −.86 → −0.35 |
| naq_mix‡ − nov2_t1 | −1.39 | +3.37 |
| sup_only‡ − best | −5.68 / −5.77 → **−5.73** | +1.28 / +.16 → +0.72 |
| mix_alt‡ − sup_only‡ | +7.40 | +1.93 |
| mix_alt p10 / p30 / p50 − best (s0) | +1.98 / +1.79 / +0.67 | +2.40 / +3.58 / +3.60 |
| nov2c_t1 − best | +.12 / +.59 / +.85 → +0.52 | +1.18 / +.33 / +1.05 → +0.86 |
| nov2c_t1 − nov2_t1 (T07, pu_feat 가중) | −1.10 (3/3 음수) | +0.96 (3/3 양수) |
  - 5s 평균: naq_mix − best +0.12 / +0.93, mix_alt − naq_mix +0.41 / +0.08, sup_only − best −3.44 / −0.80, nov2c − best +0.31 / +0.29
- 분해 (1s)
  - elec 경로 A: NaQ → naq_mix +0.11 → mix_alt +1.56 = +1.67
  - elec 경로 B: NaQ → nov2_t1 +1.62 → mix_alt +0.17 = +1.78
  - dblp 경로 A: +2.99, −0.35 = +2.65
  - dblp 경로 B: −0.11, +3.02 = +2.91
  - 두 경로 합의 차이: elec −0.11, dblp −0.27. seed 집합이 다르기 때문에 생긴 차이이고, 같은 seed라면 대수적으로 같다.
- NMI (seed 0; NMI_test / NMI_base, NaQ 대비 Δ)
  - elec
    - naq_mix +.003 / +.030
    - sup_only −.079 / +.063
    - nov2c_t1 +.011 / −.013
    - 참고 (T07): nov2_t1 +.017 / −.038, mix_alt +.011 / +.028
  - dblp
    - naq_mix +.029 / +.044
    - sup_only +.000 / +.054
    - nov2c_t1 +.019 / +.008
    - 참고 (T07): mix_alt +.031 / +.045
- G8 계산값
  - G8-a: elec mix_alt − naq_mix = +1.56 (s0 +1.46, s1 +1.66). 기준: ≥ +0.7, 두 seed 모두 양수.
  - G8-b: dblp mix_alt − sup_only = +1.93. 기준: ≥ +0.7.
  - G8-c: E2*(pu_both2) 보정 오차 elec +0.156, dblp −0.069. 기준: 두 데이터셋 모두 ≤ 0.15.
- 사전 등록 대조 (계산값만)
  - U1: pu_feat2 보정 오차 −.260 / −.375, AUROC 변화 −.035 / −.024 (T07 pu_feat 대비)
  - U2: elec nov2c_t1 − NaQ = +0.52
  - U3: +1.56 (s0 +1.46, s1 +1.66)
  - U4: dblp mix_alt − naq_mix = −0.35
  - U5: sup_only − NaQ = elec −5.73, dblp +0.72
  - U6: dblp p10 / p30 / p50 = 75.68 / 76.86 / 76.88, elec = 76.56 / 76.36 / 75.25 (범위 1.31)
  - U7: 합 A +1.67, 합 B +1.78, 차이 −0.11
- 사실 관찰 (짧게)
  - elec: 지도 30%만 섞은 naq_mix는 NaQ와 같다 (+0.11). novel 앵커를 더하면 +1.56이다. sup_only는 −5.7이다.
  - dblp: 지도 30% 혼합 자체가 +2.99다. novel 앵커를 더한 효과는 −0.35이고, nov2c(보정 p)는 +0.86이다.
  - 보정된 p(pu_both2)로 가중하면 앵커 집중도가 낮아진다 (기대 novel 비율 elec .678 vs T07 pu_feat .877).
    - elec 이득은 +1.62에서 +0.52로 줄었다.
    - dblp는 −0.11에서 +0.86으로 늘었다.
계획 대비 차이·임의 결정:
- PCA: sklearn PCA(256, randomized, random_state 0)를 전체 노드의 Xs(행 정규화 SGC)에 비지도로 적합했다.
- pu_both2 입력: [PCA256, known_struct, logdeg] 전체를 StandardScaler로 표준화했다. pu_feat2 입력은 PCA 출력 그대로다.
- pu_struct는 T07 정의(class_weight balanced) 그대로 다시 계산했다. 값은 T07과 같다 (AUROC .753 / .764).
- 지도 에피소드 비율: 에피소드마다 Bernoulli(p)로 정했다 (mix_alt와 같은 방식). sup_only는 p=1.0이다.
- 분해 표: 각 단계는 두 arm의 공통 seed에 대한 짝지은 차이의 평균이다.
- NMI 표: T07 행도 같은 코드로 다시 계산했다. 값은 T07과 같다.
- B 기한 02:12 이후 새 run 시작 금지 (해당 없음).
문제·미해결:
- mix_alt 계열은 seed가 2개 이하이고, p10 / p50은 seed 0 하나다.
- E2*는 사전 규칙의 대체 경로("보정 오차 최소")로 선택됐다. elec 보정 오차 +0.156은 기준 0.15를 근소하게 넘는다.
- results/T03_summary_message.md, results/T07_message.txt는 커밋하지 않았다.
변경·생성 파일:
- cape/novelty2.py, cape/generators.py, cape/diag_novelty3.py, cape/run_T08.sh, cape/summarize_t08.py
- results/T08A_calibration.{md,json}, results/T08A_{ds}.json ×3, results/T08_source_check.json, results/T08_summary.md
- results/expB/ JSON 18개
- reports/T08_report.md
