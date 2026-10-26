# T09 summary — 5-seed expansion (Electronics, dblp) + Amazon_clothing (r = 0.5)

‡ = L의 레이블로 만든 지도 에피소드를 포함하는 arm. † = 분석용 상한(split 이용, 방법 아님). NaQ-best: Electronics naq_feat, dblp naq_diff, Clothing naq_feat. 모든 짝지은 차이는 두 arm의 공통 seed에서 계산.

## 1. 주 표 (seed 평균 ± sd)

| dataset | arm | n | seeds | 1s LR | 5s LR | q_purity | anchor_novel_frac | best ep |
|---|---|---|---|---|---|---|---|---|
| Amazon_eletronics | naq_feat | 5 | 0,1,2,3,4 | 74.72 ± 0.19 | 87.52 ± 0.13 | 0.647 | 0.433 | 3800 |
| Amazon_eletronics | nov | 5 | 0,1,2,3,4 | 76.02 ± 0.42 | 88.21 ± 0.15 | 0.669 | 0.878 | 3600 |
| Amazon_eletronics | naq_mix ‡ | 5 | 0,1,2,3,4 | 74.56 ± 0.52 | 87.35 ± 0.21 | 0.753 | 0.301 | 3800 |
| Amazon_eletronics | mix_alt ‡ | 5 | 0,1,2,3,4 | 75.87 ± 0.54 | 87.84 ± 0.23 | 0.768 | 0.614 | 3850 |
| Amazon_eletronics | sup_only ‡ | 5 | 0,1,2,3,4 | 68.42 ± 0.53 | 83.70 ± 0.30 | 1.000 | 0.000 | 3300 |
| Amazon_eletronics | naq_unl † | 3 | 0,1,2 | 77.18 ± 0.56 | 88.63 ± 0.22 | 0.613 | 1.000 | 3917 |
| dblp | naq_diff | 5 | 0,1,2,3,4 | 73.78 ± 0.54 | 82.86 ± 0.30 | 0.293 | 0.427 | 2850 |
| dblp | nov | 5 | 0,1,2,3,4 | 73.59 ± 0.56 | 82.89 ± 0.27 | 0.381 | 0.809 | 1850 |
| dblp | naq_mix ‡ | 5 | 0,1,2,3,4 | 76.75 ± 0.30 | 83.58 ± 0.07 | 0.506 | 0.294 | 3700 |
| dblp | mix_alt ‡ | 5 | 0,1,2,3,4 | 76.50 ± 0.54 | 83.66 ± 0.17 | 0.570 | 0.568 | 3650 |
| dblp | sup_only ‡ | 5 | 0,1,2,3,4 | 74.29 ± 0.28 | 81.88 ± 0.32 | 1.000 | 0.000 | 2900 |
| dblp | naq_unl † | 3 | 0,1,2 | 74.16 ± 0.68 | 83.14 ± 0.32 | 0.306 | 1.000 | 2667 |
| Amazon_clothing | naq_feat | 3 | 0,1,2 | 84.88 ± 0.43 | 92.25 ± 0.15 | 0.596 | 0.570 | 3750 |
| Amazon_clothing | naq_diff | 3 | 0,1,2 | 80.72 ± 0.51 | 91.10 ± 0.27 | 0.569 | 0.570 | 583 |
| Amazon_clothing | nov | 3 | 0,1,2 | 84.84 ± 1.15 | 92.10 ± 0.17 | 0.686 | 0.935 | 2583 |
| Amazon_clothing | naq_mix ‡ | 3 | 0,1,2 | 84.76 ± 0.37 | 92.23 ± 0.15 | 0.719 | 0.395 | 3833 |
| Amazon_clothing | mix_alt ‡ | 3 | 0,1,2 | 85.38 ± 0.34 | 92.23 ± 0.24 | 0.778 | 0.653 | 3167 |
| Amazon_clothing | sup_only ‡ | 3 | 0,1,2 | 76.59 ± 0.80 | 89.64 ± 0.52 | 1.000 | 0.000 | 833 |
| Amazon_clothing | naq_unl † | 3 | 0,1,2 | 84.58 ± 0.42 | 92.18 ± 0.09 | 0.629 | 1.000 | 3667 |

## 2. 짝지은 차이

| dataset | 비교 | shot | s0 | s1 | s2 | s3 | s4 | 평균 | sd | 음수/양수 |
|---|---|---|---|---|---|---|---|---|---|---|
| Amazon_eletronics | mix_alt ‡ − naq_feat | 1s | +1.79 | +1.56 | +1.28 | +1.09 | +0.05 | +1.15 | 0.67 | 0/5 |
| Amazon_eletronics | mix_alt ‡ − naq_feat | 5s | +0.46 | +0.58 | +0.19 | +0.43 | -0.08 | +0.32 | 0.26 | 1/4 |
| Amazon_eletronics | mix_alt ‡ − naq_mix ‡ | 1s | +1.46 | +1.66 | +0.96 | +2.32 | +0.17 | +1.31 | 0.80 | 0/5 |
| Amazon_eletronics | mix_alt ‡ − naq_mix ‡ | 5s | +0.35 | +0.46 | +0.42 | +0.98 | +0.24 | +0.49 | 0.29 | 0/5 |
| Amazon_eletronics | naq_mix ‡ − naq_feat | 1s | +0.33 | -0.10 | +0.32 | -1.23 | -0.12 | -0.16 | 0.64 | 3/2 |
| Amazon_eletronics | naq_mix ‡ − naq_feat | 5s | +0.11 | +0.12 | -0.22 | -0.56 | -0.32 | -0.17 | 0.29 | 3/2 |
| Amazon_eletronics | nov − naq_feat | 1s | +1.39 | +1.62 | +1.83 | +1.31 | +0.38 | +1.31 | 0.56 | 0/5 |
| Amazon_eletronics | nov − naq_feat | 5s | +0.80 | +0.77 | +0.81 | +0.77 | +0.31 | +0.69 | 0.21 | 0/5 |
| Amazon_eletronics | mix_alt ‡ − nov | 1s | +0.40 | -0.07 | -0.55 | -0.22 | -0.32 | -0.15 | 0.36 | 4/1 |
| Amazon_eletronics | mix_alt ‡ − nov | 5s | -0.34 | -0.18 | -0.62 | -0.34 | -0.40 | -0.38 | 0.16 | 5/0 |
| Amazon_eletronics | mix_alt ‡ − sup_only ‡ | 1s | +7.47 | +7.33 | +7.24 | +7.92 | +7.28 | +7.45 | 0.28 | 0/5 |
| Amazon_eletronics | mix_alt ‡ − sup_only ‡ | 5s | +3.74 | +4.18 | +4.04 | +4.49 | +4.22 | +4.13 | 0.27 | 0/5 |
| Amazon_eletronics | sup_only ‡ − naq_feat | 1s | -5.68 | -5.77 | -5.96 | -6.83 | -7.23 | -6.29 | 0.69 | 5/0 |
| Amazon_eletronics | sup_only ‡ − naq_feat | 5s | -3.28 | -3.59 | -3.85 | -4.06 | -4.30 | -3.82 | 0.40 | 5/0 |
| Amazon_eletronics | naq_unl † − naq_feat | 1s | +3.15 | +2.50 | +2.12 | - | - | +2.59 | 0.52 | 0/3 |
| Amazon_eletronics | naq_unl † − naq_feat | 5s | +1.34 | +1.31 | +0.97 | - | - | +1.21 | 0.20 | 0/3 |
| dblp | mix_alt ‡ − naq_diff | 1s | +3.58 | +1.71 | +2.23 | +2.41 | +3.70 | +2.73 | 0.88 | 0/5 |
| dblp | mix_alt ‡ − naq_diff | 5s | +1.49 | +0.52 | +0.52 | +0.84 | +0.63 | +0.80 | 0.41 | 0/5 |
| dblp | mix_alt ‡ − naq_mix ‡ | 1s | +0.17 | -0.86 | -1.16 | -0.03 | +0.63 | -0.25 | 0.74 | 3/2 |
| dblp | mix_alt ‡ − naq_mix ‡ | 5s | +0.33 | -0.18 | +0.07 | +0.11 | +0.03 | +0.07 | 0.18 | 1/4 |
| dblp | naq_mix ‡ − naq_diff | 1s | +3.41 | +2.57 | +3.38 | +2.44 | +3.08 | +2.98 | 0.45 | 0/5 |
| dblp | naq_mix ‡ − naq_diff | 5s | +1.16 | +0.70 | +0.45 | +0.73 | +0.60 | +0.73 | 0.26 | 0/5 |
| dblp | nov − naq_diff | 1s | -0.16 | -0.59 | +0.42 | -0.34 | -0.28 | -0.19 | 0.38 | 4/1 |
| dblp | nov − naq_diff | 5s | +0.91 | +0.15 | -0.36 | -0.13 | -0.42 | +0.03 | 0.54 | 3/2 |
| dblp | mix_alt ‡ − nov | 1s | +3.74 | +2.30 | +1.80 | +2.75 | +3.99 | +2.92 | 0.93 | 0/5 |
| dblp | mix_alt ‡ − nov | 5s | +0.58 | +0.38 | +0.88 | +0.98 | +1.05 | +0.77 | 0.28 | 0/5 |
| dblp | mix_alt ‡ − sup_only ‡ | 1s | +2.31 | +1.55 | +2.10 | +2.33 | +2.80 | +2.22 | 0.45 | 0/5 |
| dblp | mix_alt ‡ − sup_only ‡ | 5s | +2.45 | +1.15 | +1.77 | +1.92 | +1.62 | +1.78 | 0.47 | 0/5 |
| dblp | sup_only ‡ − naq_diff | 1s | +1.28 | +0.16 | +0.13 | +0.08 | +0.91 | +0.51 | 0.55 | 0/5 |
| dblp | sup_only ‡ − naq_diff | 5s | -0.96 | -0.63 | -1.25 | -1.08 | -0.98 | -0.98 | 0.23 | 5/0 |
| dblp | naq_unl † − naq_diff | 1s | +1.33 | +0.35 | -0.38 | - | - | +0.44 | 0.86 | 1/2 |
| dblp | naq_unl † − naq_diff | 5s | +1.11 | +0.25 | -0.31 | - | - | +0.35 | 0.72 | 1/2 |
| Amazon_clothing | mix_alt ‡ − naq_feat | 1s | +0.54 | +1.24 | -0.29 | - | - | +0.50 | 0.76 | 1/2 |
| Amazon_clothing | mix_alt ‡ − naq_feat | 5s | +0.13 | +0.14 | -0.33 | - | - | -0.02 | 0.27 | 1/2 |
| Amazon_clothing | mix_alt ‡ − naq_mix ‡ | 1s | +0.32 | +0.83 | +0.72 | - | - | +0.62 | 0.27 | 0/3 |
| Amazon_clothing | mix_alt ‡ − naq_mix ‡ | 5s | -0.06 | +0.41 | -0.35 | - | - | +0.00 | 0.39 | 2/1 |
| Amazon_clothing | naq_mix ‡ − naq_feat | 1s | +0.22 | +0.41 | -1.01 | - | - | -0.12 | 0.77 | 1/2 |
| Amazon_clothing | naq_mix ‡ − naq_feat | 5s | +0.19 | -0.27 | +0.02 | - | - | -0.02 | 0.23 | 1/2 |
| Amazon_clothing | nov − naq_feat | 1s | +1.05 | -0.92 | -0.27 | - | - | -0.04 | 1.00 | 2/1 |
| Amazon_clothing | nov − naq_feat | 5s | +0.22 | -0.38 | -0.28 | - | - | -0.15 | 0.32 | 2/1 |
| Amazon_clothing | mix_alt ‡ − nov | 1s | -0.51 | +2.16 | -0.02 | - | - | +0.54 | 1.42 | 2/1 |
| Amazon_clothing | mix_alt ‡ − nov | 5s | -0.09 | +0.52 | -0.05 | - | - | +0.13 | 0.34 | 2/1 |
| Amazon_clothing | mix_alt ‡ − sup_only ‡ | 1s | +7.85 | +9.41 | +9.09 | - | - | +8.79 | 0.83 | 0/3 |
| Amazon_clothing | mix_alt ‡ − sup_only ‡ | 5s | +1.98 | +3.20 | +2.59 | - | - | +2.59 | 0.61 | 0/3 |
| Amazon_clothing | sup_only ‡ − naq_feat | 1s | -7.31 | -8.17 | -9.38 | - | - | -8.29 | 1.04 | 3/0 |
| Amazon_clothing | sup_only ‡ − naq_feat | 5s | -1.85 | -3.06 | -2.92 | - | - | -2.61 | 0.66 | 3/0 |
| Amazon_clothing | naq_unl † − naq_feat | 1s | +0.07 | -0.41 | -0.59 | - | - | -0.31 | 0.34 | 2/1 |
| Amazon_clothing | naq_unl † − naq_feat | 5s | +0.14 | -0.26 | -0.10 | - | - | -0.07 | 0.20 | 2/1 |
| Amazon_clothing | naq_diff − naq_feat | 1s | -3.72 | -3.55 | -5.21 | - | - | -4.16 | 0.91 | 3/0 |
| Amazon_clothing | naq_diff − naq_feat | 5s | -1.11 | -0.91 | -1.42 | - | - | -1.15 | 0.25 | 3/0 |

회수율 (1s; 분자·분모는 각자의 공통 seed 평균 차이):

| dataset | naq_unl† − NaQ-best | R = nov 몫 / 상한 | R_mix = mix_alt 몫 / 상한 |
|---|---|---|---|
| Amazon_eletronics | +2.59 | 0.51 | 0.45 |
| dblp | +0.44 | -0.43 | 6.25 |
| Amazon_clothing | -0.31 | 0.14 | -1.64 |

## 3. 분해 (1s; 네 arm NaQ-best·naq_mix·nov·mix_alt의 공통 seed만 사용)

| dataset | seeds | NaQ→naq_mix (지도 30%) | naq_mix→mix_alt (novel 앵커) | 합 A | NaQ→nov (novel 앵커) | nov→mix_alt (지도 30%) | 합 B |
|---|---|---|---|---|---|---|---|
| Amazon_eletronics | 0,1,2,3,4 | -0.16 | +1.31 | +1.15 | +1.31 | -0.15 | +1.15 |
| dblp | 0,1,2,3,4 | +2.98 | -0.25 | +2.73 | -0.19 | +2.92 | +2.73 |
| Amazon_clothing | 0,1,2 | -0.12 | +0.62 | +0.50 | -0.04 | +0.54 | +0.50 |

## 4. 용량-반응 (1s)

pseudo 앵커 novel 비율: 지도 에피소드가 없는 arm은 그 arm의 anchor_novel_frac seed 평균. 지도 혼합 arm은 pseudo 부분이 같은 무혼합 arm의 값을 쓴다 (mix_alt → nov2_t1). Δ = arm − NaQ-best (공통 seed 평균). nov2c_t1은 T08 run(Electronics·dblp만).

| dataset | arm | pseudo 앵커 novel 비율 | Δ 1s | seeds |
|---|---|---|---|---|
| Amazon_eletronics | naq_feat | 0.433 | +0.00 | 0,1,2,3,4 |
| Amazon_eletronics | nov2c_t1 | 0.676 | +0.52 | 0,1,2 |
| Amazon_eletronics | nov | 0.878 | +1.31 | 0,1,2,3,4 |
| Amazon_eletronics | mix_alt ‡ | 0.878 | +1.15 | 0,1,2,3,4 |
| Amazon_eletronics | naq_unl † | 1.000 | +2.59 | 0,1,2 |
| Amazon_eletronics | **Spearman ρ** (5점) | | 0.975 | |
| dblp | naq_diff | 0.427 | +0.00 | 0,1,2,3,4 |
| dblp | nov2c_t1 | 0.711 | +0.86 | 0,1,2 |
| dblp | nov | 0.809 | -0.19 | 0,1,2,3,4 |
| dblp | mix_alt ‡ | 0.809 | +2.73 | 0,1,2,3,4 |
| dblp | naq_unl † | 1.000 | +0.44 | 0,1,2 |
| dblp | **Spearman ρ** (5점) | | 0.103 | |
| Amazon_clothing | naq_feat | 0.570 | +0.00 | 0,1,2 |
| Amazon_clothing | nov | 0.935 | -0.04 | 0,1,2 |
| Amazon_clothing | mix_alt ‡ | 0.935 | +0.50 | 0,1,2 |
| Amazon_clothing | naq_unl † | 1.000 | -0.31 | 0,1,2 |
| Amazon_clothing | **Spearman ρ** (4점) | | - | |

## 5. G9 계산값 (판정은 사람이 한다)

- G9-a (기준: 세 데이터셋 `mix_alt − NaQ-best` 1s 평균 ≥ +1.0; 양수 seed Electronics·dblp ≥ 4/5, Clothing 3/3): Amazon_eletronics +1.15 (양수 5/5), dblp +2.73 (양수 5/5), Amazon_clothing +0.50 (양수 2/3)
- G9-b (기준: Electronics `mix_alt − naq_mix` 1s 평균 ≥ +0.7, 양수 ≥ 4/5): +1.31 (양수 5/5)
- G9-c (기준: 세 데이터셋 `mix_alt − sup_only` 1s > 0): Amazon_eletronics +7.45, dblp +2.22, Amazon_clothing +8.79

## 6. 사전 등록 대조표 (계산값만)

| ID | 기준 | 계산값 |
|---|---|---|
| V1 | Electronics mix_alt − naq_mix ≥ +0.7, 양수 ≥ 4/5 | +1.31 (양수 5/5) |
| V2 | dblp naq_mix − NaQ-best ≥ +1.5, |mix_alt − naq_mix| ≤ 1.0 | +2.98; -0.25 |
| V3 | mix_alt − NaQ-best: Electronics ≥ +1.3, dblp ≥ +2.0 | Electronics +1.15, dblp +2.73 |
| V4 | Clothing nov2_t1 − naq_feat ≥ +1.0, 3/3 양수 | -0.04 (양수 1/3) |
| V5 | Clothing mix_alt − naq_mix ≥ +0.7 | +0.62 |
| V6 | Clothing naq_unl† − naq_feat ≥ +1.5 | -0.31 |
| V7 | Clothing sup_only − naq_feat ≤ −2.0 | -8.29 |
| V8 | Electronics Spearman ρ ≥ 0.8 (§4의 5점) | 0.975 |
| V9 | Clothing naq_feat − naq_diff (1s) > 0 | +4.16 |

## 7. T09 실행 범위 (계획 47 run)

| 우선순위 | 계획 | 결과 JSON 있음 | 없음(미실행·제외) | run 시간 합 (분) |
|---|---|---|---|---|
| P1 | 12 | 12 | - | 25.5 |
| P2 | 12 | 12 | - | 51.3 |
| P3 | 8 | 8 | - | 34.0 |
| P4 | 9 | 9 | - | 19.6 |
| P5 | 6 | 6 | - | 22.7 |
