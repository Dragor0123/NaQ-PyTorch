# T08 summary — mixing controls and PU calibration (r = 0.5)

‡ = L의 레이블로 만든 지도 에피소드를 포함하는 arm (NaQ보다 정보량 많음).

## 1. A′ (상세: results/T08A_calibration.md). E2* = `pu_both2` — no variant with calib ≤ 0.15 on both → min mean calib error

| dataset | variant | AUROC | 보정 오차 (부호) | 적합 시간 s |
|---|---|---|---|---|
| Amazon_eletronics | pu_feat2 | 0.831 | -0.260 | 1.0 |
| Amazon_eletronics | pu_both2 | 0.730 | +0.156 | 0.4 |
| Amazon_eletronics | pu_struct | 0.753 | +0.067 | 0.1 |
| dblp | pu_feat2 | 0.724 | -0.375 | 1.2 |
| dblp | pu_both2 | 0.731 | -0.069 | 0.7 |
| dblp | pu_struct | 0.764 | -0.232 | 0.1 |
| Amazon_clothing | pu_feat2 | 0.895 | -0.193 | 1.3 |
| Amazon_clothing | pu_both2 | 0.836 | +0.070 | 0.9 |
| Amazon_clothing | pu_struct | 0.870 | -0.060 | 0.1 |

## 2. 결과

| dataset | arm | seed | 1s LR | 5s LR | q_purity | anchor_novel_frac | anchor_ent | best ep |
|---|---|---|---|---|---|---|---|---|
| Amazon_eletronics | naq_feat | 0 | 74.58 | 87.44 | 0.646 | 0.432 | 0.971 | 4000 |
| Amazon_eletronics | naq_feat | 1 | 74.69 | 87.44 | 0.649 | 0.434 | 0.970 | 4000 |
| Amazon_eletronics | naq_feat | 2 | 74.50 | 87.40 | 0.647 | 0.431 | 0.968 | 3000 |
| Amazon_eletronics | naq_mix_r0.5 ‡ | 0 | 74.91 | 87.55 | 0.752 | 0.298 | 0.974 | 3500 |
| Amazon_eletronics | naq_mix_r0.5 ‡ | 1 | 74.58 | 87.56 | 0.753 | 0.299 | 0.973 | 4000 |
| Amazon_eletronics | sup_only_r0.5 ‡ | 0 | 68.90 | 84.15 | 1.000 | 0.000 | 0.879 | 3250 |
| Amazon_eletronics | sup_only_r0.5 ‡ | 1 | 68.91 | 83.84 | 1.000 | 0.000 | 0.879 | 3750 |
| Amazon_eletronics | mix_alt_r0.5 ‡ | 0 | 76.36 | 87.89 | 0.767 | 0.621 | 0.954 | 4000 |
| Amazon_eletronics | mix_alt_r0.5 ‡ | 1 | 76.24 | 88.02 | 0.772 | 0.603 | 0.956 | 4000 |
| Amazon_eletronics | mix_alt_p10_r0.5 ‡ | 0 | 76.56 | 88.38 | 0.702 | 0.784 | 0.910 | 3750 |
| Amazon_eletronics | mix_alt_p50_r0.5 ‡ | 0 | 75.25 | 87.58 | 0.837 | 0.436 | 0.974 | 4000 |
| Amazon_eletronics | nov2_t1_r0.5 | 0 | 75.96 | 88.23 | 0.665 | 0.876 | 0.869 | 3000 |
| Amazon_eletronics | nov2_t1_r0.5 | 1 | 76.31 | 88.20 | 0.671 | 0.876 | 0.869 | 4000 |
| Amazon_eletronics | nov2_t1_r0.5 | 2 | 76.33 | 88.22 | 0.669 | 0.879 | 0.868 | 3500 |
| Amazon_eletronics | nov2c_t1_r0.5 | 0 | 74.69 | 87.42 | 0.636 | 0.669 | 0.951 | 3500 |
| Amazon_eletronics | nov2c_t1_r0.5 | 1 | 75.27 | 87.82 | 0.634 | 0.673 | 0.951 | 3250 |
| Amazon_eletronics | nov2c_t1_r0.5 | 2 | 75.35 | 87.98 | 0.632 | 0.684 | 0.949 | 4000 |
| dblp | naq_diff | 0 | 73.28 | 82.35 | 0.292 | 0.427 | 0.967 | 750 |
| dblp | naq_diff | 1 | 74.15 | 82.90 | 0.294 | 0.425 | 0.967 | 3750 |
| dblp | naq_diff | 2 | 73.75 | 83.13 | 0.292 | 0.430 | 0.967 | 4000 |
| dblp | naq_mix_r0.5 ‡ | 0 | 76.69 | 83.50 | 0.512 | 0.294 | 0.977 | 3750 |
| dblp | naq_mix_r0.5 ‡ | 1 | 76.72 | 83.60 | 0.506 | 0.296 | 0.976 | 3250 |
| dblp | sup_only_r0.5 ‡ | 0 | 74.55 | 81.39 | 1.000 | 0.000 | 0.890 | 2250 |
| dblp | sup_only_r0.5 ‡ | 1 | 74.31 | 82.27 | 1.000 | 0.000 | 0.890 | 3250 |
| dblp | mix_alt_r0.5 ‡ | 0 | 76.86 | 83.84 | 0.570 | 0.569 | 0.957 | 4000 |
| dblp | mix_alt_r0.5 ‡ | 1 | 75.86 | 83.42 | 0.576 | 0.565 | 0.958 | 3750 |
| dblp | mix_alt_p10_r0.5 ‡ | 0 | 75.68 | 83.47 | 0.442 | 0.729 | 0.914 | 2250 |
| dblp | mix_alt_p50_r0.5 ‡ | 0 | 76.88 | 83.33 | 0.688 | 0.409 | 0.977 | 4000 |
| dblp | nov2_t1_r0.5 | 0 | 73.12 | 83.25 | 0.381 | 0.807 | 0.877 | 3750 |
| dblp | nov2_t1_r0.5 | 1 | 73.56 | 83.05 | 0.381 | 0.812 | 0.875 | 2000 |
| dblp | nov2_t1_r0.5 | 2 | 74.17 | 82.78 | 0.381 | 0.808 | 0.877 | 750 |
| dblp | nov2c_t1_r0.5 | 0 | 74.46 | 83.45 | 0.325 | 0.711 | 0.931 | 3250 |
| dblp | nov2c_t1_r0.5 | 1 | 74.48 | 82.84 | 0.328 | 0.710 | 0.930 | 1500 |
| dblp | nov2c_t1_r0.5 | 2 | 74.80 | 82.95 | 0.326 | 0.712 | 0.929 | 750 |

## 3. 짝지은 차이 (같은 seed)

| dataset | 비교 | shot | s0 | s1 | s2 | 평균 | sd | 음수/양수 |
|---|---|---|---|---|---|---|---|---|
| Amazon_eletronics | naq_mix_r0.5 − naq_feat | 1s | +0.33 | -0.10 | - | +0.11 | 0.30 | 1/1 |
| Amazon_eletronics | naq_mix_r0.5 − naq_feat | 5s | +0.11 | +0.12 | - | +0.12 | 0.01 | 0/2 |
| Amazon_eletronics | mix_alt_r0.5 − naq_mix_r0.5 | 1s | +1.46 | +1.66 | - | +1.56 | 0.14 | 0/2 |
| Amazon_eletronics | mix_alt_r0.5 − naq_mix_r0.5 | 5s | +0.35 | +0.46 | - | +0.41 | 0.08 | 0/2 |
| Amazon_eletronics | naq_mix_r0.5 − nov2_t1_r0.5 | 1s | -1.06 | -1.72 | - | -1.39 | 0.47 | 2/0 |
| Amazon_eletronics | naq_mix_r0.5 − nov2_t1_r0.5 | 5s | -0.69 | -0.64 | - | -0.66 | 0.03 | 2/0 |
| Amazon_eletronics | sup_only_r0.5 − naq_feat | 1s | -5.68 | -5.77 | - | -5.73 | 0.07 | 2/0 |
| Amazon_eletronics | sup_only_r0.5 − naq_feat | 5s | -3.28 | -3.59 | - | -3.44 | 0.22 | 2/0 |
| Amazon_eletronics | mix_alt_r0.5 − sup_only_r0.5 | 1s | +7.47 | +7.33 | - | +7.40 | 0.10 | 0/2 |
| Amazon_eletronics | mix_alt_r0.5 − sup_only_r0.5 | 5s | +3.74 | +4.18 | - | +3.96 | 0.30 | 0/2 |
| Amazon_eletronics | mix_alt_p10_r0.5 − naq_feat | 1s | +1.98 | - | - | +1.98 | nan | 0/1 |
| Amazon_eletronics | mix_alt_p10_r0.5 − naq_feat | 5s | +0.94 | - | - | +0.94 | nan | 0/1 |
| Amazon_eletronics | mix_alt_r0.5 − naq_feat | 1s | +1.79 | +1.56 | - | +1.67 | 0.17 | 0/2 |
| Amazon_eletronics | mix_alt_r0.5 − naq_feat | 5s | +0.46 | +0.58 | - | +0.52 | 0.08 | 0/2 |
| Amazon_eletronics | mix_alt_p50_r0.5 − naq_feat | 1s | +0.67 | - | - | +0.67 | nan | 0/1 |
| Amazon_eletronics | mix_alt_p50_r0.5 − naq_feat | 5s | +0.14 | - | - | +0.14 | nan | 0/1 |
| Amazon_eletronics | nov2c_t1_r0.5 − naq_feat | 1s | +0.12 | +0.59 | +0.85 | +0.52 | 0.37 | 0/3 |
| Amazon_eletronics | nov2c_t1_r0.5 − naq_feat | 5s | -0.01 | +0.38 | +0.58 | +0.31 | 0.30 | 1/2 |
| Amazon_eletronics | nov2c_t1_r0.5 − nov2_t1_r0.5 | 1s | -1.27 | -1.04 | -0.99 | -1.10 | 0.15 | 3/0 |
| Amazon_eletronics | nov2c_t1_r0.5 − nov2_t1_r0.5 | 5s | -0.81 | -0.39 | -0.24 | -0.48 | 0.29 | 3/0 |
| dblp | naq_mix_r0.5 − naq_diff | 1s | +3.41 | +2.57 | - | +2.99 | 0.59 | 0/2 |
| dblp | naq_mix_r0.5 − naq_diff | 5s | +1.16 | +0.70 | - | +0.93 | 0.33 | 0/2 |
| dblp | mix_alt_r0.5 − naq_mix_r0.5 | 1s | +0.17 | -0.86 | - | -0.35 | 0.73 | 1/1 |
| dblp | mix_alt_r0.5 − naq_mix_r0.5 | 5s | +0.33 | -0.18 | - | +0.08 | 0.36 | 1/1 |
| dblp | naq_mix_r0.5 − nov2_t1_r0.5 | 1s | +3.57 | +3.16 | - | +3.37 | 0.29 | 0/2 |
| dblp | naq_mix_r0.5 − nov2_t1_r0.5 | 5s | +0.25 | +0.55 | - | +0.40 | 0.21 | 0/2 |
| dblp | sup_only_r0.5 − naq_diff | 1s | +1.28 | +0.16 | - | +0.72 | 0.79 | 0/2 |
| dblp | sup_only_r0.5 − naq_diff | 5s | -0.96 | -0.63 | - | -0.80 | 0.24 | 2/0 |
| dblp | mix_alt_r0.5 − sup_only_r0.5 | 1s | +2.31 | +1.55 | - | +1.93 | 0.53 | 0/2 |
| dblp | mix_alt_r0.5 − sup_only_r0.5 | 5s | +2.45 | +1.15 | - | +1.80 | 0.92 | 0/2 |
| dblp | mix_alt_p10_r0.5 − naq_diff | 1s | +2.40 | - | - | +2.40 | nan | 0/1 |
| dblp | mix_alt_p10_r0.5 − naq_diff | 5s | +1.12 | - | - | +1.12 | nan | 0/1 |
| dblp | mix_alt_r0.5 − naq_diff | 1s | +3.58 | +1.71 | - | +2.65 | 1.33 | 0/2 |
| dblp | mix_alt_r0.5 − naq_diff | 5s | +1.49 | +0.52 | - | +1.00 | 0.68 | 0/2 |
| dblp | mix_alt_p50_r0.5 − naq_diff | 1s | +3.60 | - | - | +3.60 | nan | 0/1 |
| dblp | mix_alt_p50_r0.5 − naq_diff | 5s | +0.98 | - | - | +0.98 | nan | 0/1 |
| dblp | nov2c_t1_r0.5 − naq_diff | 1s | +1.18 | +0.33 | +1.05 | +0.86 | 0.46 | 0/3 |
| dblp | nov2c_t1_r0.5 − naq_diff | 5s | +1.10 | -0.05 | -0.18 | +0.29 | 0.71 | 2/1 |
| dblp | nov2c_t1_r0.5 − nov2_t1_r0.5 | 1s | +1.34 | +0.92 | +0.63 | +0.96 | 0.36 | 0/3 |
| dblp | nov2c_t1_r0.5 − nov2_t1_r0.5 | 5s | +0.20 | -0.20 | +0.18 | +0.06 | 0.23 | 1/2 |

## 4. 분해 (1s, 각 단계는 해당 두 arm의 공통 seed에 대한 짝지은 차이의 평균)

| dataset | 경로 A: NaQ→+지도30%(naq_mix) | →+novel 앵커(mix_alt) | 합 A | 경로 B: NaQ→+novel 앵커(nov2_t1) | →+지도30%(mix_alt) | 합 B | 합 A − 합 B | 직접 mix_alt − NaQ |
|---|---|---|---|---|---|---|---|---|
| Amazon_eletronics | +0.11 | +1.56 | +1.67 | +1.62 (s0–2) | +0.17 (s0–1) | +1.78 | -0.11 | +1.67 |
| dblp | +2.99 | -0.35 | +2.65 | -0.11 (s0–2) | +3.02 (s0–1) | +2.91 | -0.27 | +2.65 |

같은 seed 집합이면 두 경로 합은 대수적으로 같다 (= mix_alt − NaQ). 차이는 nov2_t1 단계가 seed 0–2를 쓰고 나머지가 seed 0–1을 쓰기 때문에 생긴다.

## 5. 메커니즘 NMI (seed 0, best checkpoint 임베딩; T07 행 재계산 포함)

| dataset | run | NMI_test | NMI_base | ΔNMI_test vs NaQ | ΔNMI_base vs NaQ |
|---|---|---|---|---|---|
| Amazon_eletronics | NaQ (P3, T07) | 0.6161 | 0.6584 | +0.0000 | +0.0000 |
| Amazon_eletronics | nov2_t1 (T07) | 0.6328 | 0.6208 | +0.0167 | -0.0375 |
| Amazon_eletronics | mix_alt ‡ (T07) | 0.6268 | 0.6864 | +0.0107 | +0.0280 |
| Amazon_eletronics | naq_mix ‡ | 0.6189 | 0.6886 | +0.0028 | +0.0302 |
| Amazon_eletronics | sup_only ‡ | 0.5374 | 0.7218 | -0.0788 | +0.0634 |
| Amazon_eletronics | nov2c_t1 | 0.6274 | 0.6451 | +0.0113 | -0.0132 |
| dblp | NaQ (P3, T07) | 0.5589 | 0.5396 | +0.0000 | +0.0000 |
| dblp | nov2_t1 (T07) | 0.5729 | 0.5420 | +0.0141 | +0.0025 |
| dblp | mix_alt ‡ (T07) | 0.5896 | 0.5845 | +0.0307 | +0.0450 |
| dblp | naq_mix ‡ | 0.5881 | 0.5832 | +0.0293 | +0.0436 |
| dblp | sup_only ‡ | 0.5591 | 0.5939 | +0.0003 | +0.0543 |
| dblp | nov2c_t1 | 0.5777 | 0.5476 | +0.0189 | +0.0081 |

## 6. G8 계산값 (판정은 사람이 한다)

- G8-a: eletronics mix_alt − naq_mix (1s) = 평균 +1.56; seed별 s0 +1.46, s1 +1.66 (기준 ≥ +0.7, 2 seed 모두 양수)
- G8-b: dblp mix_alt − sup_only (1s) = +1.93 (기준 ≥ +0.7)
- G8-c: E2* (pu_both2) 보정 오차 = Amazon_eletronics +0.156, dblp -0.069 (기준: 절댓값 ≤ 0.15, 두 데이터셋)

## 7. 사전 등록 대조표 (계산값만)

| ID | 기준 | 계산값 |
|---|---|---|
| U1 | pu_feat2 보정 오차 ≤ 0.15 (둘 다), AUROC 하락 ≤ 0.03 (T07 pu_feat 대비) | Amazon_eletronics 오차 -0.260, AUROC 0.831 vs 0.867 (-0.035), dblp 오차 -0.375, AUROC 0.724 vs 0.749 (-0.024) |
| U2 | elec nov2c_t1 − NaQ ≥ +1.3 | +0.52 |
| U3 | elec mix_alt − naq_mix ≥ +0.7, 2 seed 모두 양수 | +1.56; s0 +1.46, s1 +1.66 |
| U4 | dblp abs(mix_alt − naq_mix) ≤ 1.0 | -0.35 |
| U5 | elec sup_only − NaQ ≤ −2.0; dblp sup_only − NaQ ≤ +1.0 | elec -5.73, dblp +0.72 |
| U6 | dblp p50 > p30 > p10 (s0); elec 범위 ≤ 1.0 | dblp p10/p30/p50 = 75.68 / 76.86 / 76.88; elec = 76.56 / 76.36 / 75.25 (범위 1.31) |
| U7 | 두 경로 총합 차이 ≤ 0.7 (elec) | 합 A +1.67, 합 B +1.78, 차이 -0.11 |
