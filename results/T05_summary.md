# T05 summary — mechanism diagnosis (A) and disentangling arms (B)

## 1. A 요약 (상세: results/T05A_diag.md)

| dataset | ρ(π̂,deg) | ρ(π̂,boundary) | ρ(π̂,class_size) | test frac 기각/전체 | gini | max | #cls>50% base/valid/test |
|---|---|---|---|---|---|---|---|
| Amazon_clothing | +0.382 | -0.126 | +0.011 | 0.337/0.376 | 0.295 | 0.658 | 6/40, 4/17, 0/20 |
| Amazon_eletronics | +0.091 | -0.054 | +0.202 | 0.258/0.241 | 0.321 | 0.956 | 16/90, 9/37, 11/40 |
| dblp | -0.229 | -0.325 | -0.288 | 0.255/0.240 | 0.493 | 0.816 | 14/80, 7/27, 5/30 |

| dataset | set | deg | boundary | class_size | tail | purity@10 lr | base | valid | test |
|---|---|---|---|---|---|---|---|---|---|
| Amazon_clothing | rejected | 2.60 | 0.458 | 487 | 0.076 | 0.483 | 0.469 | 0.193 | 0.337 |
| Amazon_clothing | kept | 9.40 | 0.345 | 509 | 0.077 | 0.725 | 0.413 | 0.195 | 0.392 |
| Amazon_eletronics | rejected | 1.79 | 0.644 | 300 | 0.128 | 0.514 | 0.520 | 0.222 | 0.258 |
| Amazon_eletronics | kept | 2.17 | 0.608 | 363 | 0.082 | 0.731 | 0.583 | 0.182 | 0.234 |
| dblp | rejected | 17.96 | 0.809 | 487 | 0.059 | 0.204 | 0.539 | 0.206 | 0.255 |
| dblp | kept | 12.55 | 0.668 | 373 | 0.099 | 0.343 | 0.592 | 0.175 | 0.233 |

## 2. B 결과 (naq_unl은 '레이블된 노드 = base' 정보를 쓰는 분석용 arm, T05 §0 예외)

| dataset | arm | seed | 1s LR | 5s LR | q_purity | anchor_uniq | anchor_ent | base/valid/test | best ep | 비고 |
|---|---|---|---|---|---|---|---|---|---|---|
| Amazon_eletronics | naq_feat | 0 | 74.58 | 87.44 | 0.646 | 0.798 | 0.971 | 0.57/0.19/0.24 | 4000 | 재사용(T03/T04) |
| Amazon_eletronics | naq_feat | 1 | 74.69 | 87.44 | 0.649 | 0.797 | 0.970 | 0.57/0.19/0.24 | 4000 | 재사용(T03/T04) |
| Amazon_eletronics | lr_only | 0 | 74.11 | 87.26 | 0.666 | 0.798 | 0.971 | 0.57/0.19/0.24 | 3750 | 재사용(T03/T04) |
| Amazon_eletronics | lr_only | 1 | 74.46 | 87.39 | 0.670 | 0.797 | 0.970 | 0.57/0.19/0.24 | 4000 | 재사용(T03/T04) |
| Amazon_eletronics | naq_unl | 0 | 77.72 | 88.78 | 0.611 | 0.609 | 0.820 | 0.00/0.45/0.55 | 4000 | †레이블 노드 정보 사용 |
| Amazon_eletronics | naq_unl | 1 | 77.19 | 88.75 | 0.615 | 0.610 | 0.821 | 0.00/0.45/0.55 | 4000 | †레이블 노드 정보 사용 |
| Amazon_eletronics | lr_soft | 0 | 74.14 | 87.22 | 0.666 | 0.798 | 0.971 | 0.57/0.19/0.24 | 3750 | w mean 1.00 sd 0.194 [0.05, 1.68] |
| Amazon_eletronics | lr_soft | 1 | 74.44 | 87.32 | 0.670 | 0.797 | 0.970 | 0.57/0.19/0.24 | 4000 | w mean 1.00 sd 0.193 [0.10, 1.76] |
| Amazon_eletronics | naq_hard | 0 | 74.86 | 87.87 | 0.604 | 0.788 | 0.972 | 0.56/0.20/0.25 | 4000 | K=4, sel loss 0.805 / cand 0.498, sel purity 0.604 / cand 0.646 |
| Amazon_eletronics | naq_easy | 0 | 74.81 | 87.72 | 0.685 | 0.773 | 0.965 | 0.58/0.19/0.23 | 4000 | K=4, sel loss 0.146 / cand 0.998, sel purity 0.685 / cand 0.646 |
| dblp | naq_diff | 0 | 73.28 | 82.35 | 0.292 | 0.792 | 0.967 | 0.57/0.19/0.24 | 750 | 재사용(T03/T04) |
| dblp | naq_diff | 1 | 74.15 | 82.90 | 0.294 | 0.789 | 0.967 | 0.57/0.19/0.24 | 3750 | 재사용(T03/T04) |
| dblp | lr_only | 0 | 73.77 | 82.40 | 0.298 | 0.792 | 0.967 | 0.57/0.19/0.24 | 2000 | 재사용(T03/T04) |
| dblp | lr_only | 1 | 73.30 | 82.53 | 0.303 | 0.789 | 0.967 | 0.57/0.19/0.24 | 3500 | 재사용(T03/T04) |
| dblp | naq_unl | 0 | 74.61 | 83.46 | 0.309 | 0.592 | 0.789 | 0.00/0.43/0.57 | 2250 | †레이블 노드 정보 사용 |
| dblp | naq_unl | 1 | 74.50 | 83.15 | 0.308 | 0.593 | 0.787 | 0.00/0.44/0.56 | 2750 | †레이블 노드 정보 사용 |
| dblp | lr_soft | 0 | 73.74 | 82.60 | 0.298 | 0.792 | 0.967 | 0.57/0.19/0.24 | 2000 | w mean 1.00 sd 0.150 [0.03, 1.91] |
| dblp | lr_soft | 1 | 73.31 | 82.34 | 0.303 | 0.789 | 0.967 | 0.57/0.19/0.24 | 3500 | w mean 1.00 sd 0.150 [0.02, 1.79] |
| dblp | naq_hard | 0 | 72.81 | 82.89 | 0.266 | 0.782 | 0.957 | 0.58/0.18/0.24 | 1500 | K=4, sel loss 0.284 / cand 0.154, sel purity 0.266 / cand 0.292 |
| dblp | naq_easy | 0 | 74.19 | 82.67 | 0.320 | 0.776 | 0.972 | 0.57/0.19/0.24 | 1500 | K=4, sel loss 0.049 / cand 0.371, sel purity 0.320 / cand 0.292 |

## 3. 짝지은 차이 (같은 seed; |평균| < 1.0%p(eletronics) / 1.5%p(dblp) → '잡음 범위 가능')

| dataset | 비교 | shot | s0 | s1 | 평균 | 비고 |
|---|---|---|---|---|---|---|
| Amazon_eletronics | naq_unl − naq_feat | 1s | +3.15 | +2.50 | +2.83 |  |
| Amazon_eletronics | naq_unl − naq_feat | 5s | +1.34 | +1.31 | +1.32 |  |
| Amazon_eletronics | lr_soft − lr_only | 1s | +0.03 | -0.02 | +0.01 | 잡음 범위 가능 |
| Amazon_eletronics | lr_soft − lr_only | 5s | -0.04 | -0.08 | -0.06 | 잡음 범위 가능 |
| Amazon_eletronics | naq_hard − naq_feat | 1s | +0.28 | - | +0.28 | 잡음 범위 가능 |
| Amazon_eletronics | naq_hard − naq_feat | 5s | +0.44 | - | +0.44 | 잡음 범위 가능 |
| Amazon_eletronics | naq_easy − naq_feat | 1s | +0.24 | - | +0.24 | 잡음 범위 가능 |
| Amazon_eletronics | naq_easy − naq_feat | 5s | +0.29 | - | +0.29 | 잡음 범위 가능 |
| Amazon_eletronics | naq_hard − naq_easy | 1s | +0.05 | - | +0.05 | 잡음 범위 가능 |
| Amazon_eletronics | naq_hard − naq_easy | 5s | +0.15 | - | +0.15 | 잡음 범위 가능 |
| dblp | naq_unl − naq_diff | 1s | +1.33 | +0.35 | +0.84 | 잡음 범위 가능 |
| dblp | naq_unl − naq_diff | 5s | +1.11 | +0.25 | +0.68 | 잡음 범위 가능 |
| dblp | lr_soft − lr_only | 1s | -0.03 | +0.01 | -0.01 | 잡음 범위 가능 |
| dblp | lr_soft − lr_only | 5s | +0.20 | -0.19 | +0.01 | 잡음 범위 가능 |
| dblp | naq_hard − naq_diff | 1s | -0.46 | - | -0.46 | 잡음 범위 가능 |
| dblp | naq_hard − naq_diff | 5s | +0.54 | - | +0.54 | 잡음 범위 가능 |
| dblp | naq_easy − naq_diff | 1s | +0.91 | - | +0.91 | 잡음 범위 가능 |
| dblp | naq_easy − naq_diff | 5s | +0.32 | - | +0.32 | 잡음 범위 가능 |
| dblp | naq_hard − naq_easy | 1s | -1.37 | - | -1.37 | 잡음 범위 가능 |
| dblp | naq_hard − naq_easy | 5s | +0.22 | - | +0.22 | 잡음 범위 가능 |

## 4. 사전 등록 대조표 (계산값만; 적중 판정은 사람이 한다)

| ID | 기준 | 계산값 |
|---|---|---|
| Q1 | 세 데이터셋 모두 ρ(π̂, boundary) ≤ −0.3 | Amazon_clothing -0.126, Amazon_eletronics -0.054, dblp -0.325 |
| Q2 | 기각 30%의 test 비율 > 전체 test 비율 (3개 중 ≥2) | Amazon_clothing 0.337 vs 0.376, Amazon_eletronics 0.258 vs 0.241, dblp 0.255 vs 0.240 |
| Q3 | 기각 비율 > 50% 클래스가 데이터셋마다 ≥1 | Amazon_clothing 10, Amazon_eletronics 36, dblp 26 |
| Q4 | eletronics naq_unl − NaQ-best ≥ +0.5 (seed 평균, 1s) | +2.83 |
| Q5 | 두 데이터셋 모두 abs(lr_soft − lr_only) ≤ 0.7 (1s, seed 평균) | Amazon_eletronics +0.01, dblp -0.01 |
| Q6 | eletronics naq_hard − naq_easy ≥ +1.0 (1s) | +0.05 |
| Q7 | 두 데이터셋 모두 naq_easy − NaQ-best < 0 (1s) | Amazon_eletronics +0.24, dblp +0.91 |
