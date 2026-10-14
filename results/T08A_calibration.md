# T08 A′ — calibrated PU variants (r = 0.5, labels for analysis only)

**E2\* = `pu_both2`** (no variant with calib ≤ 0.15 on both → min mean calib error; 선택은 eletronics·dblp 기준)

| dataset | variant | AUROC | mean p_novel (U) | 실제 novel 비율 | 보정 오차 (부호) | 적합 시간 s | 기대 앵커 novel 비율 τ=1 |
|---|---|---|---|---|---|---|---|
| Amazon_eletronics | pu_feat2 | 0.831 | 0.348 | 0.607 | -0.260 | 1.0 | 0.845 |
| Amazon_eletronics | pu_both2 | 0.730 | 0.763 | 0.607 | +0.156 | 0.4 | 0.678 |
| Amazon_eletronics | pu_struct | 0.753 | 0.674 | 0.607 | +0.067 | 0.1 | 0.715 |
| Amazon_eletronics | (T07) pu_feat | 0.867 | | 0.607 | abs 0.263 | | |
| Amazon_eletronics | (T07) pu_struct | 0.753 | | 0.607 | abs 0.067 | | |
| Amazon_eletronics | (T07) pu_both | 0.706 | | 0.607 | abs 0.191 | | |
| Amazon_eletronics | (PCA 적합 시간) | | | | | 12.6 | |
| dblp | pu_feat2 | 0.724 | 0.220 | 0.595 | -0.375 | 1.2 | 0.781 |
| dblp | pu_both2 | 0.731 | 0.526 | 0.595 | -0.069 | 0.7 | 0.710 |
| dblp | pu_struct | 0.764 | 0.364 | 0.595 | -0.232 | 0.1 | 0.762 |
| dblp | (T07) pu_feat | 0.749 | | 0.595 | abs 0.396 | | |
| dblp | (T07) pu_struct | 0.764 | | 0.595 | abs 0.232 | | |
| dblp | (T07) pu_both | 0.677 | | 0.595 | abs 0.067 | | |
| dblp | (PCA 적합 시간) | | | | | 11.1 | |
| Amazon_clothing | pu_feat2 | 0.895 | 0.533 | 0.726 | -0.193 | 1.3 | 0.912 |
| Amazon_clothing | pu_both2 | 0.836 | 0.796 | 0.726 | +0.070 | 0.9 | 0.816 |
| Amazon_clothing | pu_struct | 0.870 | 0.667 | 0.726 | -0.060 | 0.1 | 0.869 |
| Amazon_clothing | (T07) pu_feat | 0.909 | | 0.726 | abs 0.241 | | |
| Amazon_clothing | (T07) pu_struct | 0.870 | | 0.726 | abs 0.060 | | |
| Amazon_clothing | (T07) pu_both | 0.780 | | 0.726 | abs 0.163 | | |
| Amazon_clothing | (PCA 적합 시간) | | | | | 9.2 | |

평균 (eletronics, dblp): pu_feat2 AUROC 0.778 / 보정 오차 0.317, pu_both2 AUROC 0.730 / 보정 오차 0.113, pu_struct AUROC 0.759 / 보정 오차 0.149
