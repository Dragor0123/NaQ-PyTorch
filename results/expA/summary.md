# Exp A summary

## (a) G1

| dataset | R* | p@10(R*) | p@10(lr) | Δ_lr | Δ_gbdt | Δ_lr(valid, same R*) | Spearman(test) | Δ_lr≥0.03 | ρ≥0.3 |
|---|---|---|---|---|---|---|---|---|---|
| Amazon_clothing | sgc | 0.718 | 0.734 | +0.015 | +0.002 | +0.013 | 0.418 | FAIL | PASS |
| Amazon_eletronics | raw | 0.631 | 0.647 | +0.016 | +0.004 | +0.025 | 0.505 | FAIL | PASS |
| dblp | sgc | 0.294 | 0.294 | +0.000 | -0.010 | +0.016 | 0.215 | FAIL | FAIL |

G1 Δ_lr 조건 통과: **0 / 3** (기준 2개 이상)

## (b) purity@10

| dataset | group | raw | sgc | ppr | lr | gbdt |
|---|---|---|---|---|---|---|
| Amazon_clothing | base | 0.556 | 0.596 | 0.518 | 0.613 | 0.622 |
| Amazon_clothing | valid | 0.522 | 0.571 | 0.495 | 0.584 | 0.568 |
| Amazon_clothing | test | 0.683 | 0.718 | 0.664 | 0.734 | 0.720 |
| Amazon_clothing | all | 0.597 | 0.637 | 0.568 | 0.653 | 0.648 |
| Amazon_eletronics | base | 0.671 | 0.540 | 0.311 | 0.691 | 0.680 |
| Amazon_eletronics | valid | 0.590 | 0.504 | 0.278 | 0.615 | 0.605 |
| Amazon_eletronics | test | 0.631 | 0.473 | 0.289 | 0.647 | 0.634 |
| Amazon_eletronics | all | 0.645 | 0.517 | 0.300 | 0.666 | 0.655 |
| dblp | base | 0.161 | 0.298 | 0.283 | 0.295 | 0.301 |
| dblp | valid | 0.147 | 0.315 | 0.326 | 0.330 | 0.317 |
| dblp | test | 0.155 | 0.294 | 0.292 | 0.294 | 0.284 |
| dblp | all | 0.157 | 0.300 | 0.293 | 0.301 | 0.300 |

## (c) class-level similarity@10 vs exact purity@10

| dataset | group | ranker | class-sim | purity@10 | NaQ reported class-sim |
|---|---|---|---|---|---|
| Amazon_clothing | all | raw | 0.816 | 0.597 | 0.816 |
| Amazon_clothing | all | ppr | 0.790 | 0.568 | 0.788 |
| Amazon_clothing | all | sgc | 0.841 | 0.637 | - |
| Amazon_clothing | all | lr | 0.851 | 0.653 | - |
| Amazon_clothing | test | raw | 0.850 | 0.683 | - |
| Amazon_clothing | test | ppr | 0.830 | 0.664 | - |
| Amazon_clothing | test | sgc | 0.875 | 0.718 | - |
| Amazon_clothing | test | lr | 0.885 | 0.734 | - |
| Amazon_eletronics | all | raw | 0.819 | 0.645 | 0.819 |
| Amazon_eletronics | all | ppr | 0.527 | 0.300 | 0.527 |
| Amazon_eletronics | all | sgc | 0.730 | 0.517 | - |
| Amazon_eletronics | all | lr | 0.835 | 0.666 | - |
| Amazon_eletronics | test | raw | 0.806 | 0.631 | - |
| Amazon_eletronics | test | ppr | 0.503 | 0.289 | - |
| Amazon_eletronics | test | sgc | 0.685 | 0.473 | - |
| Amazon_eletronics | test | lr | 0.818 | 0.647 | - |
| dblp | all | raw | 0.735 | 0.157 | 0.735 |
| dblp | all | ppr | 0.830 | 0.293 | 0.830 |
| dblp | all | sgc | 0.841 | 0.300 | - |
| dblp | all | lr | 0.836 | 0.301 | - |
| dblp | test | raw | 0.739 | 0.155 | - |
| dblp | test | ppr | 0.830 | 0.292 | - |
| dblp | test | sgc | 0.842 | 0.294 | - |
| dblp | test | lr | 0.836 | 0.294 | - |

## (d) calibration

| dataset | AUC lr base | AUC lr test | AUC gbdt base | AUC gbdt test | ECE lr base | ECE lr test | ECE gbdt base | ECE gbdt test | gap base | gap valid | gap test |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Amazon_clothing | 0.730 | 0.722 | 0.778 | 0.726 | 0.180 | 0.045 | 0.171 | 0.033 | +0.187 | +0.224 | +0.079 |
| Amazon_eletronics | 0.805 | 0.842 | 0.836 | 0.830 | 0.100 | 0.113 | 0.089 | 0.104 | +0.081 | +0.133 | +0.111 |
| dblp | 0.721 | 0.701 | 0.737 | 0.702 | 0.088 | 0.089 | 0.086 | 0.084 | +0.188 | +0.147 | +0.181 |

gap = mean(π̂) − mean(purity@10_lr). AUC/ECE over all pool pairs of the group's anchors.

## (e) selective purity@10_lr (test anchors)

| dataset | f=1.0 | f=0.7 π̂ | f=0.7 random | f=0.5 π̂ | f=0.5 random |
|---|---|---|---|---|---|
| Amazon_clothing | 0.734 | 0.804 | 0.730 | 0.845 | 0.730 |
| Amazon_eletronics | 0.647 | 0.730 | 0.648 | 0.785 | 0.649 |
| dblp | 0.294 | 0.320 | 0.294 | 0.344 | 0.299 |

## (f) LR standardized coefficients

| dataset | cos_raw | cos_sgc | log_ppr | adj | mknn_raw | neg_euclid_raw |
|---|---|---|---|---|---|---|
| Amazon_clothing | +0.418 | +0.618 | +0.151 | -0.014 | -0.027 | +0.138 |
| Amazon_eletronics | +0.955 | +0.369 | +0.018 | -0.053 | -0.011 | +0.357 |
| dblp | +0.112 | +0.149 | +1.125 | -0.269 | -0.019 | -0.001 |

## (g) tail test classes (smallest 20%) purity@10

| dataset | #classes | #anchors | raw | sgc | ppr | lr | gbdt |
|---|---|---|---|---|---|---|---|
| Amazon_clothing | 4 | 529 | 0.690 | 0.708 | 0.605 | 0.753 | 0.733 |
| Amazon_eletronics | 8 | 956 | 0.462 | 0.393 | 0.220 | 0.486 | 0.466 |
| dblp | 6 | 785 | 0.137 | 0.282 | 0.276 | 0.296 | 0.292 |
