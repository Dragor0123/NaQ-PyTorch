# T07 A — novelty estimator comparison (labels for analysis only)

**E\* = `pu_feat`** (r=0.5 세 데이터셋 AUROC 평균 최고; 사전 명시 규칙)

## AUROC (U 안: novel vs 숨겨진 base)

| dataset | r | cent_sgc | cent_raw | msp | pu_feat | pu_struct | pu_both | best |
|---|---|---|---|---|---|---|---|---|
| Amazon_clothing | 0.5 | 0.707 | 0.720 | 0.763 | 0.909 | 0.870 | 0.780 | pu_feat |
| Amazon_clothing | 0.2 | 0.706 | 0.716 | 0.758 | 0.886 | 0.804 | 0.700 | pu_feat |
| Amazon_eletronics | 0.5 | 0.713 | 0.784 | 0.759 | 0.867 | 0.753 | 0.706 | pu_feat |
| Amazon_eletronics | 0.2 | 0.712 | 0.780 | 0.743 | 0.837 | 0.683 | 0.645 | pu_feat |
| dblp | 0.5 | 0.570 | 0.567 | 0.610 | 0.749 | 0.764 | 0.677 | pu_struct |
| dblp | 0.2 | 0.568 | 0.563 | 0.613 | 0.711 | 0.718 | 0.617 | pu_struct |
| mean (r=0.5) | | 0.663 | 0.690 | 0.711 | 0.841 | 0.796 | 0.721 | |

## PU 보정: mean p(novel) over U vs 실제 novel 비율 (절대 오차)

| dataset | r | true | pu_feat | pu_struct | pu_both |
|---|---|---|---|---|---|
| Amazon_clothing | 0.5 | 0.726 | 0.485 (0.241) | 0.667 (0.060) | 0.889 (0.163) |
| Amazon_clothing | 0.2 | 0.624 | 0.410 (0.214) | 0.732 (0.108) | 0.934 (0.310) |
| Amazon_eletronics | 0.5 | 0.607 | 0.344 (0.263) | 0.674 (0.067) | 0.798 (0.191) |
| Amazon_eletronics | 0.2 | 0.491 | 0.283 (0.208) | 0.782 (0.291) | 0.910 (0.419) |
| dblp | 0.5 | 0.595 | 0.199 (0.396) | 0.364 (0.232) | 0.529 (0.067) |
| dblp | 0.2 | 0.479 | 0.161 (0.318) | 0.470 (0.009) | 0.688 (0.209) |

## 기대 앵커 novel 비율 (w = score^τ over U; E\* = pu_feat)

| dataset | r | U 균등 | T06 하드 컷 | E* τ=1 | E* τ=2 | pu_both τ=1 | cent_sgc τ=1 |
|---|---|---|---|---|---|---|---|
| Amazon_clothing | 0.5 | 0.726 | 0.836 | 0.933 | 0.958 | 0.780 | 0.792 |
| Amazon_clothing | 0.2 | 0.624 | 0.751 | 0.888 | 0.925 | 0.654 | 0.704 |
| Amazon_eletronics | 0.5 | 0.607 | 0.758 | 0.877 | 0.924 | 0.667 | 0.683 |
| Amazon_eletronics | 0.2 | 0.491 | 0.648 | 0.802 | 0.867 | 0.515 | 0.573 |
| dblp | 0.5 | 0.595 | 0.644 | 0.811 | 0.871 | 0.699 | 0.629 |
| dblp | 0.2 | 0.479 | 0.523 | 0.710 | 0.786 | 0.528 | 0.513 |
