# TP01 summary

## 1. Gates

- Ga: all inputs present (data/*, cache/{ds}/pool.npz, tasks.npz, split.json, save/downstream_tasks 5way_1/5shot.pkl).
- Gb: 1 max|Δ|=9.54e-07, 2 max|Δ|=9.54e-07, 3 max|Δ|=0.00e+00 finite=True, 4 |Δ|=2.38e-07 → all_pass=True
- pprrow cache Amazon_clothing: verify max_abs_err=4.17e-07, random>50th=0, top-50 score≤0 frac=0.0794, build 45s
- pprrow cache Amazon_eletronics: verify max_abs_err=3.58e-07, random>50th=0, top-50 score≤0 frac=0.4264, build 85s
- pprrow cache dblp: verify max_abs_err=2.98e-07, random>50th=0, top-50 score≤0 frac=0.0005, build 92s
- G0 (tp_l0 s0 − expB naq_feat s0): eletronics Δ1s=-0.25 Δ5s=+0.10, dblp Δ1s=+0.03 Δ5s=+0.06 → pass=True

## 2. A — token diagnostics

| dataset | ranker | m | token purity | query purity | shortfall | empty | mean #tok | hop1 | hop2 | hop≥3 | unreach | hop med | pprrow–ppr ov@8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Amazon_clothing | sgc | 4 | 0.599 | 0.600 | 0.000 | 0.000 | 4.00 |  | | | |  |  |
| Amazon_clothing | sgc | 8 | 0.584 | 0.600 | 0.000 | 0.000 | 8.00 | 0.248 | 0.584 | 0.093 | 0.075 | 2.0 |  |
| Amazon_clothing | pprrow | 4 | 0.549 | 0.600 | 0.000 | 0.000 | 4.00 |  | | | |  |  |
| Amazon_clothing | pprrow | 8 | 0.531 | 0.600 | 0.000 | 0.000 | 8.00 | 0.371 | 0.480 | 0.079 | 0.070 | 2.0 | 0.831 |
| Amazon_clothing | NaQ query | 10 | | 0.600 | | | | 0.136 | 0.351 | 0.401 | 0.112 | 2.0 | |
| Amazon_eletronics | sgc | 4 | 0.447 | 0.647 | 0.000 | 0.000 | 4.00 |  | | | |  |  |
| Amazon_eletronics | sgc | 8 | 0.448 | 0.647 | 0.000 | 0.000 | 8.00 | 0.142 | 0.414 | 0.095 | 0.349 | 2.0 |  |
| Amazon_eletronics | pprrow | 4 | 0.278 | 0.647 | 0.000 | 0.000 | 4.00 |  | | | |  |  |
| Amazon_eletronics | pprrow | 8 | 0.259 | 0.647 | 0.000 | 0.000 | 8.00 | 0.160 | 0.461 | 0.120 | 0.260 | 2.0 | 0.825 |
| Amazon_eletronics | NaQ query | 10 | | 0.647 | | | | 0.037 | 0.148 | 0.177 | 0.638 | 2.0 | |
| dblp | sgc | 4 | 0.318 | 0.157 | 0.000 | 0.000 | 4.00 |  | | | |  |  |
| dblp | sgc | 8 | 0.300 | 0.157 | 0.000 | 0.000 | 8.00 | 0.536 | 0.433 | 0.030 | 0.000 | 1.0 |  |
| dblp | pprrow | 4 | 0.315 | 0.157 | 0.000 | 0.000 | 4.00 |  | | | |  |  |
| dblp | pprrow | 8 | 0.296 | 0.157 | 0.000 | 0.000 | 8.00 | 0.845 | 0.154 | 0.001 | 0.000 | 1.0 | 0.893 |
| dblp | NaQ query | 10 | | 0.157 | | | | 0.109 | 0.223 | 0.668 | 0.000 | 3.0 | |

## 3. B1 — seed 0

| dataset | arm | best_valid | 1s LR ± CI | 5s LR ± CI | 1s proto | best ep | shortfall | attn H/logT | mass same-label / uniform |
|---|---|---|---|---|---|---|---|---|---|
| clothing | tp_l0 | 73.00 | 84.78 ± 0.98 | 92.14 ± 0.68 | 80.40 | 3750 | — | — | — / — |
| clothing | tp_a2_sgc_l25_m4 | 72.00 | 85.25 ± 1.00 | 92.07 ± 0.69 | 78.98 | 1750 | 0.000 | 0.972 | 0.605 / 0.599 |
| clothing | tp_a2_sgc_l25_m8 | 72.25 | 85.18 ± 1.00 | 92.12 ± 0.67 | 78.83 | 1750 | 0.000 | 0.971 | 0.589 / 0.583 |
| clothing | tp_a2_sgc_l50_m4 | 71.95 | 84.84 ± 1.01 | 92.14 ± 0.67 | 78.62 | 2000 | 0.000 | 0.973 | 0.605 / 0.599 |
| clothing | tp_a2_sgc_l50_m8 | 71.90 | 84.37 ± 1.03 | 92.08 ± 0.68 | 78.28 | 2250 | 0.000 | 0.968 | 0.589 / 0.583 |
| clothing | tp_b_sgc_l50_m8 | 71.30 | 84.45 ± 1.02 | 91.95 ± 0.67 | 78.35 | 1750 | 0.000 | 0.970 | 0.589 / 0.583 |
| clothing | tp_a2_pprrow_l50_m8 | 72.20 | 83.88 ± 1.05 | 92.07 ± 0.70 | 77.70 | 2250 | 0.000 | 0.935 | 0.556 / 0.527 |
| eletronics | tp_l0 | 76.20 | 74.33 ± 1.14 | 87.54 ± 0.76 | 70.46 | 4000 | — | — | — / — |
| eletronics | tp_a2_sgc_l25_m4 | 75.00 | 74.35 ± 1.10 | 87.82 ± 0.74 | 69.20 | 3750 | 0.000 | 0.949 | 0.476 / 0.465 |
| eletronics | tp_a2_sgc_l25_m8 | 75.00 | 74.53 ± 1.11 | 87.75 ± 0.75 | 69.23 | 3750 | 0.000 | 0.950 | 0.479 / 0.465 |
| eletronics | tp_a2_sgc_l50_m4 | 74.50 | 73.10 ± 1.12 | 87.31 ± 0.75 | 66.83 | 3000 | 0.000 | 0.951 | 0.474 / 0.465 |
| eletronics | tp_a2_sgc_l50_m8 | 73.95 | 74.47 ± 1.09 | 87.69 ± 0.77 | 67.35 | 2750 | 0.000 | 0.951 | 0.477 / 0.465 |
| eletronics | tp_b_sgc_l50_m8 | 72.80 | 73.12 ± 1.13 | 87.52 ± 0.76 | 67.28 | 3750 | 0.000 | 0.947 | 0.479 / 0.465 |
| eletronics | tp_a2_pprrow_l50_m8 | 74.50 | 73.81 ± 1.12 | 87.64 ± 0.77 | 67.28 | 3750 | 0.000 | 0.824 | 0.349 / 0.272 |
| dblp | tp_l0 | 69.30 | 71.75 ± 1.19 | 81.41 ± 1.00 | 68.58 | 1750 | — | — | — / — |
| dblp | tp_a2_sgc_l25_m4 | 68.15 | 71.55 ± 1.17 | 81.56 ± 1.00 | 68.29 | 1750 | 0.000 | 0.967 | 0.328 / 0.326 |
| dblp | tp_a2_sgc_l25_m8 | 68.05 | 72.15 ± 1.17 | 82.10 ± 0.95 | 68.53 | 750 | 0.000 | 0.971 | 0.308 / 0.305 |
| dblp | tp_a2_sgc_l50_m4 | 68.10 | 71.98 ± 1.17 | 82.27 ± 0.95 | 68.38 | 750 | 0.000 | 0.975 | 0.327 / 0.326 |
| dblp | tp_a2_sgc_l50_m8 | 68.25 | 72.06 ± 1.18 | 82.33 ± 0.94 | 68.19 | 750 | 0.000 | 0.975 | 0.308 / 0.305 |
| dblp | tp_b_sgc_l50_m8 | 66.95 | 71.08 ± 1.16 | 82.05 ± 0.94 | 67.66 | 750 | 0.000 | 0.974 | 0.308 / 0.305 |
| dblp | tp_a2_pprrow_l50_m8 | 68.65 | 72.05 ± 1.17 | 82.08 ± 0.95 | 68.31 | 750 | 0.000 | 0.946 | 0.303 / 0.295 |

## 4. B1 − tp_l0 (seed 0, %p)

| arm | clothing 1s | clothing 5s | eletronics 1s | eletronics 5s | dblp 1s | dblp 5s |
|---|---|---|---|---|---|---|
| tp_a2_sgc_l25_m4 | +0.48 | -0.06 | +0.03 | +0.28 | -0.20 | +0.14 |
| tp_a2_sgc_l25_m8 | +0.41 | -0.01 | +0.20 | +0.21 | +0.41 | +0.69 |
| tp_a2_sgc_l50_m4 | +0.07 | +0.01 | -1.23 | -0.23 | +0.24 | +0.86 |
| tp_a2_sgc_l50_m8 | -0.41 | -0.06 | +0.14 | +0.15 | +0.31 | +0.91 |
| tp_b_sgc_l50_m8 | -0.33 | -0.19 | -1.21 | -0.02 | -0.66 | +0.64 |
| tp_a2_pprrow_l50_m8 | -0.90 | -0.07 | -0.52 | +0.10 | +0.30 | +0.67 |

## 5. Selection (mean best_valid over 3 datasets, arms 2–7)

| arm | mean best_valid |
|---|---|
| tp_a2_sgc_l25_m4 | 71.717 |
| tp_a2_sgc_l25_m8 | 71.767 |
| tp_a2_sgc_l50_m4 | 71.517 |
| tp_a2_sgc_l50_m8 | 71.367 |
| tp_b_sgc_l50_m8 | 70.350 |
| tp_a2_pprrow_l50_m8 | 71.783 |

C* = `tp_a2_pprrow_l50_m8`

## 6. B2 — seeds

| dataset | arm | s0 1s | s1 1s | s2 1s | mean ± sd 1s | s0 5s | s1 5s | s2 5s | mean ± sd 5s |
|---|---|---|---|---|---|---|---|---|---|
| clothing | tp_l0 | 84.78 | 84.10 | 85.37 | 84.75 ± 0.64 | 92.14 | 92.13 | 92.36 | 92.21 ± 0.13 |
| clothing | tp_a2_pprrow_l50_m8 | 83.88 | 83.40 | 83.80 | 83.69 ± 0.26 | 92.07 | 91.99 | 92.20 | 92.09 ± 0.11 |
| eletronics | tp_l0 | 74.33 | 74.70 | 74.52 | 74.52 ± 0.19 | 87.54 | 87.42 | 87.39 | 87.45 ± 0.08 |
| eletronics | tp_a2_pprrow_l50_m8 | 73.81 | 73.94 | 73.69 | 73.82 ± 0.13 | 87.64 | 87.78 | 87.62 | 87.68 ± 0.09 |
| dblp | tp_l0 | 71.75 | 72.75 | 72.96 | 72.48 ± 0.65 | 81.41 | 81.97 | 82.08 | 81.82 ± 0.36 |
| dblp | tp_a2_pprrow_l50_m8 | 72.05 | 72.78 | 73.25 | 72.70 ± 0.61 | 82.08 | 82.30 | 82.81 | 82.40 ± 0.38 |

Paired differences C* − tp_l0 (same seed):

| dataset | metric | s0 | s1 | s2 | mean | sd | #pos / #neg |
|---|---|---|---|---|---|---|---|
| clothing | test1_lr | -0.90 | -0.70 | -1.57 | -1.06 | 0.45 | 0 / 3 |
| clothing | test5_lr | -0.07 | -0.14 | -0.16 | -0.12 | 0.05 | 0 / 3 |
| eletronics | test1_lr | -0.52 | -0.76 | -0.82 | -0.70 | 0.16 | 0 / 3 |
| eletronics | test5_lr | +0.10 | +0.37 | +0.23 | 0.23 | 0.14 | 3 / 0 |
| dblp | test1_lr | +0.30 | +0.03 | +0.30 | 0.21 | 0.15 | 3 / 0 |
| dblp | test5_lr | +0.67 | +0.32 | +0.73 | 0.57 | 0.22 | 3 / 0 |

## 7. tp_l0 vs existing naq_feat (expB)

| dataset | seed | naq_feat 1s / 5s | tp_l0 1s / 5s |
|---|---|---|---|
| eletronics | 0 | 74.58 / 87.44 | 74.33 / 87.54 |
| eletronics | 1 | 74.69 / 87.44 | 74.70 / 87.42 |
| eletronics | 2 | 74.50 / 87.40 | 74.52 / 87.39 |
| dblp | 0 | 71.72 / 81.35 | 71.75 / 81.41 |

## 8. Preregistration — arithmetic comparison (no verdicts)

| ID | actual values |
|---|---|
| TPP001 | Gb1 9.54e-07, Gb2 9.54e-07, Gb3 0.00e+00 (finite=True), Gb4 2.38e-07; threshold 1e-5 |
| TPP002 | eletronics |Δ1s|=0.25, |Δ5s|=0.10; dblp |Δ1s|=0.03, |Δ5s|=0.06; threshold 0.7 |
| TPP003 | eletronics sgc m8 token purity 0.448 vs query purity 0.647; dblp sgc m8 token purity 0.300 vs query purity 0.157 |
| TPP004 | clothing overlap@8 0.831, pprrow hop≤2 0.851; eletronics overlap@8 0.825, pprrow hop≤2 0.621; dblp overlap@8 0.893, pprrow hop≤2 0.999; thresholds 0.5 / 0.7 |
| TPP005 | clothing tp_a2_sgc_l25_m8 Δ1s=+0.41; eletronics tp_a2_sgc_l25_m4 Δ1s=+0.03; dblp tp_a2_sgc_l50_m8 Δ1s=+0.31; threshold +0.5, ≥2 datasets |
| TPP006 | tp_a2_sgc_l50_m8 Δ1s: clothing -0.41, eletronics 0.14, dblp 0.31 |
| TPP007 | 1s LR b vs a2 (l50 m8): clothing 84.45 vs 84.37, eletronics 73.12 vs 74.47, dblp 71.08 vs 72.06 |
| TPP008 | 1s LR sgc vs pprrow (a2 l50 m8): clothing 84.37 vs 83.88, eletronics 74.47 vs 73.81, dblp 72.06 vs 72.05 |
| TPP009 | tp_a2_sgc_l50_m8 norm entropy: clothing 0.968, eletronics 0.951, dblp 0.975; threshold 0.9 |
| TPP010 | C*=tp_a2_pprrow_l50_m8: clothing mean Δ1s=-1.06, #pos=0/3; eletronics mean Δ1s=-0.70, #pos=0/3; dblp mean Δ1s=0.21, #pos=3/3; threshold +0.5 and ≥2 positive, ≥2 datasets |
