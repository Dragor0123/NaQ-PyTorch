# TP02 summary

## 1. Regression check and oracle_filter call sites

- clothing: tokens_equal=True, |L_new − L_TP01|=0.00e+00, |mean − softmax| (identical tokens)=0.00e+00
- eletronics: tokens_equal=True, |L_new − L_TP01|=0.00e+00, |mean − softmax| (identical tokens)=0.00e+00
- dblp: tokens_equal=True, |L_new − L_TP01|=0.00e+00, |mean − softmax| (identical tokens)=0.00e+00
- pass=True

`grep -rn oracle_filter tokproto/`:

```
tokproto/summarize_tp02.py:39:    grep = subprocess.run(["grep", "-rn", "oracle_filter", "tokproto/"], capture_output=True, text=True).stdout
tokproto/summarize_tp02.py:41:    S["regression"], S["grep_oracle_filter"] = reg, grep
tokproto/summarize_tp02.py:42:    L += ["## 1. Regression check and oracle_filter call sites", ""]
tokproto/summarize_tp02.py:46:    L += [f"- pass={reg['pass']}", "", "`grep -rn oracle_filter tokproto/`:", "", "```", grep, "```", ""]
tokproto/tokens.py:23:def oracle_filter(cand, ds):
tokproto/diag_oracle.py:8:from tokproto.tokens import load_candidates, oracle_filter, token_sets, token_stats
tokproto/diag_oracle.py:21:                   "sgcora": token_stats(*token_sets(oracle_filter(cand, ds), spt, qry, 8), spt, y)}
tokproto/train_tp.py:18:from tokproto.tokens import load_candidates, oracle_filter, token_sets, token_stats
tokproto/train_tp.py:74:            cand = oracle_filter(load_candidates(ds, "sgc", nbr, cue), ds)
```

## 2. Oracle-token diagnostics

| dataset | tok | token purity | shortfall | mean #tok | #tok=0 | 1–3 | 4–7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Amazon_clothing | sgc | 0.584 | 0.000 | 8.00 | 0.000 | 0.000 | 0.000 | 1.000 |
| Amazon_clothing | sgcora † | 1.000 | 0.251 | 6.64 | 0.063 | 0.107 | 0.081 | 0.749 |
| Amazon_eletronics | sgc | 0.448 | 0.000 | 8.00 | 0.000 | 0.000 | 0.000 | 1.000 |
| Amazon_eletronics | sgcora † | 1.000 | 0.311 | 6.32 | 0.083 | 0.117 | 0.111 | 0.689 |
| dblp | sgc | 0.300 | 0.000 | 8.00 | 0.000 | 0.000 | 0.000 | 1.000 |
| dblp | sgcora † | 1.000 | 0.451 | 5.79 | 0.071 | 0.189 | 0.192 | 0.549 |

## 3. Three seeds (LR %)

| dataset | arm | s0 1s | s1 1s | s2 1s | mean ± sd 1s | s0 5s | s1 5s | s2 5s | mean ± sd 5s |
|---|---|---|---|---|---|---|---|---|---|
| clothing | tp_l0 | 84.78 | 84.10 | 85.37 | 84.75 ± 0.64 | 92.14 | 92.13 | 92.36 | 92.21 ± 0.13 |
| clothing | tp_a2_sgc_l25_m8 | 85.18 | 84.34 | 84.90 | 84.81 ± 0.42 | 92.12 | 92.28 | 92.20 | 92.20 ± 0.08 |
| clothing | tp_a2_sgc_l25_m8_mean | 84.57 | 84.06 | 84.93 | 84.52 ± 0.43 | 91.73 | 92.29 | 92.31 | 92.11 ± 0.33 |
| eletronics | tp_l0 | 74.33 | 74.70 | 74.52 | 74.52 ± 0.19 | 87.54 | 87.42 | 87.39 | 87.45 ± 0.08 |
| eletronics | tp_a2_sgc_l25_m8 | 74.53 | 74.37 | 74.52 | 74.47 ± 0.09 | 87.75 | 88.00 | 87.63 | 87.79 ± 0.19 |
| eletronics | tp_a2_sgc_l25_m8_mean | 74.54 | 74.12 | 74.54 | 74.40 ± 0.25 | 87.80 | 87.93 | 87.87 | 87.86 ± 0.07 |
| dblp | tp_l0 | 71.75 | 72.75 | 72.96 | 72.48 ± 0.65 | 81.41 | 81.97 | 82.08 | 81.82 ± 0.36 |
| dblp | tp_a2_sgc_l25_m8 | 72.15 | 72.47 | 72.36 | 72.33 ± 0.16 | 82.10 | 82.16 | 81.94 | 82.07 ± 0.11 |
| dblp | tp_a2_sgc_l25_m8_mean | 71.60 | 72.43 | 72.08 | 72.04 ± 0.42 | 81.67 | 82.11 | 82.24 | 82.01 ± 0.30 |

## 4. Paired differences (same seed, %p)

| comparison | dataset | metric | s0 | s1 | s2 | mean | sd | #pos / n |
|---|---|---|---|---|---|---|---|---|
| tp_a2_sgc_l25_m8 − tp_l0 | clothing | test1_lr | +0.41 | +0.25 | -0.47 | +0.06 | 0.47 | 2 / 3 |
| tp_a2_sgc_l25_m8 − tp_l0 | clothing | test5_lr | -0.01 | +0.16 | -0.16 | -0.01 | 0.16 | 1 / 3 |
| tp_a2_sgc_l25_m8 − tp_l0 | eletronics | test1_lr | +0.20 | -0.33 | +0.00 | -0.04 | 0.27 | 2 / 3 |
| tp_a2_sgc_l25_m8 − tp_l0 | eletronics | test5_lr | +0.21 | +0.58 | +0.24 | +0.34 | 0.21 | 3 / 3 |
| tp_a2_sgc_l25_m8 − tp_l0 | dblp | test1_lr | +0.41 | -0.28 | -0.60 | -0.16 | 0.52 | 1 / 3 |
| tp_a2_sgc_l25_m8 − tp_l0 | dblp | test5_lr | +0.69 | +0.18 | -0.14 | +0.24 | 0.42 | 2 / 3 |
| tp_a2_sgc_l25_m8_mean − tp_l0 | clothing | test1_lr | -0.20 | -0.03 | -0.44 | -0.23 | 0.20 | 0 / 3 |
| tp_a2_sgc_l25_m8_mean − tp_l0 | clothing | test5_lr | -0.41 | +0.16 | -0.05 | -0.10 | 0.29 | 1 / 3 |
| tp_a2_sgc_l25_m8_mean − tp_l0 | eletronics | test1_lr | +0.21 | -0.59 | +0.02 | -0.12 | 0.42 | 2 / 3 |
| tp_a2_sgc_l25_m8_mean − tp_l0 | eletronics | test5_lr | +0.26 | +0.51 | +0.47 | +0.41 | 0.14 | 3 / 3 |
| tp_a2_sgc_l25_m8_mean − tp_l0 | dblp | test1_lr | -0.15 | -0.32 | -0.88 | -0.45 | 0.38 | 0 / 3 |
| tp_a2_sgc_l25_m8_mean − tp_l0 | dblp | test5_lr | +0.26 | +0.14 | +0.16 | +0.19 | 0.06 | 3 / 3 |
| tp_a2_sgc_l25_m8 − tp_a2_sgc_l25_m8_mean | clothing | test1_lr | +0.61 | +0.28 | -0.03 | +0.29 | 0.32 | 2 / 3 |
| tp_a2_sgc_l25_m8 − tp_a2_sgc_l25_m8_mean | clothing | test5_lr | +0.39 | -0.00 | -0.11 | +0.09 | 0.27 | 1 / 3 |
| tp_a2_sgc_l25_m8 − tp_a2_sgc_l25_m8_mean | eletronics | test1_lr | -0.01 | +0.25 | -0.02 | +0.08 | 0.16 | 1 / 3 |
| tp_a2_sgc_l25_m8 − tp_a2_sgc_l25_m8_mean | eletronics | test5_lr | -0.05 | +0.07 | -0.23 | -0.07 | 0.15 | 1 / 3 |
| tp_a2_sgc_l25_m8 − tp_a2_sgc_l25_m8_mean | dblp | test1_lr | +0.56 | +0.04 | +0.27 | +0.29 | 0.26 | 3 / 3 |
| tp_a2_sgc_l25_m8 − tp_a2_sgc_l25_m8_mean | dblp | test5_lr | +0.43 | +0.04 | -0.30 | +0.06 | 0.37 | 2 / 3 |

## 5. Seed 0

| dataset | arm | 1s LR ± CI | 5s LR ± CI | best ep | token purity | mean #tok | attn H/logT | mass same-label / uniform |
|---|---|---|---|---|---|---|---|---|
| clothing | tp_l0 | 84.78 ± 0.98 | 92.14 ± 0.68 | 3750 | — | — | — | — / — |
| clothing | tp_a2_sgc_l25_m8 | 85.18 ± 1.00 | 92.12 ± 0.67 | 1750 | 0.580 | 8.00 | 0.971 | 0.589 / 0.583 |
| clothing | tp_a2_sgc_l50_m8 | 84.37 ± 1.03 | 92.08 ± 0.68 | 2250 | 0.580 | 8.00 | 0.968 | 0.589 / 0.583 |
| clothing | tp_a2_sgcora_l25_m8 † | 84.41 ± 1.04 | 91.74 ± 0.69 | 1000 | 1.000 | 6.62 | 0.976 | 1.000 / 1.000 |
| clothing | tp_a2_sgcora_l50_m8 † | 84.93 ± 0.99 | 92.10 ± 0.67 | 1750 | 1.000 | 6.62 | 0.972 | 1.000 / 1.000 |
| clothing | tp_a2_sgc_l25_m8_t02 | 84.84 ± 1.01 | 92.07 ± 0.67 | 1750 | 0.580 | 8.00 | 0.812 | 0.598 / 0.583 |
| eletronics | tp_l0 | 74.33 ± 1.14 | 87.54 ± 0.76 | 4000 | — | — | — | — / — |
| eletronics | tp_a2_sgc_l25_m8 | 74.53 ± 1.11 | 87.75 ± 0.75 | 3750 | 0.448 | 8.00 | 0.950 | 0.479 / 0.465 |
| eletronics | tp_a2_sgc_l50_m8 | 74.47 ± 1.09 | 87.69 ± 0.77 | 2750 | 0.448 | 8.00 | 0.951 | 0.477 / 0.465 |
| eletronics | tp_a2_sgcora_l25_m8 † | 74.17 ± 1.14 | 87.50 ± 0.76 | 4000 | 1.000 | 6.31 | 0.953 | 1.000 / 1.000 |
| eletronics | tp_a2_sgcora_l50_m8 † | 74.00 ± 1.13 | 87.61 ± 0.76 | 3500 | 1.000 | 6.31 | 0.958 | 1.000 / 1.000 |
| eletronics | tp_a2_sgc_l25_m8_t02 | 74.04 ± 1.14 | 87.55 ± 0.76 | 4000 | 0.448 | 8.00 | 0.692 | 0.493 / 0.465 |
| dblp | tp_l0 | 71.75 ± 1.19 | 81.41 ± 1.00 | 1750 | — | — | — | — / — |
| dblp | tp_a2_sgc_l25_m8 | 72.15 ± 1.17 | 82.10 ± 0.95 | 750 | 0.299 | 8.00 | 0.971 | 0.308 / 0.305 |
| dblp | tp_a2_sgc_l50_m8 | 72.06 ± 1.18 | 82.33 ± 0.94 | 750 | 0.299 | 8.00 | 0.975 | 0.308 / 0.305 |
| dblp | tp_a2_sgcora_l25_m8 † | 72.06 ± 1.18 | 82.05 ± 0.97 | 750 | 1.000 | 5.79 | 0.960 | 1.000 / 1.000 |
| dblp | tp_a2_sgcora_l50_m8 † | 71.44 ± 1.18 | 81.98 ± 0.96 | 1250 | 1.000 | 5.79 | 0.958 | 1.000 / 1.000 |
| dblp | tp_a2_sgc_l25_m8_t02 | 72.20 ± 1.18 | 82.00 ± 0.96 | 750 | 0.299 | 8.00 | 0.720 | 0.313 / 0.305 |

Seed 0 differences (%p):

| comparison | clothing 1s | clothing 5s | eletronics 1s | eletronics 5s | dblp 1s | dblp 5s |
|---|---|---|---|---|---|---|
| tp_a2_sgcora_l25_m8 † − tp_a2_sgc_l25_m8 | -0.77 | -0.39 | -0.36 | -0.25 | -0.09 | -0.06 |
| tp_a2_sgcora_l50_m8 † − tp_a2_sgc_l50_m8 | +0.56 | +0.02 | -0.47 | -0.08 | -0.62 | -0.34 |
| tp_a2_sgcora_l25_m8 † − tp_l0 | -0.37 | -0.40 | -0.15 | -0.04 | +0.31 | +0.63 |
| tp_a2_sgcora_l50_m8 † − tp_l0 | +0.15 | -0.03 | -0.33 | +0.07 | -0.31 | +0.57 |
| tp_a2_sgc_l25_m8_t02 − tp_a2_sgc_l25_m8 | -0.34 | -0.05 | -0.49 | -0.19 | +0.05 | -0.10 |

## 6. Preregistration — arithmetic comparison (no verdicts)

| ID | actual values |
|---|---|
| TPP011 | sgc_l25_m8 − tp_l0, 3-seed mean 1s (#pos): clothing +0.06 (2/3), eletronics -0.04 (2/3), dblp -0.16 (1/3); criterion: datasets with mean ≥ +0.3 and #pos ≥ 2 is ≤ 1 |
| TPP012 | |sgc_l25_m8 − sgc_l25_m8_mean|, 3-seed mean 1s: clothing 0.29, eletronics 0.08, dblp 0.29; threshold ≤ 0.3 all |
| TPP013 | t02 norm entropy: clothing 0.812, eletronics 0.692, dblp 0.720; threshold ≤ 0.8 all |
| TPP014 | t02 − sgc_l25_m8 (s0, 1s): clothing -0.34, eletronics -0.49, dblp +0.05; criterion: ≤ +0.3 in ≥ 2 datasets |
| TPP015 | sgcora_l50_m8† − tp_l0 (s0, 1s): clothing +0.15, eletronics -0.33, dblp -0.31; criterion: datasets ≥ +1.0 is ≤ 1 |
| TPP016 | sgcora_l50_m8† − sgc_l50_m8 (s0, 1s): clothing +0.56, eletronics -0.47, dblp -0.62; criterion: largest in eletronics |
