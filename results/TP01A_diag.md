# TP01 A — token diagnostics (seed 0, first 2,000 training episodes)

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
