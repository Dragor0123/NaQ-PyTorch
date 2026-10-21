"""TP02 §4: sgcora token shortfall / count distribution vs sgc purity (seed 0, first 2,000 eps, m=8)
-> results/TP02A_oracle_tokens.{json,md}. Labels: analysis only."""
import json

import numpy as np

from cape.data import DATASETS, load_graph
from tokproto.tokens import load_candidates, oracle_filter, token_sets, token_stats
from tokproto.train_tp import make_episodes


def main():
    res = {}
    for ds in DATASETS:
        z = np.load(f"cache/{ds}/pool.npz")
        spt, qry = make_episodes(z["nbr"], z["cue"], 0, 4000)
        spt, qry = spt[:2000], qry[:2000]
        y = load_graph(ds)["y"]
        cand = load_candidates(ds, "sgc", z["nbr"], z["cue"])
        res[ds] = {"sgc": token_stats(*token_sets(cand, spt, qry, 8), spt, y),
                   "sgcora": token_stats(*token_sets(oracle_filter(cand, ds), spt, qry, 8), spt, y)}
    json.dump(res, open("results/TP02A_oracle_tokens.json", "w"), indent=1)
    L = ["# TP02 A — oracle tokens (seed 0, first 2,000 episodes, m = 8)", "",
         "| dataset | tok | token purity | shortfall | mean #tok | #tok=0 | 1–3 | 4–7 | 8 |", "|---|---|---|---|---|---|---|---|---|"]
    for ds, r in res.items():
        for t in ("sgc", "sgcora"):
            s, c = r[t], r[t]["count_dist"]
            L.append(f"| {ds} | {t}{' †' if t == 'sgcora' else ''} | {s['token_purity']:.3f} | {s['shortfall_rate']:.3f} | {s['mean_tokens']:.2f} | "
                     f"{c['0']:.3f} | {c['1-3']:.3f} | {c['4-7']:.3f} | {c['8']:.3f} |")
    open("results/TP02A_oracle_tokens.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
