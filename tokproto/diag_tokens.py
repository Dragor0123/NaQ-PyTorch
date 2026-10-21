"""A: training-free token diagnostics (labels used for analysis only) -> results/TP01A_diag.{json,md}."""
import argparse
import json
import time

import numpy as np
from scipy.sparse.csgraph import shortest_path

from cape.data import DATASETS, load_graph
from cape.generators import N_QRY, N_WAY, top_in_pool
from cape.pool import log
from tokproto.tokens import load_candidates, token_sets, token_stats
from tokproto.train_tp import make_episodes


def hop_dist(D):
    """D: hop distances (inf = unreachable) -> fractions and median over reachable."""
    D = np.asarray(D, np.float64).reshape(-1)
    fin = np.isfinite(D)
    return {"1": float((D == 1).mean()), "2": float((D == 2).mean()), ">=3": float((fin & (D >= 3)).mean()),
            "unreachable": float((~fin).mean()), "le2": float((D <= 2).mean()),
            "median": float(np.median(D[fin])) if fin.any() else None}


def run(ds, n_eps, n_hop):
    t0 = time.time()
    z = np.load(f"cache/{ds}/pool.npz")
    nbr, cue = z["nbr"], z["cue"]
    spt, qry = make_episodes(nbr, cue, 0, 4000)
    spt, qry = spt[:n_eps], qry[:n_eps]
    g = load_graph(ds)
    y = g["y"]
    q3 = qry.reshape(len(spt), N_WAY, N_QRY)
    out = {"query_purity": float((y[q3] == y[spt][:, :, None]).mean())}
    # first n_hop (episode, way) anchors
    flat_a = spt.reshape(-1)[:n_hop]
    ua, inv = np.unique(flat_a, return_inverse=True)
    t = time.time()
    Dm = shortest_path(g["A"], directed=False, unweighted=True, indices=ua)   # [n_unique, N]
    out["hop_time_s"] = time.time() - t
    ew = np.unravel_index(np.arange(len(flat_a)), spt.shape)
    out["hop_query"] = hop_dist(Dm[inv[:, None], q3[ew]])
    cands = {}
    for tok in ("sgc", "pprrow"):
        cand = load_candidates(ds, tok, nbr, cue)
        cands[tok] = cand
        for m in (4, 8):
            tk, mk = token_sets(cand, spt, qry, m)
            out[f"{tok}_m{m}"] = token_stats(tk, mk, spt, y)
            if m == 8:
                sub_t, sub_m = tk[ew], mk[ew]
                out[f"{tok}_m{m}"]["hop"] = hop_dist(Dm[inv[:, None], sub_t][sub_m])
    ppr8 = top_in_pool(nbr, cue[:, :, 2], 8)
    pr8 = cands["pprrow"][:, :8]
    out["pprrow_ppr_overlap8"] = float(np.mean([len(np.intersect1d(pr8[a], ppr8[a])) / 8 for a in flat_a]))
    sc = np.load(f"cache/{ds}/pprrow_top50.npz")["score"]
    out["pprrow_zero_score_frac_top8"] = float((sc[flat_a, :8] <= 0).mean())
    out["n_eps"], out["n_hop_anchors"], out["time_s"] = n_eps, len(flat_a), time.time() - t0
    log(f"{ds} diag done {out['time_s']:.1f}s")
    return out


def to_md(res):
    L = ["# TP01 A — token diagnostics (seed 0, first 2,000 training episodes)", "",
         "| dataset | ranker | m | token purity | query purity | shortfall | empty | mean #tok | hop1 | hop2 | hop≥3 | unreach | hop med | pprrow–ppr ov@8 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds, r in res.items():
        for tok in ("sgc", "pprrow"):
            for m in (4, 8):
                s = r[f"{tok}_m{m}"]
                h = s.get("hop")
                hs = " | ".join(f"{h[k]:.3f}" for k in ("1", "2", ">=3", "unreachable")) + f" | {h['median']}" if h else " | | | | "
                ov = f"{r['pprrow_ppr_overlap8']:.3f}" if tok == "pprrow" and m == 8 else ""
                L.append(f"| {ds} | {tok} | {m} | {s['token_purity']:.3f} | {r['query_purity']:.3f} | {s['shortfall_rate']:.3f} | "
                         f"{s['empty_rate']:.3f} | {s['mean_tokens']:.2f} | {hs} | {ov} |")
        h = r["hop_query"]
        L.append(f"| {ds} | NaQ query | 10 | | {r['query_purity']:.3f} | | | | " + " | ".join(f"{h[k]:.3f}" for k in ("1", "2", ">=3", "unreachable")) + f" | {h['median']} | |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_eps", type=int, default=2000)
    ap.add_argument("--n_hop", type=int, default=1000)
    args = ap.parse_args()
    res = {ds: run(ds, args.n_eps, args.n_hop) for ds in DATASETS}
    json.dump(res, open("results/TP01A_diag.json", "w"), indent=1)
    open("results/TP01A_diag.md", "w").write(to_md(res))
    print(to_md(res))


if __name__ == "__main__":
    main()
