"""Candidate pool (union of top-50 raw / ppr / sgc) and 6 pair cues -> cache/{ds}/pool.npz."""
import argparse
import json
import os
import time

import numpy as np
import torch

from cape.data import load_graph, to_torch_sparse
from cape.graph_ops import adj_hat_torch, ppr_columns, row_normalize_dense, sgc_features

DEV = "cuda"
P, TOPK = 150, 50
CUES = ["cos_raw", "cos_sgc", "log_ppr", "adj", "mknn_raw", "neg_euclid_raw"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_features(g):
    X = to_torch_sparse(g["X"], DEV)
    Ah = adj_hat_torch(g["A"], DEV)
    Xs = sgc_features(Ah, X)
    Xn = row_normalize_dense(X)
    del X
    torch.cuda.empty_cache()
    return Xn, Xs, Ah


def pool_blocks(Xn, Xs, Ah, B):
    N = Xn.shape[0]
    nbr = np.full((N, P), -1, np.int32)
    cue = np.full((N, P, 3), np.nan, np.float32)
    ov = {"raw_sgc": [], "raw_ppr": [], "sgc_ppr": []}
    for s in range(0, N, B):
        b = torch.arange(s, min(s + B, N), device=DEV)
        nb, ar = len(b), torch.arange(len(b), device=DEV)
        S_raw = Xn[b] @ Xn.t()                  # [nb, N]
        S_sgc = Xs[b] @ Xs.t()
        S_ppr = ppr_columns(Ah, b).t().contiguous()  # symmetric: column block == row block
        tops = []
        for S in (S_raw, S_sgc, S_ppr):
            S[ar, b] = -float("inf")
            tops.append(torch.topk(S, TOPK, dim=1).indices)
        for (k1, k2), (i, j) in zip([("raw", "sgc"), ("raw", "ppr"), ("sgc", "ppr")], [(0, 1), (0, 2), (1, 2)]):
            ov[f"{k1}_{k2}"].append(((tops[i][:, :, None] == tops[j][:, None, :]).any(2).sum(1).float() / TOPK).cpu())
        c = torch.cat(tops, 1).sort(1).values
        dup = torch.zeros_like(c, dtype=torch.bool)
        dup[:, 1:] = c[:, 1:] == c[:, :-1]
        c[dup] = N
        c = c.sort(1).values
        pad = c == N
        idx = c.clamp(max=N - 1)
        vals = torch.stack([S_raw.gather(1, idx), S_sgc.gather(1, idx),
                            torch.log(S_ppr.gather(1, idx) + 1e-6)], 2)
        vals[pad] = float("nan")
        c[pad] = -1
        nbr[s:s + nb] = c.cpu().numpy()
        cue[s:s + nb] = vals.cpu().numpy()
        del S_raw, S_sgc, S_ppr, tops, vals
    torch.cuda.empty_cache()
    return nbr, cue, {k: float(torch.cat(v).mean()) for k, v in ov.items()}


def graph_cues(g, nbr, cos_raw):
    N = nbr.shape[0]
    u = np.repeat(np.arange(N, dtype=np.int64), P).reshape(N, P)
    v = nbr.astype(np.int64)
    valid = v >= 0
    key = u * N + v
    A = g["A"].tocoo()
    adj = np.isin(key, A.row.astype(np.int64) * N + A.col).astype(np.float32)
    # mutual kNN on raw cosine (top-10 inside pool == global top-10)
    order = np.argsort(-np.where(valid, cos_raw, -np.inf), axis=1, kind="stable")[:, :10]
    t10 = np.take_along_axis(v, order, 1)
    t10_keys = (np.arange(N, dtype=np.int64)[:, None] * N + t10).ravel()
    in_u = np.isin(key, t10_keys)            # v in top10(u)
    in_v = np.isin(v * N + u, t10_keys)      # u in top10(v)
    mknn = (in_u & in_v).astype(np.float32)
    nrm = np.sqrt(np.asarray(g["X"].multiply(g["X"]).sum(1)).reshape(-1)).astype(np.float32)
    nu, nv = nrm[u], nrm[np.where(valid, v, 0)]
    neg_euc = -np.sqrt(np.maximum(nu ** 2 + nv ** 2 - 2 * nu * nv * cos_raw, 0))
    out = np.stack([adj, mknn, neg_euc], 2)
    out[~valid] = np.nan
    return out


def topk_in_pool(nbr, score, k):
    sc = np.where(nbr >= 0, score, -np.inf)
    o = np.argsort(-sc, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(nbr, o, 1), np.take_along_axis(sc, o, 1)


def run_checks(Xn, Xs, Ah, nbr, cue, seed=0):
    N = nbr.shape[0]
    rng = np.random.default_rng(seed)
    res = {}
    # 1. brute-force top-10 cosine (raw, sgc)
    a = rng.choice(N, 100, replace=False)
    at = torch.from_numpy(a).to(DEV)
    for name, M, ci in (("raw", Xn, 0), ("sgc", Xs, 1)):
        S = M[at] @ M.t()
        S[torch.arange(100, device=DEV), at] = -float("inf")
        bv, bi = torch.topk(S, 10, dim=1)
        bv, bi = bv.cpu().numpy(), bi.cpu().numpy()
        pi, pv = topk_in_pool(nbr[a], cue[a, :, ci], 10)
        exact = np.array([set(bi[i]) == set(pi[i]) for i in range(100)])
        tie = np.array([np.allclose(np.sort(bv[i]), np.sort(pv[i]), atol=1e-5) for i in range(100)])
        res[f"check1_{name}"] = {"exact_match": int(exact.sum()), "tie_only_diff": int((~exact & tie).sum()),
                                 "real_mismatch": int((~exact & ~tie).sum())}
    # 2. PPR K=100 vs pooled K=50 top-10
    a = rng.choice(N, 50, replace=False)
    P100 = ppr_columns(Ah, torch.from_numpy(a).to(DEV), K=100).t()
    P100[torch.arange(50, device=DEV), torch.from_numpy(a).to(DEV)] = -float("inf")
    t100 = torch.topk(P100, 10, dim=1).indices.cpu().numpy()
    p50, _ = topk_in_pool(nbr[a], cue[a, :, 2], 10)
    res["check2_ppr_overlap"] = float(np.mean([len(set(t100[i]) & set(p50[i])) / 10 for i in range(50)]))
    # 3. symmetry on pairs present in both pools
    errs, errs_big, tries = [], [], 0
    while len(errs) < 1000 and tries < 200000:
        tries += 1
        u = int(rng.integers(N)); j = int(rng.integers(P))
        v = int(nbr[u, j])
        if v < 0:
            continue
        k = np.nonzero(nbr[v] == u)[0]
        if len(k) == 0:
            continue
        x, y = np.exp(np.float64(cue[u, j, 2])), np.exp(np.float64(cue[v, k[0], 2]))  # = Pi + 1e-6
        errs.append(abs(x - y) / max(x, y))
        if min(x, y) > 1e-4:
            errs_big.append(abs(x - y) / max(x, y))
    res["check3_sym_median_relerr"] = float(np.median(errs))
    res["check3_sym_median_relerr_pi_gt_1e-4"] = float(np.median(errs_big)) if errs_big else None
    res["check3_n_pairs"] = len(errs)
    res["passed"] = bool(all(res[f"check1_{n}"]["real_mismatch"] == 0 for n in ("raw", "sgc"))
                         and res["check2_ppr_overlap"] >= 0.95 and res["check3_sym_median_relerr"] <= 1e-3
                         and len(errs) == 1000)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--block", type=int, default=2048)
    args = ap.parse_args()
    ds = args.dataset
    torch.manual_seed(0); np.random.seed(0)
    path = f"cache/{ds}/pool.npz"
    t0 = time.time(); times = {}
    g = load_graph(ds)
    Xn, Xs, Ah = build_features(g)
    times["features_s"] = time.time() - t0
    log(f"{ds} features {times['features_s']:.1f}s, gpu mem {torch.cuda.max_memory_allocated() / 2**30:.2f}GB")
    stats_path = f"results/expA/{ds}_pool.json"
    if os.path.exists(path) and not args.force:
        z = np.load(path); nbr, cue = z["nbr"], z["cue"]
        stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {}
        log("reusing pool.npz")
    else:
        t = time.time()
        try:
            nbr, cue3, ov = pool_blocks(Xn, Xs, Ah, args.block)
            block = args.block
        except torch.cuda.OutOfMemoryError:
            log("OOM, retry block 1024"); torch.cuda.empty_cache()
            nbr, cue3, ov = pool_blocks(Xn, Xs, Ah, 1024)
            block = 1024
        times["blocks_s"] = time.time() - t
        log(f"blocks {times['blocks_s']:.1f}s")
        t = time.time()
        cue = np.concatenate([cue3, graph_cues(g, nbr, cue3[:, :, 0])], 2).astype(np.float32)
        times["graph_cues_s"] = time.time() - t
        log(f"graph cues {times['graph_cues_s']:.1f}s")
        ps = (nbr >= 0).sum(1)
        np.savez(path, nbr=nbr, cue=cue, pool_size=ps.astype(np.int32))
        stats = {"block": block, "pool_size": {"mean": float(ps.mean()), "median": float(np.median(ps)),
                                               "min": int(ps.min()), "max": int(ps.max())},
                 "top50_overlap": ov, "cue_names": CUES,
                 "adj_frac": float(np.nanmean(cue[:, :, 3])), "mknn_frac": float(np.nanmean(cue[:, :, 4]))}
    t = time.time()
    stats["checks"] = run_checks(Xn, Xs, Ah, nbr, cue)
    times["checks_s"] = time.time() - t
    stats["times"] = {**stats.get("times", {}), **times}
    stats["peak_gpu_GB"] = torch.cuda.max_memory_allocated() / 2**30
    os.makedirs("results/expA", exist_ok=True)
    json.dump(stats, open(stats_path, "w"), indent=1)
    log(json.dumps(stats))
    if not stats["checks"]["passed"]:
        raise SystemExit(f"{ds}: pool checks FAILED")


if __name__ == "__main__":
    main()
