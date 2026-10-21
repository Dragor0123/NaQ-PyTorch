"""pprrow ranker: cosine between PPR row vectors, top-50 per node -> cache/{ds}/pprrow_top50.npz."""
import argparse
import json
import os
import time

import numpy as np
import torch

from cape.data import load_graph
from cape.graph_ops import adj_hat_torch, ppr_columns
from cape.pool import log

DEV = "cuda"
TOPK = 50


def ppr_apply(Ahat, M, alpha=0.1, K=50):
    """Pi @ M = alpha * sum_{k=0..K} (1-alpha)^k Ahat^k M for dense M [N, B] (same series as ppr_columns)."""
    Z = M
    acc = alpha * M
    for k in range(1, K + 1):
        Z = torch.sparse.mm(Ahat, Z)
        acc.add_(Z, alpha=alpha * (1 - alpha) ** k)
    return acc


def build(ds, B=2048):
    T = {}
    g = load_graph(ds)
    N = g["A"].shape[0]
    Ah = adj_hat_torch(g["A"], DEV)
    t = time.time()
    norm = torch.zeros(N, device=DEV)
    for s in range(0, N, B):
        b = torch.arange(s, min(s + B, N), device=DEV)
        norm[b] = ppr_columns(Ah, b).norm(dim=0)
    T["norm_s"] = time.time() - t
    t = time.time()
    nbr = np.zeros((N, TOPK), np.int64)
    score = np.zeros((N, TOPK), np.float32)
    for s in range(0, N, B):
        b = torch.arange(s, min(s + B, N), device=DEV)
        G = ppr_apply(Ah, ppr_columns(Ah, b))              # [N, nb] = (Pi^2)[:, b]
        G.div_(norm[:, None] * norm[b][None, :])
        G[b, torch.arange(len(b), device=DEV)] = -float("inf")
        v, i = torch.topk(G, TOPK, dim=0)                   # per column
        nbr[s:s + len(b)] = i.t().cpu().numpy()
        score[s:s + len(b)] = v.t().cpu().numpy()
        del G
    T["topk_s"] = time.time() - t
    return Ah, norm, nbr, score, T


def verify(Ah, norm, nbr, score, seed=0):
    N = len(nbr)
    rng = np.random.default_rng(seed)
    us = rng.choice(N, 200, replace=False)
    max_err, viol = 0.0, 0
    for u in us:
        js = rng.choice(TOPK, 5, replace=False)
        vs = nbr[u, js]
        P = ppr_columns(Ah, np.concatenate([[u], vs]))
        c = torch.nn.functional.cosine_similarity(P[:, :1], P[:, 1:], dim=0).cpu().numpy()
        max_err = max(max_err, float(np.abs(c - score[u, js]).max()))
        r = rng.choice(N, 1000, replace=False)
        r = r[(r != u) & ~np.isin(r, nbr[u])]
        Pu = ppr_columns(Ah, [int(u)])
        Pr = ppr_columns(Ah, r)
        cr = (Pr * Pu).sum(0) / (Pr.norm(dim=0) * Pu.norm())
        viol += int((cr.cpu().numpy() > score[u, -1] + 1e-4).sum())
    return {"max_abs_err": max_err, "random_exceed_50th": viol, "pass": bool(max_err <= 1e-4 and viol == 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ds = args.dataset
    path = f"cache/{ds}/pprrow_top50.npz"
    if os.path.exists(path) and not args.force:
        log(f"{path} exists, skip")
        return
    t0 = time.time()
    Ah, norm, nbr, score, T = build(ds)
    t = time.time()
    chk = verify(Ah, norm, nbr, score)
    T["verify_s"] = time.time() - t
    T["total_s"] = time.time() - t0
    log(f"{ds} pprrow verify={chk} time={T} zero_score_frac={(score <= 0).mean():.4f}")
    if not chk["pass"]:
        raise SystemExit("pprrow verification failed")
    np.savez(path, nbr=nbr, score=score)
    os.makedirs("results/tp", exist_ok=True)
    json.dump({"verify": chk, "time": T, "zero_score_frac": float((score <= 0).mean())},
              open(f"results/tp/{ds}_pprrow_build.json", "w"), indent=1)


if __name__ == "__main__":
    main()
