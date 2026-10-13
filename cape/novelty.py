"""Centroid novelty from labeled nodes only (CLAUDE.md §3: inputs = features, L_idx, y_L). No split access."""
import os

import numpy as np
import torch

from cape.data import load_graph, to_torch_sparse
from cape.graph_ops import adj_hat_torch, row_normalize_dense, sgc_features

DEV = "cuda"


def novelty_scores(Xs, L_idx, y_L):
    """Xs: row-L2-normalized features (torch [N,d]). novelty(u) = 1 - max_c cos(Xs[u], mu_c)."""
    L = torch.as_tensor(L_idx, device=Xs.device)
    yl = torch.as_tensor(y_L, device=Xs.device)
    cls, inv = torch.unique(yl, return_inverse=True)
    mu = torch.zeros(len(cls), Xs.shape[1], device=Xs.device).index_add_(0, inv, Xs[L])
    mu = mu / mu.norm(dim=1, keepdim=True).clamp_min(1e-12)
    out = torch.empty(Xs.shape[0], device=Xs.device)
    for s in range(0, Xs.shape[0], 8192):
        out[s:s + 8192] = 1 - (Xs[s:s + 8192] @ mu.t()).max(1).values
    return out.cpu().numpy()


def load_novelty(ds, r, L_idx, y_L):
    """Cached {'sgc': used by generators, 'raw': analysis only}. Uses graph structure/features only (no labels)."""
    path = f"cache/{ds}/novelty_r{r}.npz"
    if not os.path.exists(path):
        g = load_graph(ds)
        X = to_torch_sparse(g["X"], DEV)
        Ah = adj_hat_torch(g["A"], DEV)
        Xs = sgc_features(Ah, X)
        nov_sgc = novelty_scores(Xs, L_idx, y_L)
        del Xs; torch.cuda.empty_cache()
        Xn = row_normalize_dense(X)
        nov_raw = novelty_scores(Xn, L_idx, y_L)
        del Xn; torch.cuda.empty_cache()
        np.savez(path, sgc=nov_sgc.astype(np.float32), raw=nov_raw.astype(np.float32))
    z = np.load(path)
    return {"sgc": z["sgc"], "raw": z["raw"]}
