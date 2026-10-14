"""T07 novelty estimators. Inputs: graph structure/features, L_idx, y_L only (CLAUDE.md §3). No split access."""
import os
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from cape.data import load_graph, to_torch_sparse
from cape.graph_ops import adj_hat_torch, row_normalize_dense, sgc_features
from cape.labelmask import make_mask
from cape.novelty import novelty_scores
from cape.pool import log

DEV = "cuda"
ESTIMATORS = ["cent_sgc", "cent_raw", "msp", "pu_feat", "pu_struct", "pu_both"]


def known_struct(Ah, L_idx, alpha=0.1, K=50):
    """m_L / m_V with m_S = sum_{k=1..K} (1-alpha)^k Ah^k 1_S."""
    N = Ah.shape[0]
    Z = torch.zeros(N, 2, device=Ah.device)
    Z[torch.as_tensor(L_idx, device=Ah.device), 0] = 1.0
    Z[:, 1] = 1.0
    acc = torch.zeros_like(Z)
    for k in range(1, K + 1):
        Z = torch.sparse.mm(Ah, Z)
        acc.add_(Z, alpha=(1 - alpha) ** k)
    return (acc[:, 0] / (acc[:, 1] + 1e-12)).cpu().numpy()


def pu_novelty(F, L_idx, seed=0):
    """Elkan-Noto: g = P(s=1|x) with s = 1[u in L]; c = mean OOF g on L; novelty = 1 - min(1, g/c)."""
    s = np.zeros(len(F), np.int64); s[L_idx] = 1
    mk = lambda: LogisticRegression(max_iter=2000, class_weight="balanced")
    g = mk().fit(F, s).predict_proba(F)[:, 1]
    oof = np.zeros(len(F))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(F, s):
        oof[te] = mk().fit(F[tr], s[tr]).predict_proba(F[te])[:, 1]
    c = float(oof[L_idx].mean())
    return 1 - np.minimum(1.0, g / c), c


def compute_estimators(ds, r):
    path = f"cache/{ds}/novelty2_r{r}.npz"
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    L_idx, y_L = make_mask(ds, r)
    g = load_graph(ds)                       # uses A and X only
    A, Xsp = g["A"], g["X"]
    X = to_torch_sparse(Xsp, DEV)
    Ah = adj_hat_torch(A, DEV)
    T = {}
    t = time.time()
    Xs_t = sgc_features(Ah, X)
    out = {"cent_sgc": novelty_scores(Xs_t, L_idx, y_L)}
    Xs = Xs_t.cpu().numpy(); del Xs_t
    Xr_t = row_normalize_dense(X)
    out["cent_raw"] = novelty_scores(Xr_t, L_idx, y_L); del Xr_t
    ks = known_struct(Ah, L_idx)
    ld = np.log1p(np.asarray(A.sum(1)).reshape(-1))
    torch.cuda.empty_cache()
    T["features_s"] = time.time() - t
    t = time.time()
    P = LogisticRegression(max_iter=2000).fit(Xs[L_idx], y_L).predict_proba(Xs)
    out["msp"] = 1 - P.max(1)
    T["msp_s"] = time.time() - t
    cs = {}
    t = time.time()
    out["pu_feat"], cs["pu_feat"] = pu_novelty(Xs, L_idx)
    T["pu_feat_s"] = time.time() - t; t = time.time()
    S = StandardScaler().fit_transform(np.stack([ks, ld], 1))
    out["pu_struct"], cs["pu_struct"] = pu_novelty(S, L_idx)
    T["pu_struct_s"] = time.time() - t; t = time.time()
    B = StandardScaler().fit_transform(np.concatenate([Xs, ks[:, None], ld[:, None]], 1))
    out["pu_both"], cs["pu_both"] = pu_novelty(B, L_idx)
    T["pu_both_s"] = time.time() - t
    log(f"{ds} r={r} estimators times {T} c={cs}")
    out = {k: v.astype(np.float32) for k, v in out.items()}
    np.savez(path, **out, known_struct=ks.astype(np.float32),
             pu_c=np.array([cs[k] for k in ("pu_feat", "pu_struct", "pu_both")], np.float32))
    return {**out, "known_struct": ks}


# --- T08: calibrated PU variants (no class_weight, PCA-256 of Xs fit unsupervised on all nodes)
ESTIMATORS3 = ["pu_feat2", "pu_both2", "pu_struct"]


def pu_novelty2(F, L_idx, seed=0):
    """Elkan-Noto as pu_novelty, but LogisticRegression(C=1.0, max_iter=2000) without class_weight."""
    s = np.zeros(len(F), np.int64); s[L_idx] = 1
    mk = lambda: LogisticRegression(C=1.0, max_iter=2000)
    g = mk().fit(F, s).predict_proba(F)[:, 1]
    oof = np.zeros(len(F))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(F, s):
        oof[te] = mk().fit(F[tr], s[tr]).predict_proba(F[te])[:, 1]
    c = float(oof[L_idx].mean())
    return 1 - np.minimum(1.0, g / c), c


def compute_estimators3(ds, r, skip_both=False):
    from sklearn.decomposition import PCA
    path = f"cache/{ds}/novelty3_r{r}.npz"
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    L_idx, y_L = make_mask(ds, r)
    g = load_graph(ds)                       # uses A and X only
    A = g["A"]
    Ah = adj_hat_torch(A, DEV)
    Xs = sgc_features(Ah, to_torch_sparse(g["X"], DEV)).cpu().numpy()
    torch.cuda.empty_cache()
    T, out, cs = {}, {}, {}
    t = time.time()
    Z = PCA(n_components=256, svd_solver="randomized", random_state=0).fit_transform(Xs)
    del Xs
    T["pca_s"] = time.time() - t
    ks = known_struct(Ah, L_idx)
    ld = np.log1p(np.asarray(A.sum(1)).reshape(-1))
    t = time.time()
    out["pu_feat2"], cs["pu_feat2"] = pu_novelty2(Z, L_idx)
    T["pu_feat2_s"] = time.time() - t
    if not skip_both:
        t = time.time()
        out["pu_both2"], cs["pu_both2"] = pu_novelty2(StandardScaler().fit_transform(np.concatenate([Z, ks[:, None], ld[:, None]], 1)), L_idx)
        T["pu_both2_s"] = time.time() - t
    t = time.time()
    out["pu_struct"], cs["pu_struct"] = pu_novelty(StandardScaler().fit_transform(np.stack([ks, ld], 1)), L_idx)
    T["pu_struct_s"] = time.time() - t
    log(f"{ds} r={r} estimators3 times {T} c={cs}")
    out = {k: v.astype(np.float32) for k, v in out.items()}
    np.savez(path, **out, **{f"time_{k}": np.float32(v) for k, v in T.items()}, **{f"c_{k}": np.float32(v) for k, v in cs.items()})
    return compute_estimators3(ds, r)
