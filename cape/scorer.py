"""Pair scorer (LR / GBDT) trained on base-class pairs only -> cache/{ds}/scores.npz."""
import argparse
import json
import os
import time
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cape.data import load_graph, load_split
from cape.pool import CUES, log

N_ANCHOR = 10000


def fit_lr(Xtr, ytr):
    for it in (2000, 5000):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", ConvergenceWarning)
            lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=it)).fit(Xtr, ytr)
        if not any(issubclass(x.category, ConvergenceWarning) for x in w):
            return lr, it
        log(f"LR ConvergenceWarning at max_iter={it}")
    raise SystemExit("LR did not converge at max_iter=5000")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    ds = args.dataset
    np.random.seed(0)
    out_path = f"cache/{ds}/scores.npz"
    if os.path.exists(out_path) and not args.force:
        log("scores.npz exists, skip (use --force)"); return
    t0 = time.time()
    y = load_graph(ds)["y"]
    split = load_split(ds)
    z = np.load(f"cache/{ds}/pool.npz")
    nbr, cue = z["nbr"], z["cue"]
    N = len(y)

    # training pairs: base anchors, base candidates (base labels only, CLAUDE.md §3)
    is_base = np.isin(y, split["base"])
    V_base = np.nonzero(is_base)[0]
    rng = np.random.default_rng(0)
    anchors = np.sort(rng.choice(V_base, min(N_ANCHOR, len(V_base)), replace=False))
    nb = nbr[anchors]
    m = (nb >= 0) & is_base[np.where(nb >= 0, nb, 0)]
    Xtr = cue[anchors][m].astype(np.float64)
    ytr = (y[np.repeat(anchors, nbr.shape[1]).reshape(nb.shape)][m] == y[nb[m]]).astype(np.int64)
    log(f"{ds} train pairs {len(ytr)}, pos rate {ytr.mean():.4f}")

    t = time.time()
    lr, lr_iter = fit_lr(Xtr, ytr)
    t_lr = time.time() - t
    t = time.time()
    gb = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(Xtr, ytr)
    t_gb = time.time() - t
    log(f"fit lr {t_lr:.1f}s gbdt {t_gb:.1f}s")

    t = time.time()
    valid = nbr >= 0
    Xall = cue[valid].astype(np.float64)
    s_lr = np.full(nbr.shape, np.nan, np.float32)
    s_gb = np.full(nbr.shape, np.nan, np.float32)
    s_lr[valid] = lr.predict_proba(Xall)[:, 1]
    s_gb[valid] = gb.predict_proba(Xall)[:, 1]
    t_inf = time.time() - t
    top = np.sort(np.where(valid, s_lr, -np.inf), axis=1)[:, ::-1][:, :10]
    pi_hat = top.mean(1).astype(np.float32)
    np.savez(out_path, s_lr=s_lr, s_gbdt=s_gb, pi_hat=pi_hat)

    sc, clf = lr[0], lr[1]
    res = {
        "n_train_anchors": int(len(anchors)), "n_train_pairs": int(len(ytr)), "pos_rate": float(ytr.mean()),
        "lr_max_iter_used": lr_iter, "lr_n_iter": int(clf.n_iter_[0]),
        "lr_coef_std": dict(zip(CUES, clf.coef_[0].astype(float).tolist())),
        "lr_intercept": float(clf.intercept_[0]),
        "cue_mean": dict(zip(CUES, sc.mean_.tolist())), "cue_std": dict(zip(CUES, sc.scale_.tolist())),
        "train_auc_lr": float(roc_auc_score(ytr, lr.predict_proba(Xtr)[:, 1])),
        "train_auc_gbdt": float(roc_auc_score(ytr, gb.predict_proba(Xtr)[:, 1])),
        "time_s": {"fit_lr": t_lr, "fit_gbdt": t_gb, "inference": t_inf, "total": time.time() - t0},
    }
    os.makedirs("results/expA", exist_ok=True)
    json.dump(res, open(f"results/expA/{ds}_scorer.json", "w"), indent=1)
    log(json.dumps(res))


if __name__ == "__main__":
    main()
