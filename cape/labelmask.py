"""Base-label-scarce mask (CLAUDE.md §6.1). The only T06 module that reads the split; outputs L_idx, y_L only."""
import os

import numpy as np

from cape.data import load_graph, load_split


def mask_path(ds, r):
    return f"cache/{ds}/labelmask_r{r}.npz"


def make_mask(ds, r, seed=0):
    path = mask_path(ds, r)
    if not os.path.exists(path):
        y = load_graph(ds)["y"]
        rng = np.random.default_rng(seed)
        L = []
        for c in load_split(ds)["base"]:
            idx = np.nonzero(y == c)[0]
            k = max(1, int(round(r * len(idx))))
            L.append(rng.choice(idx, k, replace=False))
        L_idx = np.sort(np.concatenate(L)).astype(np.int64)
        np.savez(path, L_idx=L_idx, y_L=y[L_idx])
    z = np.load(path)
    return z["L_idx"], z["y_L"]
