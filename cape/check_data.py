"""T01 data integrity checks and statistics for all three datasets."""
import glob
import json
import os
import time

import numpy as np

from cape.data import DATASETS, _load_pkl, _pool_nodes, load_graph, load_split, load_tasks

# CLAUDE.md §6
EXPECTED = {
    "Amazon_clothing": dict(N=24919, E=91680, d=9034, C=77, base=40, valid=17, test=20),
    "Amazon_eletronics": dict(N=42318, E=43556, d=8669, C=167, base=90, valid=37, test=40),
    "dblp": dict(N=40672, E=288270, d=7202, C=137, base=80, valid=27, test=30),
}
PRE = {  # precomputed reference (homophily, avg degree) and NaQ Table 7 homophily
    "Amazon_clothing": (0.676, 7.36, 0.62),
    "Amazon_eletronics": (0.373, 2.06, 0.38),
    "dblp": (0.258, 14.18, 0.29),
}
TASK_SHAPES = {"valid": (50, 5), "test1": (500, 5), "test5": (500, 25)}


def check_tasks(T, y, split):
    errs = []
    for pre, (n, ns) in TASK_SHAPES.items():
        spt, yspt, qry, yqry = T[f"{pre}_spt"], T[f"{pre}_yspt"], T[f"{pre}_qry"], T[f"{pre}_yqry"]
        for name, a, shp in [("spt", spt, (n, ns)), ("yspt", yspt, (n, ns)), ("qry", qry, (n, 40)), ("yqry", yqry, (n, 40))]:
            if a.shape != shp or a.dtype != np.int64:
                errs.append(f"{pre}_{name} shape/dtype {a.shape}/{a.dtype} != {shp}/int64")
        if errs:
            continue
        allowed = set(split["valid"] if pre == "valid" else split["test"])
        for i in range(n):
            if set(spt[i]) & set(qry[i]):
                errs.append(f"{pre}[{i}] support/query overlap")
            ids = np.concatenate([spt[i], qry[i]])
            ys = np.concatenate([yspt[i], yqry[i]])
            pairs = set(zip(ys.tolist(), y[ids].tolist()))
            if len(pairs) != 5 or len({p[0] for p in pairs}) != 5 or len({p[1] for p in pairs}) != 5:
                errs.append(f"{pre}[{i}] y<->label mapping not bijective over 5 classes: {sorted(pairs)}")
            if not set(y[ids].tolist()) <= allowed:
                errs.append(f"{pre}[{i}] classes outside {'valid' if pre == 'valid' else 'test'} split")
    return errs


def run(ds):
    t0 = time.time()
    g = load_graph(ds)
    split = load_split(ds)
    T = load_tasks(ds)
    A, X, y = g["A"], g["X"], g["y"]
    N = A.shape[0]
    deg = np.asarray(A.sum(1)).reshape(-1)
    r, c = A.nonzero()
    up = r < c
    counts = np.bincount(y)
    p = counts / N
    sp2 = float((p ** 2).sum())
    st = {
        "N": N, "E_undirected": int(up.sum()), "d": X.shape[1],
        "x_nnz_per_row": float(X.nnz / N),
        "n_classes": len(counts), "n_base": len(split["base"]), "n_valid": len(split["valid"]),
        "n_test": len(split["test"]), "valid_per_file": split["valid_per_file"],
        "all_labeled": bool((y >= 0).all()), "labels_contiguous": bool(np.array_equal(np.unique(y), np.arange(len(counts)))),
        "min_class_size": int(counts.min()), "sum_p2": sp2,
        "way_contamination": {n: 1 - (1 - sp2) ** (n - 1) for n in (5, 10, 20)},
        "edge_homophily": float((y[r[up]] == y[c[up]]).mean()),
        "avg_degree": float(deg.mean()), "n_deg0": int((deg == 0).sum()),
        "n_self_loops": int(A.diagonal().sum()),
        "ref_homophily": PRE[ds][0], "ref_avg_degree": PRE[ds][1], "naq_table7_homophily": PRE[ds][2],
    }
    errs = []
    e = EXPECTED[ds]
    got = dict(N=st["N"], E=st["E_undirected"], d=st["d"], C=st["n_classes"], base=st["n_base"], valid=st["n_valid"], test=st["n_test"])
    for k in e:
        if e[k] != got[k]:
            errs.append(f"check1 {k}: got {got[k]} expected {e[k]}")
    base, valid, test, train = set(split["base"]), set(split["valid"]), set(split["test"]), set(g["train_classes"])
    if base & valid: errs.append("check2 base∩valid nonempty")
    if base | valid != train: errs.append("check2 base∪valid != train")
    if test & train: errs.append("check2 test∩train nonempty")
    errs += ["check3 " + s for s in check_tasks(T, y, split)]
    for pth in sorted(glob.glob(f"save/downstream_tasks/{ds}/*.pkl")):
        vp, _ = _load_pkl(pth)
        if not set(np.unique(y[_pool_nodes(vp)]).tolist()) <= valid:
            errs.append(f"check4 {os.path.basename(pth)} valid classes not subset of split")
    st["errors"] = errs
    st["passed"] = not errs
    st["sec"] = time.time() - t0
    return st


def main():
    out = {}
    for ds in DATASETS:
        out[ds] = run(ds)
        print(f"[{ds}] done in {out[ds]['sec']:.1f}s, passed={out[ds]['passed']}", flush=True)
        for e in out[ds]["errors"][:20]:
            print("  ERR", e)
    os.makedirs("results", exist_ok=True)
    with open("results/T01_data_stats.json", "w") as f:
        json.dump(out, f, indent=1)
    cols = ["N", "E_undirected", "d", "x_nnz_per_row", "n_classes", "n_base", "n_valid", "n_test", "min_class_size",
            "sum_p2", "edge_homophily", "ref_homophily", "naq_table7_homophily", "avg_degree", "ref_avg_degree", "n_deg0", "passed"]
    print("| stat | " + " | ".join(DATASETS) + " |")
    print("|---|" + "---|" * len(DATASETS))
    for k in cols:
        vals = [out[ds][k] for ds in DATASETS]
        print(f"| {k} | " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in vals) + " |")
    for n in (5, 10, 20):
        print(f"| way_contam@{n} | " + " | ".join(f"{out[ds]['way_contamination'][n]:.3f}" for ds in DATASETS) + " |")
    print("| valid_per_file | " + " | ".join(str(out[ds]["valid_per_file"]) for ds in DATASETS) + " |")


if __name__ == "__main__":
    main()
