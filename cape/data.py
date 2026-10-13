"""Graph / feature / label loading, class split, and fixed NaQ evaluation tasks."""
import glob
import json
import os
import pickle

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import torch

DATASETS = ["Amazon_clothing", "Amazon_eletronics", "dblp"]


def load_graph(ds, root="data"):
    edges = np.loadtxt(os.path.join(root, f"{ds}_network"), dtype=np.int64, delimiter="\t")
    N = int(edges.max()) + 1
    r, c = edges[:, 0], edges[:, 1]
    keep = r != c
    r, c = r[keep], c[keep]
    A = sp.coo_matrix((np.ones(2 * len(r), dtype=np.float32),
                       (np.concatenate([r, c]), np.concatenate([c, r]))), shape=(N, N)).tocsr()
    A.data[:] = 1.0  # binarize (removes duplicate multiplicity)
    A.eliminate_zeros()

    tr = sio.loadmat(os.path.join(root, f"{ds}_train.mat"))
    te = sio.loadmat(os.path.join(root, f"{ds}_test.mat"))
    d = tr["Attributes"].shape[1]
    y = np.full(N, -1, dtype=np.int64)
    rows, mats = [], []
    count = np.zeros(N, dtype=np.int64)
    for m in (tr, te):
        idx = np.asarray(m["Index"]).reshape(-1).astype(np.int64)
        y[idx] = np.asarray(m["Label"]).reshape(-1)
        np.add.at(count, idx, 1)
        rows.append(idx)
        mats.append(sp.csr_matrix(m["Attributes"], dtype=np.float32))
    assert (count == 1).all(), f"{ds}: nodes without exactly one feature/label: {(count != 1).sum()}"
    assert (y >= 0).all()
    idx = np.concatenate(rows)
    Xs = sp.vstack(mats).tocsr()
    perm = sp.csr_matrix((np.ones(N, dtype=np.float32), (idx, np.arange(N))), shape=(N, N))
    X = (perm @ Xs).tocsr().astype(np.float32)  # row idx[i] <- Xs row i
    return {
        "A": A, "X": X, "y": y,
        "train_classes": sorted(int(k) for k in np.unique(np.asarray(tr["Label"]).reshape(-1))),
        "test_classes": sorted(int(k) for k in np.unique(np.asarray(te["Label"]).reshape(-1))),
    }


def norm_adj(A):
    A = A + sp.eye(A.shape[0], dtype=np.float32, format="csr")
    deg = np.asarray(A.sum(1)).reshape(-1)
    dinv = sp.diags(deg ** -0.5)
    return (dinv @ A @ dinv).tocsr().astype(np.float32)


def to_torch_sparse(mat, device):
    coo = mat.tocoo()
    idx = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64))
    val = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, coo.shape).coalesce().to(device)


def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _pool_nodes(pool):
    out = []
    for task in pool:
        out.append(np.asarray(task[0]).reshape(-1))
        out.append(np.asarray(task[2]).reshape(-1))
    return np.concatenate(out).astype(np.int64)


def load_split(ds, root="data", cache="cache", save="save/downstream_tasks"):
    path = os.path.join(cache, ds, "split.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    g = load_graph(ds, root)
    y = g["y"]
    valid, per_file = set(), {}
    for p in sorted(glob.glob(os.path.join(save, ds, "*.pkl"))):
        valid_pool, _ = _load_pkl(p)
        cls = set(int(k) for k in np.unique(y[_pool_nodes(valid_pool)]))
        per_file[os.path.basename(p)] = len(cls)
        valid |= cls
    split = {
        "base": sorted(set(g["train_classes"]) - valid),
        "valid": sorted(valid),
        "test": list(g["test_classes"]),
        "valid_per_file": per_file,
        "source": f"union of NaQ save/downstream_tasks/{ds}/*.pkl valid pools",
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=1)
    return split


def _stack(pool, prefix):
    to_np = lambda a: (a.numpy() if torch.is_tensor(a) else np.asarray(a)).astype(np.int64).reshape(-1)
    return {
        f"{prefix}_spt": np.stack([to_np(t[0]) for t in pool]),
        f"{prefix}_yspt": np.stack([to_np(t[1]) for t in pool]),
        f"{prefix}_qry": np.stack([to_np(t[2]) for t in pool]),
        f"{prefix}_yqry": np.stack([to_np(t[3]) for t in pool]),
    }


def load_tasks(ds, cache="cache", save="save/downstream_tasks"):
    path = os.path.join(cache, ds, "tasks.npz")
    if not os.path.exists(path):
        v1, t1 = _load_pkl(os.path.join(save, ds, "5way_1shot.pkl"))
        _, t5 = _load_pkl(os.path.join(save, ds, "5way_5shot.pkl"))
        out = {**_stack(v1, "valid"), **_stack(t1, "test1"), **_stack(t5, "test5")}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, **out)
    z = np.load(path)
    return {k: z[k] for k in z.files}
