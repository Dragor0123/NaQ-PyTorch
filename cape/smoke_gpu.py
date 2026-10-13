"""T01 GPU timing: GCN ProtoNet step, pool-block (cos_raw, PPR) cost, X_sgc build."""
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn

from cape.data import load_graph, norm_adj, to_torch_sparse

DEV = "cuda"
GB = 1024 ** 3


def seed_all(s=0):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def timed(fn):
    torch.cuda.synchronize(); t = time.time(); out = fn(); torch.cuda.synchronize()
    return out, time.time() - t


class GCN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.l1, self.l2 = nn.Linear(d, 512), nn.Linear(512, 256)
        for l in (self.l1, self.l2):
            nn.init.xavier_uniform_(l.weight); nn.init.zeros_(l.bias)

    def forward(self, X, Ah):
        h = torch.relu(torch.sparse.mm(Ah, torch.sparse.mm(X, self.l1.weight.t())) + self.l1.bias)
        return torch.relu(torch.sparse.mm(Ah, h @ self.l2.weight.t()) + self.l2.bias)


def proto_loss(H, spt, qry):
    p, q = H[spt], H[qry]                      # [5,256], [50,256]
    logits = -torch.cdist(q, p) ** 2
    yq = torch.arange(5, device=H.device).repeat_interleave(10)
    return nn.functional.cross_entropy(logits, yq)


def time_gcn(X, Ah, N):
    torch.cuda.reset_peak_memory_stats()
    model = GCN(X.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    rng = np.random.default_rng(0)

    def step():
        ids = torch.from_numpy(rng.choice(N, 55, replace=False)).to(DEV)
        model.train(); opt.zero_grad()
        loss = proto_loss(model(X, Ah), ids[:5], ids[5:])
        loss.backward(); opt.step()
        return loss

    for _ in range(5):
        step()
    _, t = timed(lambda: [step() for _ in range(50)])
    step_ms = t / 50 * 1000
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(X, Ah)
        _, tf = timed(lambda: [model(X, Ah) for _ in range(20)])
    fwd_ms = tf / 20 * 1000
    return {"step_ms": step_ms, "fwd_ms": fwd_ms, "peak_mem_GB": torch.cuda.max_memory_allocated() / GB,
            "est_16k_train_min": (step_ms * 16000 + fwd_ms * 40) / 60000,
            "note": "excludes test LR probing time"}


def time_blocks(Xs, Ah, N, B):
    res = {}
    # cos_raw block
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    Xd = Xs.to_dense()
    Xd = Xd / Xd.norm(dim=1, keepdim=True).clamp_min(1e-12)
    blk = lambda: torch.topk(Xd @ Xd[:B].t(), 51, dim=0)
    blk()
    _, t = timed(blk)
    nb = math.ceil(N / B)
    res["cos_raw"] = {"block": B, "block_s": t, "n_blocks": nb, "est_total_s": t * nb,
                      "peak_mem_GB": torch.cuda.max_memory_allocated() / GB}
    del Xd
    # PPR block: Pi[:, cols] = alpha * sum_k (1-alpha)^k Ah^k e_cols, K=50 (51 terms)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    alpha = 0.1

    def ppr():
        cols = torch.arange(B, device=DEV)
        Z = torch.zeros(N, B, device=DEV); Z[cols, cols] = 1.0
        acc = alpha * Z
        for k in range(1, 51):
            Z = torch.sparse.mm(Ah, Z)
            acc = acc + alpha * (1 - alpha) ** k * Z
        return torch.topk(acc, 51, dim=0)

    ppr()
    _, t = timed(ppr)
    res["ppr"] = {"block": B, "block_s": t, "n_blocks": nb, "est_total_s": t * nb,
                  "peak_mem_GB": torch.cuda.max_memory_allocated() / GB}
    # X_sgc = Ah^2 X (dense)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    def sgc():
        return torch.sparse.mm(Ah, torch.sparse.mm(Ah, Xs).to_dense())

    Xg, t = timed(sgc)
    res["x_sgc"] = {"build_s": t, "tensor_GB": Xg.numel() * 4 / GB,
                    "peak_mem_GB": torch.cuda.max_memory_allocated() / GB,
                    "peak_over_inputs_GB": (torch.cuda.max_memory_allocated() - base) / GB}
    del Xg
    return res


def run(ds, B=2048):
    seed_all(0)
    t0 = time.time()
    g = load_graph(ds)
    N = g["A"].shape[0]
    X = to_torch_sparse(g["X"], DEV)
    Ah = to_torch_sparse(norm_adj(g["A"]), DEV)
    print(f"[{ds}] load+upload {time.time() - t0:.1f}s", flush=True)
    out = {"N": N, "gcn": time_gcn(X, Ah, N)}
    print(f"[{ds}] gcn {out['gcn']}", flush=True)
    try:
        out["blocks"] = time_blocks(X, Ah, N, B)
    except torch.cuda.OutOfMemoryError:
        print(f"[{ds}] OOM at block {B}, retry 1024", flush=True)
        torch.cuda.empty_cache()
        out["blocks"] = time_blocks(X, Ah, N, 1024)
    print(f"[{ds}] blocks {out['blocks']}", flush=True)
    out["total_s"] = time.time() - t0
    return out


def main():
    res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    for ds in ["Amazon_eletronics", "dblp"]:
        res[ds] = run(ds)
        torch.cuda.empty_cache()
    os.makedirs("results", exist_ok=True)
    with open("results/T01_timing.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
