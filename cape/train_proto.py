"""Exp B: GCN ProtoNet trained on generated episodes; valid proto acc for checkpointing; LR / proto test."""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

from cape.data import load_graph, load_tasks, norm_adj, to_torch_sparse
from cape.generators import N_QRY, N_WAY, episode_stats, load_inputs, make_gen
from cape.pool import log

DEV = "cuda"


class GCN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.l1, self.l2 = nn.Linear(d, 512), nn.Linear(512, 256)
        for l in (self.l1, self.l2):
            nn.init.xavier_uniform_(l.weight); nn.init.zeros_(l.bias)

    def forward(self, X, Ah):
        h = torch.relu(torch.sparse.mm(Ah, torch.sparse.mm(X, self.l1.weight.t())) + self.l1.bias)
        return torch.relu(torch.sparse.mm(Ah, h @ self.l2.weight.t()) + self.l2.bias)


def proto_acc(H, spt, yspt, qry, yqry):
    """H: torch [N,D]; arrays [T,ns], [T,nq]. Prototype = mean support embedding per label."""
    accs = []
    for s, ys, q, yq in zip(spt, yspt, qry, yqry):
        labs = np.unique(ys)
        P = torch.stack([H[s[ys == c]].mean(0) for c in labs])
        pred = labs[torch.cdist(H[q], P).argmin(1).cpu().numpy()]
        accs.append((pred == yq).mean())
    return np.array(accs)


def lr_task(Z, s, ys, q, yq):
    return LogisticRegression(max_iter=1000, C=1e4).fit(Z[s], ys).score(Z[q], yq)


def ci(a):
    return {"mean": float(a.mean() * 100), "ci95": float(1.96 * a.std() / np.sqrt(len(a)) * 100)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ds, arm, seed = args.dataset, args.arm, args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    T, t0 = {}, time.time()

    # episodes (generated up front)
    inp = load_inputs(ds)
    gen = make_gen(arm, inp)
    rng = np.random.default_rng(seed)
    t = time.time()
    eps = [gen(rng) for _ in range(args.episodes)]
    T["generate_s"] = time.time() - t
    spt = np.stack([e[0] for e in eps]); qry = np.stack([e[1] for e in eps])
    stats = episode_stats(spt, qry, inp["y"], inp["split"])
    log(f"{ds}/{arm} generated {args.episodes} eps in {T['generate_s']:.1f}s stats={stats}")

    g = load_graph(ds)
    X = to_torch_sparse(g["X"], DEV)
    Ah = to_torch_sparse(norm_adj(g["A"]), DEV)
    tasks = load_tasks(ds)
    model = GCN(X.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    yq_t = torch.arange(N_WAY, device=DEV).repeat_interleave(N_QRY)
    spt_t = torch.from_numpy(spt).to(DEV); qry_t = torch.from_numpy(qry).to(DEV)

    def valid_acc():
        model.eval()
        with torch.no_grad():
            H = model(X, Ah)
        model.train()
        return float(proto_acc(H, tasks["valid_spt"], tasks["valid_yspt"], tasks["valid_qry"], tasks["valid_yqry"]).mean())

    curve = [(0, valid_acc())]
    best_ep, best_acc, best_state = 0, curve[0][1], copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
    t = time.time(); t_eval = 0.0
    model.train()
    for ep in range(1, args.episodes + 1):
        opt.zero_grad()
        H = model(X, Ah)
        P, Q = H[spt_t[ep - 1]], H[qry_t[ep - 1]]
        loss = nn.functional.cross_entropy(-torch.cdist(Q, P) ** 2, yq_t)
        if not torch.isfinite(loss):
            raise SystemExit(f"NaN/inf loss at episode {ep}")
        loss.backward(); opt.step()
        if ep % args.eval_every == 0:
            te = time.time()
            acc = valid_acc()
            t_eval += time.time() - te
            curve.append((ep, acc))
            if acc > best_acc:
                best_ep, best_acc = ep, acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            log(f"{ds}/{arm} ep {ep} loss {loss.item():.4f} valid {acc:.4f} best {best_acc:.4f}@{best_ep}")
    torch.cuda.synchronize()
    T["train_s"] = time.time() - t; T["valid_eval_s"] = t_eval
    T["step_ms"] = (T["train_s"] - t_eval) / args.episodes * 1000

    # test with best checkpoint
    t = time.time()
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        H = model(X, Ah)
    Z = H.cpu().numpy()
    test = {}
    for k in ("test1", "test5"):
        a = [tasks[f"{k}_{s}"] for s in ("spt", "yspt", "qry", "yqry")]
        lr = np.array(Parallel(n_jobs=args.n_jobs)(delayed(lr_task)(Z, *x) for x in zip(*a)))
        test[f"{k}_lr"] = ci(lr)
        test[f"{k}_proto"] = ci(proto_acc(H, *a))
    T["test_s"] = time.time() - t
    T["total_s"] = time.time() - t0
    res = {"config": {**vars(args), "R_star": inp["R_star"], "gcn": "d-512-256 relu xavier", "lr": 1e-4,
                      "cape_lite_eligible": getattr(gen, "eligible", None)},
           "episode_stats": stats, "valid_curve": curve, "best_episode": best_ep, "best_valid": best_acc,
           "test": test, "time": T, "peak_gpu_GB": torch.cuda.max_memory_allocated() / 2**30}
    out = args.out or f"results/expB/{ds}_{arm}_s{seed}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    log(f"{ds}/{arm} DONE test={test} time={T}")


if __name__ == "__main__":
    main()
