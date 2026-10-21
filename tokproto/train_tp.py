"""TP training / evaluation. Loop follows cape.train_proto.main (arm naq_feat); only the loss changes."""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
from joblib import Parallel, delayed

from cape.data import load_graph, load_split, load_tasks, norm_adj, to_torch_sparse
from cape.generators import N_QRY, N_WAY, episode_stats, make_topk_gen, top_in_pool
from cape.pool import log
from cape.train_proto import GCN, ci, lr_task, proto_acc
from tokproto.loss import attn_diag, naq_loss, tp_loss
from tokproto.tokens import load_candidates, oracle_filter, token_sets, token_stats

DEV = "cuda"


def arm_name(neg, tok, lam, m, tau=0.1, attn="softmax"):
    if lam == 0:
        return "tp_l0"
    a = f"tp_{neg}_{tok}_l{round(100 * lam)}_m{m}"
    if attn == "mean":
        return a + "_mean"
    return a if tau == 0.1 else a + f"_t{round(100 * tau):02d}"


def make_episodes(nbr, cue, seed, n):
    rng = np.random.default_rng(seed)
    gen = make_topk_gen(top_in_pool(nbr, cue[:, :, 0], N_QRY))
    eps = [gen(rng) for _ in range(n)]
    return np.stack([e[0] for e in eps]), np.stack([e[1] for e in eps])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--neg", choices=["a2", "b"], default="a2")
    ap.add_argument("--tok", choices=["sgc", "pprrow", "sgcora"], default="sgc")
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--tau_att", type=float, default=0.1)
    ap.add_argument("--attn", choices=["softmax", "mean"], default="softmax")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ds, seed, lam = args.dataset, args.seed, args.lam
    arm = arm_name(args.neg, args.tok, lam, args.m, args.tau_att, args.attn)
    analysis_only = args.tok == "sgcora"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    T, t0 = {}, time.time()

    # episodes (generated up front, same as cape.train_proto naq_feat)
    z = np.load(f"cache/{ds}/pool.npz")
    nbr, cue = z["nbr"], z["cue"]
    t = time.time()
    spt, qry = make_episodes(nbr, cue, seed, args.episodes)
    T["generate_s"] = time.time() - t

    g = load_graph(ds)
    y = g["y"]
    stats = episode_stats(spt, qry, y, load_split(ds))   # analysis only
    extra = {}
    if lam > 0:  # tokens: deterministic, no rng
        t = time.time()
        if args.tok == "sgcora":   # TP02 §0 exception: label-filtered tokens (analysis-only upper bound)
            cand = oracle_filter(load_candidates(ds, "sgc", nbr, cue), ds)
        else:                      # no rng, no labels
            cand = load_candidates(ds, args.tok, nbr, cue)
        tok, mask = token_sets(cand, spt, qry, args.m)
        T["tokens_s"] = time.time() - t
        extra["token_stats"] = token_stats(tok, mask, spt, y)   # analysis only
        tok_t = torch.from_numpy(tok).to(DEV); mask_t = torch.from_numpy(mask).to(DEV)
    log(f"{ds}/{arm} s{seed} generated {len(spt)} eps stats={stats} extra={extra}")

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
        if lam == 0:
            loss = naq_loss(Q, P, yq_t)
        else:
            loss = tp_loss(Q, P, H[tok_t[ep - 1]], mask_t[ep - 1], lam, args.tau_att, args.neg, yq_t, args.attn)
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
            log(f"{ds}/{arm} s{seed} ep {ep} loss {loss.item():.4f} valid {acc:.4f} best {best_acc:.4f}@{best_ep}")
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

    if lam > 0:  # attention diagnostics at best checkpoint, first 200 training episodes (labels: analysis only)
        y_t = torch.from_numpy(y).to(DEV)
        acc = {}
        for e in range(min(200, args.episodes)):
            d = attn_diag(H[qry_t[e]], H[tok_t[e]], mask_t[e], args.tau_att, yq_t,
                          y_t[tok_t[e]], y_t[spt_t[e]], y_t[qry_t[e]], args.attn)
            for k, v in d.items():
                acc.setdefault(k, []).extend(v)
        extra["attn_diag"] = {k: float(np.mean(v)) if v else None for k, v in acc.items()}
        extra["attn_diag"]["n_queries"] = len(acc["mass_anchor_label"])
    T["total_s"] = time.time() - t0
    res = {"config": {**vars(args), "arm": arm, "gcn": "d-512-256 relu xavier", "lr": 1e-4},
           "attn": args.attn, "analysis_only": analysis_only, "episode_stats": stats, **extra, "valid_curve": curve, "best_episode": best_ep, "best_valid": best_acc,
           "test": test, "time": T, "peak_gpu_GB": torch.cuda.max_memory_allocated() / 2**30}
    out = args.out or f"results/tp/{ds}_{arm}_s{seed}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    log(f"{ds}/{arm} s{seed} DONE test={test} extra={extra} time={T}")


if __name__ == "__main__":
    main()
