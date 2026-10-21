"""Gb: lambda=0 TP loss == NaQ loss; lambda=1 with anchor-copy tokens == NaQ; empty token sets; real first episode."""
import json
import random

import numpy as np
import torch

from cape.data import load_graph, norm_adj, to_torch_sparse
from cape.generators import N_QRY, N_WAY
from cape.train_proto import GCN
from tokproto.loss import naq_loss, tp_logits, tp_loss
from tokproto.tokens import load_candidates, token_sets
from tokproto.train_tp import make_episodes

TOL, TAU = 1e-5, 0.1


def main():
    out = {}
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    N, D, m = 300, 16, 8
    H = torch.relu(torch.randn(N, D))
    yq = torch.arange(N_WAY).repeat_interleave(N_QRY)
    rng = np.random.default_rng(0)
    d1, d2, d3, ok3 = [], [], [], True
    for e in range(5):
        a = rng.choice(N, N_WAY, replace=False)
        q = rng.choice(N, N_WAY * N_QRY)
        tok = torch.from_numpy(rng.choice(N, (N_WAY, m)))
        mask = torch.ones(N_WAY, m, dtype=torch.bool)
        mask[e % N_WAY] = False                        # one empty way per episode
        mask[(e + 1) % N_WAY, m // 2:] = False         # one partially filled way
        Q, P = H[q], H[a]
        Ln = naq_loss(Q, P, yq).item()
        for neg in ("a2", "b"):
            d1.append(abs(tp_loss(Q, P, H[tok], mask, 0.0, TAU, neg, yq).item() - Ln))
            Ht = P[:, None, :].expand(N_WAY, m, D).clone()   # 2: tokens = copies of the anchor, lambda = 1
            d2.append(abs(tp_loss(Q, P, Ht, torch.ones_like(mask), 1.0, TAU, neg, yq).item() - Ln))
            # 3: empty way -> prototype == anchor, no NaN (also in gradients)
            Hg = H.clone().requires_grad_(True)
            lg = tp_logits(Hg[q], Hg[a], Hg[tok], mask, 0.5, TAU, neg, yq)
            c = e % N_WAY
            d3.append(abs(lg[:, c] + ((Hg[q] - Hg[a][c]) ** 2).sum(-1)).max().item())
            lg.logsumexp(1).sum().backward()
            ok3 &= bool(torch.isfinite(lg).all() and torch.isfinite(Hg.grad).all())
    out["gb1_max_abs_diff"] = max(d1)
    out["gb2_max_abs_diff"] = max(d2)
    out["gb3_max_abs_diff"] = max(d3); out["gb3_finite"] = ok3

    # 4: real input, Amazon_eletronics seed 0 first episode, GCN right after init (same seed order as train_tp)
    ds, seed = "Amazon_eletronics", 0
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    z = np.load(f"cache/{ds}/pool.npz")
    spt, qry = make_episodes(z["nbr"], z["cue"], seed, 1)
    tok, mask = token_sets(load_candidates(ds, "sgc", z["nbr"], z["cue"]), spt, qry, 8)
    g = load_graph(ds)
    X = to_torch_sparse(g["X"], "cuda"); Ah = to_torch_sparse(norm_adj(g["A"]), "cuda")
    model = GCN(X.shape[1]).cuda()
    with torch.no_grad():
        Hr = model(X, Ah)
        yqc = yq.cuda()
        Q, P = Hr[torch.from_numpy(qry[0]).cuda()], Hr[torch.from_numpy(spt[0]).cuda()]
        Ln = naq_loss(Q, P, yqc).item()
        Lt = tp_loss(Q, P, Hr[torch.from_numpy(tok[0]).cuda()], torch.from_numpy(mask[0]).cuda(), 0.0, TAU, "a2", yqc).item()
    out["gb4_naq"], out["gb4_tp"], out["gb4_abs_diff"] = Ln, Lt, abs(Lt - Ln)
    out["gb4_first_anchors"] = spt[0].tolist()
    out["pass"] = {"gb1": out["gb1_max_abs_diff"] <= TOL, "gb2": out["gb2_max_abs_diff"] <= TOL,
                   "gb3": out["gb3_max_abs_diff"] <= TOL and ok3, "gb4": out["gb4_abs_diff"] <= TOL}
    out["all_pass"] = all(out["pass"].values())
    print(json.dumps(out, indent=1))
    json.dump(out, open("results/tp/TP01_Gb.json", "w"), indent=1)
    if not out["all_pass"]:
        raise SystemExit("Gb FAILED")


if __name__ == "__main__":
    main()
