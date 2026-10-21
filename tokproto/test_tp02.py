"""TP02 §2 regression: new default path == TP01 code path (commit c3f6ea7); mean == softmax when tokens are identical."""
import json
import random
import subprocess
import types

import numpy as np
import torch

from cape.data import load_graph, norm_adj, to_torch_sparse
from cape.generators import N_QRY, N_WAY
from cape.train_proto import GCN
from tokproto.loss import tp_loss
from tokproto.tokens import load_candidates, token_sets
from tokproto.train_tp import make_episodes

TOL = 1e-6


def old_module(path, name):
    m = types.ModuleType(name)
    exec(subprocess.check_output(["git", "show", f"c3f6ea7:{path}"]).decode(), m.__dict__)
    return m


def main():
    out = {}
    old_loss, old_tok = old_module("tokproto/loss.py", "old_loss"), old_module("tokproto/tokens.py", "old_tokens")
    seed = 0
    yq = torch.arange(N_WAY, device="cuda").repeat_interleave(N_QRY)
    for ds in ("Amazon_clothing", "Amazon_eletronics", "dblp"):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        z = np.load(f"cache/{ds}/pool.npz")
        spt, qry = make_episodes(z["nbr"], z["cue"], seed, 1)
        tok_n, mk_n = token_sets(load_candidates(ds, "sgc", z["nbr"], z["cue"]), spt, qry, 8)
        tok_o, mk_o = old_tok.token_sets(old_tok.load_candidates(ds, "sgc", z["nbr"], z["cue"]), spt, qry, 8)
        g = load_graph(ds)
        X = to_torch_sparse(g["X"], "cuda"); Ah = to_torch_sparse(norm_adj(g["A"]), "cuda")
        model = GCN(X.shape[1]).cuda()
        with torch.no_grad():
            H = model(X, Ah)
            c = lambda a: torch.from_numpy(a[0]).cuda()
            Q, P = H[c(qry)], H[c(spt)]
            Ln = tp_loss(Q, P, H[c(tok_n)], c(mk_n), 0.25, 0.1, "a2", yq, "softmax").item()
            Lo = old_loss.tp_loss(Q, P, H[c(tok_o)], c(mk_o), 0.25, 0.1, "a2", yq).item()
            # mean vs softmax with identical tokens (all tokens = copy of the first valid token's embedding)
            Ht = H[c(tok_n)][:, :1].expand(-1, 8, -1).contiguous()
            Ls = tp_loss(Q, P, Ht, c(mk_n), 0.25, 0.1, "a2", yq, "softmax").item()
            Lm = tp_loss(Q, P, Ht, c(mk_n), 0.25, 0.1, "a2", yq, "mean").item()
        out[ds] = {"tokens_equal": bool((tok_n == tok_o).all() and (mk_n == mk_o).all()), "loss_new": Ln, "loss_tp01": Lo,
                   "abs_diff": abs(Ln - Lo), "mean_vs_softmax_identical_tokens": abs(Ls - Lm)}
    out["pass"] = all(v["tokens_equal"] and v["abs_diff"] <= TOL and v["mean_vs_softmax_identical_tokens"] <= TOL
                      for k, v in out.items())
    print(json.dumps(out, indent=1))
    json.dump(out, open("results/tp/TP02_regression.json", "w"), indent=1)
    if not out["pass"]:
        raise SystemExit("TP02 regression FAILED")


if __name__ == "__main__":
    main()
