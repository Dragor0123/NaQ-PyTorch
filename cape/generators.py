"""Episode generators for Exp B. Each gen(rng) -> (id_spt[5], id_qry[50]); way i = spt[i], qry[10i:10i+10].

Only `oracle` / `oracle_base` read labels while generating (oracle_base: base labels only, CLAUDE.md §3).
"""
import json

import numpy as np

from cape.data import load_graph, load_split

N_WAY, N_QRY = 5, 10
ARMS = ["naq_feat", "naq_diff", "lr_only", "cape_lite", "oracle", "oracle_base"]
T05_ARMS = ["naq_unl", "lr_soft", "naq_hard", "naq_easy"]
NAQ_BEST_COL = {"Amazon_clothing": 0, "Amazon_eletronics": 0, "dblp": 2}  # NaQ-best ranker from T03 seed 0: raw / ppr
SCORE_COL = {"raw": 0, "sgc": 1, "ppr": 2}


def top_in_pool(nbr, score, k):
    sc = np.where(nbr >= 0, score, -np.inf)
    o = np.argsort(-sc, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(nbr, o, 1)


def load_inputs(ds):
    z = np.load(f"cache/{ds}/pool.npz")
    s = np.load(f"cache/{ds}/scores.npz")
    g = load_graph(ds)
    return {"ds": ds, "nbr": z["nbr"], "cue": z["cue"], "s_lr": s["s_lr"], "pi_hat": s["pi_hat"], "y": g["y"],
            "split": load_split(ds), "R_star": json.load(open(f"results/expA/{ds}.json"))["R_star"]}


def make_topk_gen(top10, anchor_pool=None):
    N = len(top10)

    def gen(rng):
        a = rng.choice(N, N_WAY, replace=False) if anchor_pool is None else rng.choice(anchor_pool, N_WAY, replace=False)
        return a, top10[a].reshape(-1)
    return gen


def make_cape_lite(nbr, s_lr, pi_hat, top10_lr, max_tries=1000):
    E = np.nonzero(pi_hat >= np.percentile(pi_hat, 30))[0]

    def gen(rng):
        used, anchors, tries = set(), [], 0
        while len(anchors) < N_WAY:
            tries += 1
            if tries > max_tries:
                raise RuntimeError("cape_lite: >1000 tries for one episode")
            a = int(E[rng.integers(len(E))])
            if a in used:                                   # (i)
                continue
            q = top10_lr[a]
            if any(int(v) in used for v in q):             # (ii)
                continue
            ok = True
            for b in anchors:                               # (iii)
                j = np.nonzero(nbr[a] == b)[0]
                if len(j) and s_lr[a, j[0]] >= 0.5:
                    ok = False
                    break
            if not ok:
                continue
            anchors.append(a)
            used.add(a); used.update(int(v) for v in q)
        a = np.array(anchors)
        return a, top10_lr[a].reshape(-1)
    gen.eligible = len(E)
    return gen


def make_oracle(nbr, score, y, anchor_pool):
    """Distinct-label anchors drawn uniformly from anchor_pool; query = same-label pool nodes by R*, random fill."""
    same = np.where(nbr >= 0, y[np.maximum(nbr, 0)] == y[:, None], False)
    sc = np.where(same, score, -np.inf)
    o = np.argsort(-sc, axis=1, kind="stable")[:, :N_QRY]
    top = np.where(np.take_along_axis(sc, o, 1) > -np.inf, np.take_along_axis(nbr, o, 1), -1)
    by_class = {int(c): np.nonzero(y == c)[0] for c in np.unique(y)}

    def gen(rng):
        anchors, labs = [], set()
        while len(anchors) < N_WAY:
            a = int(anchor_pool[rng.integers(len(anchor_pool))])
            if int(y[a]) not in labs:
                anchors.append(a); labs.add(int(y[a]))
        qs = []
        for a in anchors:
            q = [int(v) for v in top[a] if v >= 0]
            if len(q) < N_QRY:
                cand = np.setdiff1d(by_class[int(y[a])], np.array(q + [a]))
                q += rng.choice(cand, N_QRY - len(q), replace=False).tolist()
            qs.append(q)
        return np.array(anchors), np.array(qs).reshape(-1)
    return gen


# --- T06: label-scarce anchoring. Inputs restricted to (pool-derived top10, L_idx, y_L, novelty); no split access.
def make_unl_uni(top10, L_idx, y_L, novelty=None):
    U = np.setdiff1d(np.arange(len(top10)), L_idx)
    return make_topk_gen(top10, U)


def make_nov(top10, L_idx, y_L, novelty):
    U = np.setdiff1d(np.arange(len(top10)), L_idx)
    k = int(np.ceil(len(U) / 2))                               # top 50% novelty within U (fixed)
    A_nov = np.sort(U[np.argsort(-novelty[U], kind="stable")[:k]])
    return make_topk_gen(top10, A_nov)


# --- T07: proportional anchoring with p_E*(novel|u)^tau, and mixing with supervised episodes from L.
def make_nov2(top10, L_idx, y_L, weights):
    U = np.setdiff1d(np.arange(len(top10)), L_idx)
    w = np.clip(weights[U].astype(np.float64), 0, None)
    if w.sum() <= 0:
        raise RuntimeError("nov2: total anchor weight over U is 0")
    U, w = U[w > 0], w[w > 0]
    p = w / w.sum()

    def gen(rng):
        a = rng.choice(U, N_WAY, replace=False, p=p)
        return a, top10[a].reshape(-1)
    return gen


def make_mix_alt(top10, L_idx, y_L, weights, sup_frac=0.3):
    nov = make_nov2(top10, L_idx, y_L, weights)
    by_c = {int(c): L_idx[y_L == c] for c in np.unique(y_L)}
    cls = np.array([c for c, v in by_c.items() if len(v) >= 1 + N_QRY])

    def gen(rng):
        if rng.random() >= sup_frac:
            return nov(rng)
        cs = rng.choice(cls, N_WAY, replace=False)
        picks = [rng.choice(by_c[int(c)], 1 + N_QRY, replace=False) for c in cs]
        return np.array([p[0] for p in picks]), np.concatenate([p[1:] for p in picks])
    return gen


def make_scarce_gen(arm, ds, nbr, cue):
    import json
    from cape.labelmask import make_mask
    from cape.novelty import load_novelty
    r = float(arm.rsplit("_r", 1)[1])
    L_idx, y_L = make_mask(ds, r)
    top10 = top_in_pool(nbr, cue[:, :, NAQ_BEST_COL[ds]], N_QRY)
    if arm.startswith(("nov2_", "mix_alt_r")):
        from cape.novelty2 import compute_estimators
        E = json.load(open("results/T07A_estimators.json"))["E_star"]
        p_nov = compute_estimators(ds, r)[E]
        if arm.startswith("mix_alt_r"):
            return make_mix_alt(top10, L_idx, y_L, p_nov)
        tau = int(arm.split("_")[1][1:])
        return make_nov2(top10, L_idx, y_L, np.clip(p_nov, 0, None) ** tau)
    if arm.startswith("unl_uni_r"):
        return make_unl_uni(top10, L_idx, y_L)
    return make_nov(top10, L_idx, y_L, load_novelty(ds, r, L_idx, y_L)["sgc"])


def make_gen(arm, inp):
    nbr, cue, s_lr, y = inp["nbr"], inp["cue"], inp["s_lr"], inp["y"]
    if arm.startswith(("unl_uni_r", "nov_r", "nov2_", "mix_alt_r")):
        return make_scarce_gen(arm, inp["ds"], nbr, cue)
    if arm == "naq_feat":
        return make_topk_gen(top_in_pool(nbr, cue[:, :, 0], N_QRY))
    if arm == "naq_diff":
        return make_topk_gen(top_in_pool(nbr, cue[:, :, 2], N_QRY))
    if arm in ("naq_best", "naq_unl"):  # T05; naq_unl uses labeled-node identity (T05 §0 exception)
        top10 = top_in_pool(nbr, cue[:, :, NAQ_BEST_COL[inp["ds"]]], N_QRY)
        if arm == "naq_best":
            return make_topk_gen(top10)
        return make_topk_gen(top10, np.nonzero(~np.isin(y, inp["split"]["base"]))[0])
    top10_lr = top_in_pool(nbr, s_lr, N_QRY)
    if arm in ("lr_only", "lr_soft"):
        return make_topk_gen(top10_lr)
    if arm == "cape_lite":
        return make_cape_lite(nbr, s_lr, inp["pi_hat"], top10_lr)
    rscore = cue[:, :, SCORE_COL[inp["R_star"]]]
    if arm == "oracle":
        return make_oracle(nbr, rscore, y, np.arange(len(y)))
    if arm == "oracle_base":
        return make_oracle(nbr, rscore, y, np.nonzero(np.isin(y, inp["split"]["base"]))[0])
    raise ValueError(arm)


def episode_stats(spt, qry, y, split):
    """spt [E,5], qry [E,50]. Labels used for analysis only."""
    E = len(spt)
    ya = y[spt]                                             # [E,5]
    yq = y[qry].reshape(E, N_WAY, N_QRY)
    purity = float((yq == ya[:, :, None]).mean())
    coll = np.array([[(ya[e] == ya[e, i]).sum() > 1 for i in range(N_WAY)] for e in range(E)])
    q3 = qry.reshape(E, N_WAY, N_QRY)
    ov = []
    for e in range(E):
        ways_of = {}
        for i in range(N_WAY):
            for v in set(q3[e, i].tolist()):
                ways_of[v] = ways_of.get(v, 0) + 1
        ov.append(np.mean([ways_of[v] > 1 for v in q3[e].reshape(-1).tolist()]))
    C = int(y.max()) + 1
    p = np.bincount(ya.reshape(-1), minlength=C) / ya.size
    p = p[p > 0]
    return {
        "query_purity": purity,
        "way_collision": float(coll.mean()),
        "query_overlap": float(np.mean(ov)),
        "anchor_unique_frac": float(len(np.unique(spt)) / spt.size),
        "anchor_class_entropy": float(-(p * np.log(p)).sum() / np.log(C)),
        "anchor_group_frac": {gname: float(np.isin(ya, split[gname]).mean()) for gname in ("base", "valid", "test")},
        "anchor_novel_frac": float(np.isin(ya, split["valid"] + split["test"]).mean()),
    }
