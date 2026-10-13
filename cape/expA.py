"""Exp A: generator diagnostics (purity, class-level sim, calibration, selective purity, G1)."""
import argparse
import json
import math
import os
import time

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from cape.data import DATASETS, load_graph, load_split
from cape.pool import CUES, log

RANKERS = ["raw", "sgc", "ppr", "lr", "gbdt"]
SINGLE = ["raw", "ppr", "sgc"]
GROUPS = ["base", "valid", "test", "all"]
QS = [5, 10, 20]
NAQ_CLS_SIM = {"Amazon_clothing": {"raw": 0.816, "ppr": 0.788}, "Amazon_eletronics": {"raw": 0.819, "ppr": 0.527},
               "dblp": {"raw": 0.735, "ppr": 0.830}}


def top_ids(nbr, score, k):
    sc = np.where(nbr >= 0, score, -np.inf)
    o = np.argsort(-sc, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(nbr, o, 1)


def ece(p, y, bins=10):
    b = np.minimum((p * bins).astype(int), bins - 1)
    e = 0.0
    for i in range(bins):
        m = b == i
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def run(ds):
    t0 = time.time()
    g = load_graph(ds)
    y, X = g["y"], g["X"]
    split = load_split(ds)
    z = np.load(f"cache/{ds}/pool.npz"); nbr, cue = z["nbr"], z["cue"]
    s = np.load(f"cache/{ds}/scores.npz"); s_lr, s_gb, pi_hat = s["s_lr"], s["s_gbdt"], s["pi_hat"]
    N = len(y)
    score = {"raw": cue[:, :, 0], "sgc": cue[:, :, 1], "ppr": cue[:, :, 2], "lr": s_lr, "gbdt": s_gb}
    grp = {"base": np.isin(y, split["base"]), "valid": np.isin(y, split["valid"]),
           "test": np.isin(y, split["test"]), "all": np.ones(N, bool)}

    # per-anchor purity@Q and class-level similarity@10
    C = y.max() + 1
    cnt = np.bincount(y, minlength=C)
    cen = np.asarray((__import__("scipy.sparse", fromlist=["x"]).csr_matrix(
        (np.ones(N), (y, np.arange(N))), shape=(C, N)) @ X).todense()) / cnt[:, None]
    cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    csim = cen @ cen.T
    pur, clsim, tops = {}, {}, {}
    for r in RANKERS:
        t20 = top_ids(nbr, score[r], 20)
        tops[r] = t20
        same = y[t20] == y[:, None]
        pur[r] = {q: same[:, :q].mean(1) for q in QS}
        clsim[r] = csim[y[:, None], y[t20[:, :10]]].mean(1)
    res = {"purity": {r: {g_: {q: float(pur[r][q][grp[g_]].mean()) for q in QS} for g_ in GROUPS} for r in RANKERS}}
    res["class_sim"] = {r: {g_: {"class_sim@10": float(clsim[r][grp[g_]].mean()),
                                 "purity@10": float(pur[r][10][grp[g_]].mean())} for g_ in ("all", "test")}
                        for r in ["raw", "ppr", "sgc", "lr"]}
    res["naq_class_sim_reported"] = NAQ_CLS_SIM[ds]

    # discrimination / calibration over all pool pairs of group anchors
    valid = nbr >= 0
    same_all = (y[np.where(valid, nbr, 0)] == y[:, None])
    res["calib"] = {}
    for g_ in ("base", "test"):
        m = valid & grp[g_][:, None]
        yy = same_all[m].astype(int)
        res["calib"][g_] = {"n_pairs": int(m.sum()), "pos_rate": float(yy.mean())}
        for name, sc in (("lr", s_lr), ("gbdt", s_gb)):
            p = sc[m].astype(np.float64)
            res["calib"][g_][f"auc_{name}"] = float(roc_auc_score(yy, p))
            res["calib"][g_][f"ece_{name}"] = ece(p, yy)
    res["calib_gap"] = {g_: float(pi_hat[grp[g_]].mean() - pur["lr"][10][grp[g_]].mean())
                        for g_ in ("base", "valid", "test")}
    res["pi_hat_mean"] = {g_: float(pi_hat[grp[g_]].mean()) for g_ in ("base", "valid", "test")}
    res["spearman"] = {g_: float(spearmanr(pi_hat[grp[g_]], pur["lr"][10][grp[g_]]).correlation)
                       for g_ in ("test", "valid")}

    # selective purity on test anchors
    ti = np.nonzero(grp["test"])[0]
    order = ti[np.argsort(-pi_hat[ti], kind="stable")]
    perm = np.random.default_rng(0).permutation(ti)
    res["selective"] = {}
    for f in (1.0, 0.7, 0.5):
        k = int(round(f * len(ti)))
        res["selective"][str(f)] = {"by_pi_hat": float(pur["lr"][10][order[:k]].mean()),
                                    "random": float(pur["lr"][10][perm[:k]].mean()), "n": k}

    # tail test classes (smallest 20%, ceil)
    tc = np.array(split["test"])
    nt = math.ceil(0.2 * len(tc))
    tail = tc[np.argsort(cnt[tc], kind="stable")[:nt]]
    tm = np.isin(y, tail)
    res["tail"] = {"classes": tail.tolist(), "sizes": cnt[tail].tolist(), "n_anchors": int(tm.sum()),
                   "purity@10": {r: float(pur[r][10][tm].mean()) for r in RANKERS}}

    # best single cue and G1
    p10 = lambda r, g_: res["purity"][r][g_][10]
    R = max(SINGLE, key=lambda r: p10(r, "test"))
    Rv = max(SINGLE, key=lambda r: p10(r, "valid"))
    res["R_star"] = R
    res["G1"] = {"R_star": R, "purity10_Rstar": p10(R, "test"), "purity10_lr": p10("lr", "test"),
                 "purity10_gbdt": p10("gbdt", "test"),
                 "delta_lr": p10("lr", "test") - p10(R, "test"), "delta_gbdt": p10("gbdt", "test") - p10(R, "test"),
                 "delta_lr_valid_Rstar": p10("lr", "valid") - p10(R, "valid"),
                 "R_star_valid": Rv, "delta_lr_valid_bestvalid": p10("lr", "valid") - p10(Rv, "valid"),
                 "spearman_test": res["spearman"]["test"]}
    res["G1"]["pass_delta"] = bool(res["G1"]["delta_lr"] >= 0.03)
    res["G1"]["pass_spearman"] = bool(res["G1"]["spearman_test"] >= 0.3)
    res["way_contamination"] = {n: 1 - (1 - float(((cnt / N) ** 2).sum())) ** (n - 1) for n in (5, 10, 20)}
    res["lr_coef_std"] = json.load(open(f"results/expA/{ds}_scorer.json"))["lr_coef_std"]
    res["time_s"] = time.time() - t0
    os.makedirs("results/expA", exist_ok=True)
    json.dump(res, open(f"results/expA/{ds}.json", "w"), indent=1)
    log(f"{ds} expA done {res['time_s']:.1f}s G1={res['G1']}")


def f3(x):
    return f"{x:.3f}"


def summarize():
    R = {ds: json.load(open(f"results/expA/{ds}.json")) for ds in DATASETS}
    L = ["# Exp A summary", "", "## (a) G1", "",
         "| dataset | R* | p@10(R*) | p@10(lr) | Δ_lr | Δ_gbdt | Δ_lr(valid, same R*) | Spearman(test) | Δ_lr≥0.03 | ρ≥0.3 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    npass = 0
    for ds, r in R.items():
        g = r["G1"]; npass += g["pass_delta"]
        L.append(f"| {ds} | {g['R_star']} | {f3(g['purity10_Rstar'])} | {f3(g['purity10_lr'])} | {g['delta_lr']:+.3f} | "
                 f"{g['delta_gbdt']:+.3f} | {g['delta_lr_valid_Rstar']:+.3f} | {f3(g['spearman_test'])} | "
                 f"{'PASS' if g['pass_delta'] else 'FAIL'} | {'PASS' if g['pass_spearman'] else 'FAIL'} |")
    L += ["", f"G1 Δ_lr 조건 통과: **{npass} / 3** (기준 2개 이상)", "",
          "## (b) purity@10", "", "| dataset | group | " + " | ".join(RANKERS) + " |", "|---|---|" + "---|" * len(RANKERS)]
    for ds, r in R.items():
        for g_ in GROUPS:
            L.append(f"| {ds} | {g_} | " + " | ".join(f3(r["purity"][k][g_]["10"]) for k in RANKERS) + " |")
    L += ["", "## (c) class-level similarity@10 vs exact purity@10", "",
          "| dataset | group | ranker | class-sim | purity@10 | NaQ reported class-sim |", "|---|---|---|---|---|---|"]
    for ds, r in R.items():
        for g_ in ("all", "test"):
            for k in ("raw", "ppr", "sgc", "lr"):
                c = r["class_sim"][k][g_]
                naq = r["naq_class_sim_reported"].get(k) if g_ == "all" else None
                L.append(f"| {ds} | {g_} | {k} | {f3(c['class_sim@10'])} | {f3(c['purity@10'])} | {f3(naq) if naq else '-'} |")
    L += ["", "## (d) calibration", "",
          "| dataset | AUC lr base | AUC lr test | AUC gbdt base | AUC gbdt test | ECE lr base | ECE lr test | ECE gbdt base | ECE gbdt test | gap base | gap valid | gap test |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds, r in R.items():
        c, gp = r["calib"], r["calib_gap"]
        L.append(f"| {ds} | " + " | ".join(f3(c[g_][f"{m}_{k}"]) for m in ("auc", "ece") for k in ("lr", "gbdt") for g_ in ("base", "test"))
                 + " | " + " | ".join(f"{gp[g_]:+.3f}" for g_ in ("base", "valid", "test")) + " |")
    L += ["", "gap = mean(π̂) − mean(purity@10_lr). AUC/ECE over all pool pairs of the group's anchors.", "",
          "## (e) selective purity@10_lr (test anchors)", "",
          "| dataset | f=1.0 | f=0.7 π̂ | f=0.7 random | f=0.5 π̂ | f=0.5 random |", "|---|---|---|---|---|---|"]
    for ds, r in R.items():
        s = r["selective"]
        L.append(f"| {ds} | {f3(s['1.0']['by_pi_hat'])} | {f3(s['0.7']['by_pi_hat'])} | {f3(s['0.7']['random'])} | "
                 f"{f3(s['0.5']['by_pi_hat'])} | {f3(s['0.5']['random'])} |")
    L += ["", "## (f) LR standardized coefficients", "", "| dataset | " + " | ".join(CUES) + " |", "|---|" + "---|" * len(CUES)]
    for ds, r in R.items():
        L.append(f"| {ds} | " + " | ".join(f"{r['lr_coef_std'][c]:+.3f}" for c in CUES) + " |")
    L += ["", "## (g) tail test classes (smallest 20%) purity@10", "",
          "| dataset | #classes | #anchors | " + " | ".join(RANKERS) + " |", "|---|---|---|" + "---|" * len(RANKERS)]
    for ds, r in R.items():
        t = r["tail"]
        L.append(f"| {ds} | {len(t['classes'])} | {t['n_anchors']} | " + " | ".join(f3(t["purity@10"][k]) for k in RANKERS) + " |")
    open("results/expA/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize()
    else:
        run(args.dataset)


if __name__ == "__main__":
    main()
