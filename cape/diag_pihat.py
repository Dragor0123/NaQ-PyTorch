"""T05 A: what does pi_hat reject? (CPU, labels for analysis only) -> results/T05A_diag.{json,md}"""
import json
import math
import time

import numpy as np
from scipy.stats import spearmanr

from cape.data import DATASETS, load_graph, load_split
from cape.generators import top_in_pool
from cape.pool import log

GROUPS = ["base", "valid", "test"]


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if x.sum() > 0 else 0.0


def run(ds):
    g = load_graph(ds); split = load_split(ds)
    A, y = g["A"], g["y"]
    N = len(y)
    z = np.load(f"cache/{ds}/pool.npz"); s = np.load(f"cache/{ds}/scores.npz")
    pi = s["pi_hat"]
    deg = np.asarray(A.sum(1)).reshape(-1)
    r, c = A.nonzero()
    diff = np.bincount(r, weights=(y[r] != y[c]).astype(float), minlength=N)
    boundary = np.where(deg > 0, diff / np.maximum(deg, 1), np.nan)
    cnt = np.bincount(y)
    csize = cnt[y]
    grp = np.full(N, "", object)
    tail_cls = set()
    for gname in GROUPS:
        cl = np.array(split[gname])
        grp[np.isin(y, cl)] = gname
        nt = math.ceil(0.2 * len(cl))
        tail_cls |= set(cl[np.argsort(cnt[cl], kind="stable")[:nt]].tolist())
    tail = np.isin(y, list(tail_cls))
    t10 = top_in_pool(z["nbr"], s["s_lr"], 10)
    pur = (y[t10] == y[:, None]).mean(1)

    res = {"spearman": {k: float(spearmanr(pi, v, nan_policy="omit").correlation)
                        for k, v in (("deg", deg), ("boundary", boundary), ("class_size", csize))}}
    # deciles by pi_hat rank (1 = lowest)
    dec = np.minimum((np.argsort(np.argsort(pi, kind="stable")) * 10) // N, 9)
    res["deciles"] = [{"decile": d + 1, "pi_hat": float(pi[dec == d].mean()), "deg": float(deg[dec == d].mean()),
                       "boundary": float(np.nanmean(boundary[dec == d])), "test_frac": float((grp[dec == d] == "test").mean()),
                       "tail_frac": float(tail[dec == d].mean()), "purity10_lr": float(pur[dec == d].mean())} for d in range(10)]
    rej = pi < np.percentile(pi, 30)   # complement of cape_lite's eligible set E
    res["reject_frac"] = float(rej.mean())

    def desc(m):
        return {"n": int(m.sum()), "deg": float(deg[m].mean()), "boundary": float(np.nanmean(boundary[m])),
                "class_size": float(csize[m].mean()), "tail_frac": float(tail[m].mean()), "purity10_lr": float(pur[m].mean()),
                **{f"{gname}_frac": float((grp[m] == gname).mean()) for gname in GROUPS}}
    res["rejected"], res["kept"], res["all"] = desc(rej), desc(~rej), desc(np.ones(N, bool))
    cls_rej = np.bincount(y, weights=rej.astype(float)) / cnt
    res["class_coverage"] = {"gini_reject_frac": gini(cls_rej), "max_reject_frac": float(cls_rej.max()),
                             "n_classes_reject_gt_50": {gname: int((cls_rej[split[gname]] > 0.5).sum()) for gname in GROUPS},
                             "n_classes": {gname: len(split[gname]) for gname in GROUPS}}
    return res


def main():
    t0 = time.time()
    out = {ds: run(ds) for ds in DATASETS}
    json.dump(out, open("results/T05A_diag.json", "w"), indent=1)
    L = ["# T05 A — π̂ mechanism diagnosis", "", "레이블은 분석용으로만 사용. 기각 = π̂ < 전체 30번째 백분위수 (cape_lite 적격 집합의 여집합).", "",
         "## Spearman ρ(π̂, 속성) — 전체 노드", "", "| dataset | deg | boundary | class_size |", "|---|---|---|---|"]
    for ds, r in out.items():
        L.append(f"| {ds} | " + " | ".join(f"{r['spearman'][k]:+.3f}" for k in ("deg", "boundary", "class_size")) + " |")
    L += ["", "## π̂ 십분위 (1 = 최저)", ""]
    for ds, r in out.items():
        L += [f"**{ds}**", "", "| decile | π̂ | deg | boundary | test frac | tail frac | purity@10 lr |", "|---|---|---|---|---|---|---|"]
        for d in r["deciles"]:
            L.append(f"| {d['decile']} | {d['pi_hat']:.3f} | {d['deg']:.2f} | {d['boundary']:.3f} | {d['test_frac']:.3f} | {d['tail_frac']:.3f} | {d['purity10_lr']:.3f} |")
        L.append("")
    L += ["## 기각 30% vs 유지 70%", "", "| dataset | set | deg | boundary | class_size | tail | purity@10 lr | base | valid | test |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for ds, r in out.items():
        for k in ("rejected", "kept", "all"):
            d = r[k]
            L.append(f"| {ds} | {k} | {d['deg']:.2f} | {d['boundary']:.3f} | {d['class_size']:.0f} | {d['tail_frac']:.3f} | {d['purity10_lr']:.3f} | "
                     f"{d['base_frac']:.3f} | {d['valid_frac']:.3f} | {d['test_frac']:.3f} |")
    L += ["", "## 클래스 커버리지 (클래스별 기각 비율)", "", "| dataset | gini | max | #cls >50% base | valid | test |", "|---|---|---|---|---|---|"]
    for ds, r in out.items():
        cc = r["class_coverage"]; n = cc["n_classes_reject_gt_50"]; tot = cc["n_classes"]
        L.append(f"| {ds} | {cc['gini_reject_frac']:.3f} | {cc['max_reject_frac']:.3f} | " +
                 " | ".join(f"{n[gname]}/{tot[gname]}" for gname in GROUPS) + " |")
    open("results/T05A_diag.md", "w").write("\n".join(L) + "\n")
    log(f"diag done {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
