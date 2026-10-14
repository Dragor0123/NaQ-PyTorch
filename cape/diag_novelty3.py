"""T08 A': calibrated PU variants (labels for analysis only) -> results/T08A_calibration.{md,json}; --source_check."""
import argparse
import inspect
import json
import os
import re
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from cape import generators, novelty, novelty2
from cape.data import load_graph, load_split
from cape.labelmask import make_mask
from cape.novelty2 import ESTIMATORS3, compute_estimators3
from cape.pool import log

R = 0.5


def diag(ds, skip_both):
    t0 = time.time()
    est = compute_estimators3(ds, R, skip_both)
    y, split = load_graph(ds)["y"], load_split(ds)
    novel = np.isin(y, split["valid"] + split["test"])
    L, _ = make_mask(ds, R)
    U = np.setdiff1d(np.arange(len(y)), L)
    t07 = json.load(open(f"results/T07A_{ds}.json"))["0.5"]
    d = {"novel_frac_U": float(novel[U].mean()), "variants": {}, "t07": {e: {"auroc": t07["auroc"][e], "calib_abs_err": t07["calib_abs_err"][e]}
                                                                            for e in ("pu_feat", "pu_struct", "pu_both")}}
    for e in ESTIMATORS3:
        if e not in est:
            continue
        s = est[e][U]
        w = np.clip(s, 0, None)
        d["variants"][e] = {"auroc": float(roc_auc_score(novel[U], s)), "mean_p_novel_U": float(s.mean()),
                            "calib_err": float(s.mean() - novel[U].mean()), "calib_abs_err": float(abs(s.mean() - novel[U].mean())),
                            "fit_s": float(est.get(f"time_{e}_s", np.nan)), "expected_novel_frac_tau1": float(w[novel[U]].sum() / w.sum())}
    d["pca_s"] = float(est.get("time_pca_s", np.nan)); d["wall_s"] = time.time() - t0
    json.dump(d, open(f"results/T08A_{ds}.json", "w"), indent=1)
    log(f"{ds} A' done {d['wall_s']:.1f}s")


def source_check():
    pat = re.compile(r"(?<!\.)\b(split|base|valid|test|load_split)\b")   # same checker as T07
    funcs = [novelty2.pu_novelty2, novelty2.compute_estimators3, novelty2.known_struct, novelty2.pu_novelty, novelty.novelty_scores,
             generators.make_mix, generators.make_mix_alt, generators.make_nov2, generators.make_scarce_gen,
             generators.make_topk_gen, generators.top_in_pool]
    hits = {f"{f.__module__}.{f.__name__}": [l.strip() for l in inspect.getsource(f).splitlines() if pat.search(l)] for f in funcs}
    imp = {m.__name__: [l for l in inspect.getsource(m).splitlines() if l.startswith(("import", "from")) and "load_split" in l] for m in (novelty, novelty2)}
    out = {"function_hits": hits, "module_imports_load_split": imp, "passed": not any(hits.values()) and not any(imp.values())}
    json.dump(out, open("results/T08_source_check.json", "w"), indent=1)
    log(json.dumps(out))
    if not out["passed"]:
        raise SystemExit("source check FAILED")


def summarize():
    dss = [ds for ds in ("Amazon_eletronics", "dblp", "Amazon_clothing") if os.path.exists(f"results/T08A_{ds}.json")]
    Rr = {ds: json.load(open(f"results/T08A_{ds}.json")) for ds in dss}
    main2 = ["Amazon_eletronics", "dblp"]
    vs = [e for e in ESTIMATORS3 if all(e in Rr[ds]["variants"] for ds in main2)]
    ok = [e for e in vs if all(Rr[ds]["variants"][e]["calib_abs_err"] <= 0.15 for ds in main2)]
    mean_auc = {e: float(np.mean([Rr[ds]["variants"][e]["auroc"] for ds in main2])) for e in vs}
    mean_cal = {e: float(np.mean([Rr[ds]["variants"][e]["calib_abs_err"] for ds in main2])) for e in vs}
    E2 = max(ok, key=lambda e: mean_auc[e]) if ok else min(vs, key=lambda e: mean_cal[e])
    rule = "calib ≤ 0.15 on both → max mean AUROC" if ok else "no variant with calib ≤ 0.15 on both → min mean calib error"
    out = {"per_ds": Rr, "eligible": ok, "mean_auroc": mean_auc, "mean_calib_abs_err": mean_cal, "E2_star": E2, "rule_applied": rule}
    json.dump(out, open("results/T08A_calibration.json", "w"), indent=1)
    L = ["# T08 A′ — calibrated PU variants (r = 0.5, labels for analysis only)", "", f"**E2\\* = `{E2}`** ({rule}; 선택은 eletronics·dblp 기준)", "",
         "| dataset | variant | AUROC | mean p_novel (U) | 실제 novel 비율 | 보정 오차 (부호) | 적합 시간 s | 기대 앵커 novel 비율 τ=1 |", "|---|---|---|---|---|---|---|---|"]
    for ds in dss:
        d = Rr[ds]
        for e, v in d["variants"].items():
            L.append(f"| {ds} | {e} | {v['auroc']:.3f} | {v['mean_p_novel_U']:.3f} | {d['novel_frac_U']:.3f} | {v['calib_err']:+.3f} | {v['fit_s']:.1f} | {v['expected_novel_frac_tau1']:.3f} |")
        for e, v in d["t07"].items():
            L.append(f"| {ds} | (T07) {e} | {v['auroc']:.3f} | | {d['novel_frac_U']:.3f} | abs {v['calib_abs_err']:.3f} | | |")
        L.append(f"| {ds} | (PCA 적합 시간) | | | | | {d['pca_s']:.1f} | |")
    L += ["", "평균 (eletronics, dblp): " + ", ".join(f"{e} AUROC {mean_auc[e]:.3f} / 보정 오차 {mean_cal[e]:.3f}" for e in vs)]
    open("results/T08A_calibration.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds"); ap.add_argument("--skip_both", action="store_true")
    ap.add_argument("--summarize", action="store_true"); ap.add_argument("--source_check", action="store_true")
    a = ap.parse_args()
    if a.source_check:
        source_check()
    elif a.summarize:
        summarize()
    else:
        diag(a.ds, a.skip_both)


if __name__ == "__main__":
    main()
