"""T07 A: estimator comparison (labels for analysis only) -> results/T07A_estimators.{md,json}.
Usage: --ds X (compute + diag one dataset, writes results/T07A_{ds}.json) | --summarize | --source_check."""
import argparse
import inspect
import json
import re

import numpy as np
from sklearn.metrics import roc_auc_score

from cape import generators, novelty, novelty2
from cape.data import load_graph, load_split
from cape.labelmask import make_mask
from cape.novelty2 import ESTIMATORS, compute_estimators
from cape.pool import log

DSS = ["Amazon_clothing", "Amazon_eletronics", "dblp"]
RS = [0.5, 0.2]


def diag(ds):
    y, split = load_graph(ds)["y"], load_split(ds)
    novel = np.isin(y, split["valid"] + split["test"])
    res = {}
    for r in RS:
        est = compute_estimators(ds, r)
        L, _ = make_mask(ds, r)
        U = np.setdiff1d(np.arange(len(y)), L)
        d = {"novel_frac_U": float(novel[U].mean()), "auroc": {}, "calib_abs_err": {}, "mean_p_novel_U": {}, "expected_novel_frac": {}}
        for e in ESTIMATORS:
            s = est[e][U]
            d["auroc"][e] = float(roc_auc_score(novel[U], s))
            if e.startswith("pu_"):
                d["mean_p_novel_U"][e] = float(s.mean())
                d["calib_abs_err"][e] = float(abs(s.mean() - novel[U].mean()))
            w0 = np.clip(s, 0, None)
            d["expected_novel_frac"][e] = {f"tau{t}": float((w0 ** t)[novel[U]].sum() / (w0 ** t).sum()) for t in (1, 2)}
        # T06 hard cut (cent_sgc top 50% of U)
        k = int(np.ceil(len(U) / 2))
        d["t06_hardcut_novel_frac"] = float(novel[U[np.argsort(-est["cent_sgc"][U], kind="stable")[:k]]].mean())
        res[str(r)] = d
    json.dump(res, open(f"results/T07A_{ds}.json", "w"), indent=1)
    log(f"{ds} diag done")


def source_check():
    # identifiers only; method calls ".split(" (sklearn CV folds, str.split) are not split-membership access
    pat = re.compile(r"(?<!\.)\b(split|base|valid|test|load_split)\b")
    funcs = {"novelty2": [novelty2.known_struct, novelty2.pu_novelty, novelty2.compute_estimators],
             "novelty": [novelty.novelty_scores, novelty.load_novelty],
             "generators": [generators.make_nov2, generators.make_mix_alt, generators.make_scarce_gen,
                            generators.make_topk_gen, generators.top_in_pool]}
    hits = {f"{m}.{f.__name__}": [l.strip() for l in inspect.getsource(f).splitlines() if pat.search(l)]
            for m, fs in funcs.items() for f in fs}
    mod_imports = {m: [l.strip() for l in inspect.getsource(mod).splitlines() if l.startswith(("import", "from")) and "load_split" in l]
                   for m, mod in (("novelty2", novelty2), ("novelty", novelty))}
    out = {"function_hits": hits, "module_imports_load_split": mod_imports,
           "passed": not any(hits.values()) and not any(mod_imports.values())}
    json.dump(out, open("results/T07_source_check.json", "w"), indent=1)
    log(json.dumps(out))
    if not out["passed"]:
        raise SystemExit("source check FAILED")


def summarize():
    R = {ds: json.load(open(f"results/T07A_{ds}.json")) for ds in DSS}
    mean05 = {e: float(np.mean([R[ds]["0.5"]["auroc"][e] for ds in DSS])) for e in ESTIMATORS}
    E = max(ESTIMATORS, key=lambda e: mean05[e])
    best_per_ds = {ds: {r: max(ESTIMATORS, key=lambda e: R[ds][r]["auroc"][e]) for r in ("0.5", "0.2")} for ds in DSS}
    out = {"per_ds": R, "mean_auroc_r0.5": mean05, "E_star": E, "best_per_ds": best_per_ds}
    json.dump(out, open("results/T07A_estimators.json", "w"), indent=1)
    L = ["# T07 A — novelty estimator comparison (labels for analysis only)", "",
         f"**E\\* = `{E}`** (r=0.5 세 데이터셋 AUROC 평균 최고; 사전 명시 규칙)", "",
         "## AUROC (U 안: novel vs 숨겨진 base)", "", "| dataset | r | " + " | ".join(ESTIMATORS) + " | best |", "|---|---|" + "---|" * (len(ESTIMATORS) + 1)]
    for ds in DSS:
        for r in ("0.5", "0.2"):
            a = R[ds][r]["auroc"]
            L.append(f"| {ds} | {r} | " + " | ".join(f"{a[e]:.3f}" for e in ESTIMATORS) + f" | {best_per_ds[ds][r]} |")
    L.append("| mean (r=0.5) | | " + " | ".join(f"{mean05[e]:.3f}" for e in ESTIMATORS) + " | |")
    L += ["", "## PU 보정: mean p(novel) over U vs 실제 novel 비율 (절대 오차)", "", "| dataset | r | true | pu_feat | pu_struct | pu_both |", "|---|---|---|---|---|---|"]
    for ds in DSS:
        for r in ("0.5", "0.2"):
            d = R[ds][r]
            L.append(f"| {ds} | {r} | {d['novel_frac_U']:.3f} | " + " | ".join(
                f"{d['mean_p_novel_U'][e]:.3f} ({d['calib_abs_err'][e]:.3f})" for e in ("pu_feat", "pu_struct", "pu_both")) + " |")
    L += ["", f"## 기대 앵커 novel 비율 (w = score^τ over U; E\\* = {E})", "",
          "| dataset | r | U 균등 | T06 하드 컷 | E* τ=1 | E* τ=2 | pu_both τ=1 | cent_sgc τ=1 |", "|---|---|---|---|---|---|---|---|"]
    for ds in DSS:
        for r in ("0.5", "0.2"):
            d = R[ds][r]; x = d["expected_novel_frac"]
            L.append(f"| {ds} | {r} | {d['novel_frac_U']:.3f} | {d['t06_hardcut_novel_frac']:.3f} | {x[E]['tau1']:.3f} | {x[E]['tau2']:.3f} | "
                     f"{x['pu_both']['tau1']:.3f} | {x['cent_sgc']['tau1']:.3f} |")
    open("results/T07A_estimators.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds"); ap.add_argument("--summarize", action="store_true"); ap.add_argument("--source_check", action="store_true")
    a = ap.parse_args()
    if a.source_check:
        source_check()
    elif a.summarize:
        summarize()
    else:
        diag(a.ds)


if __name__ == "__main__":
    main()
