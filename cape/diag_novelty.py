"""T06: §1.4 prechecks (--precheck) and A novelty diagnosis -> results/T06A_novelty.{md,json}. Labels for analysis only."""
import argparse
import inspect
import json
import re
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from cape import generators
from cape.data import load_graph, load_split
from cape.labelmask import make_mask
from cape.novelty import load_novelty
from cape.pool import log

DSS = ["Amazon_eletronics", "dblp"]
RS = [0.5, 0.2]


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if x.sum() > 0 else 0.0


def precheck():
    out = {}
    for ds in DSS:
        y, split = load_graph(ds)["y"], load_split(ds)
        L, _ = make_mask(ds, 0.5)
        bad = []
        for c in split["base"]:
            n_c = int((y == c).sum()); l_c = int((y[L] == c).sum())
            if abs(l_c / n_c - 0.5) > 1 / n_c + 1e-12:
                bad.append((c, l_c, n_c))
        L1, _ = make_mask(ds, 1.0)
        U1 = np.setdiff1d(np.arange(len(y)), L1)
        novel = np.nonzero(np.isin(y, split["valid"] + split["test"]))[0]
        out[ds] = {"check1_bad_classes": bad, "check2_U_equals_novel_at_r1": bool(np.array_equal(U1, novel))}
    pat = re.compile(r"\b(split|base|valid|test)\b")
    src = {f: inspect.getsource(getattr(generators, f)) for f in ("make_unl_uni", "make_nov", "make_scarce_gen", "make_topk_gen", "top_in_pool")}
    hits = {f: [l.strip() for l in s.splitlines() if pat.search(l)] for f, s in src.items()}
    out["check3_split_refs_in_generator_src"] = hits
    out["passed"] = all(not out[ds]["check1_bad_classes"] and out[ds]["check2_U_equals_novel_at_r1"] for ds in DSS) \
        and not any(hits.values())
    json.dump(out, open("results/T06_precheck.json", "w"), indent=1)
    log(json.dumps(out))
    if not out["passed"]:
        raise SystemExit("T06 precheck FAILED")


def diag(ds, r):
    y, split = load_graph(ds)["y"], load_split(ds)
    N = len(y)
    L, yL = make_mask(ds, r)
    nov = load_novelty(ds, r, L, yL)
    U = np.setdiff1d(np.arange(N), L)
    is_novel = np.isin(y, split["valid"] + split["test"])
    grp = {g: np.isin(y, split[g]) for g in ("base", "valid", "test")}
    comp = lambda idx: {"hidden_base" if g == "base" else g: float(grp[g][idx].mean()) for g in grp}
    k = int(np.ceil(len(U) / 2))
    A_nov = U[np.argsort(-nov["sgc"][U], kind="stable")[:k]]
    novel_cls = np.array(split["valid"] + split["test"])
    cnt = np.bincount(y, minlength=y.max() + 1)
    incl = np.bincount(y[A_nov], minlength=len(cnt))[novel_cls] / cnt[novel_cls]
    return {"n_L": int(len(L)), "n_U": int(len(U)), "U_comp": comp(U),
            "auroc_sgc": float(roc_auc_score(is_novel[U], nov["sgc"][U])),
            "auroc_raw": float(roc_auc_score(is_novel[U], nov["raw"][U])),
            "A_nov_n": int(k), "A_nov_comp": comp(A_nov),
            "expected_anchor_novel_frac": {"naq_all": float(is_novel.mean()), "unl_uni": float(is_novel[U].mean()),
                                           "nov": float(is_novel[A_nov].mean()), "naq_unl_r1": 1.0},
            "A_nov_novel_class_coverage": float((incl > 0).mean()), "A_nov_class_incl_gini": gini(incl),
            "A_nov_class_incl_min_max": [float(incl.min()), float(incl.max())]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precheck", action="store_true")
    args = ap.parse_args()
    if args.precheck:
        precheck(); return
    t0 = time.time()
    out = {ds: {str(r): diag(ds, r) for r in RS} for ds in DSS}
    json.dump(out, open("results/T06A_novelty.json", "w"), indent=1)
    L = ["# T06 A — novelty diagnosis (labels for analysis only)", "",
         "| dataset | r | |L| | |U| | U: hidden base / valid / test | AUROC sgc | AUROC raw | A_nov: hidden base / valid / test | E[novel frac] NaQ / U-uni / nov / naq_unl | novel cls covered | gini incl |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds in DSS:
        for r in RS:
            d = out[ds][str(r)]; u, a, e = d["U_comp"], d["A_nov_comp"], d["expected_anchor_novel_frac"]
            L.append(f"| {ds} | {r} | {d['n_L']} | {d['n_U']} | {u['hidden_base']:.3f} / {u['valid']:.3f} / {u['test']:.3f} | "
                     f"{d['auroc_sgc']:.3f} | {d['auroc_raw']:.3f} | {a['hidden_base']:.3f} / {a['valid']:.3f} / {a['test']:.3f} | "
                     f"{e['naq_all']:.3f} / {e['unl_uni']:.3f} / {e['nov']:.3f} / 1.0 | {d['A_nov_novel_class_coverage']:.3f} | {d['A_nov_class_incl_gini']:.3f} |")
    open("results/T06A_novelty.md", "w").write("\n".join(L) + "\n")
    log(f"diag done {time.time() - t0:.1f}s")
    print("\n".join(L))


if __name__ == "__main__":
    main()
