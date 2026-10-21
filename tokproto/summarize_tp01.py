"""TP01 summary -> results/TP01_summary.{md,json}."""
import json
import os

import numpy as np

from cape.data import DATASETS

ARMS = ["tp_l0", "tp_a2_sgc_l25_m4", "tp_a2_sgc_l25_m8", "tp_a2_sgc_l50_m4", "tp_a2_sgc_l50_m8",
        "tp_b_sgc_l50_m8", "tp_a2_pprrow_l50_m8"]
A2SGC = ARMS[1:5]
SHORT = {"Amazon_clothing": "clothing", "Amazon_eletronics": "eletronics", "dblp": "dblp"}


def load(ds, arm, s):
    p = f"results/tp/{ds}_{arm}_s{s}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, d=2):
    return "—" if x is None else f"{x:.{d}f}"


def main():
    S, L = {}, ["# TP01 summary", ""]
    # 1. gates
    gb = json.load(open("results/tp/TP01_Gb.json"))
    S["Gb"] = gb
    L += ["## 1. Gates", "", "- Ga: all inputs present (data/*, cache/{ds}/pool.npz, tasks.npz, split.json, save/downstream_tasks 5way_1/5shot.pkl).",
          f"- Gb: 1 max|Δ|={gb['gb1_max_abs_diff']:.2e}, 2 max|Δ|={gb['gb2_max_abs_diff']:.2e}, 3 max|Δ|={gb['gb3_max_abs_diff']:.2e} finite={gb['gb3_finite']}, "
          f"4 |Δ|={gb['gb4_abs_diff']:.2e} → all_pass={gb['all_pass']}"]
    for ds in DATASETS:
        p = f"results/tp/{ds}_pprrow_build.json"
        if os.path.exists(p):
            b = json.load(open(p))
            L.append(f"- pprrow cache {ds}: verify max_abs_err={b['verify']['max_abs_err']:.2e}, random>50th={b['verify']['random_exceed_50th']}, "
                     f"top-50 score≤0 frac={b['zero_score_frac']:.4f}, build {b['time']['total_s']:.0f}s")
    G0 = {}
    for ds in ("Amazon_eletronics", "dblp"):
        r, ref = load(ds, "tp_l0", 0), json.load(open(f"results/expB/{ds}_naq_feat_s0.json"))
        if r:
            G0[ds] = {k: r["test"][k]["mean"] - ref["test"][k]["mean"] for k in ("test1_lr", "test5_lr")}
    S["G0"] = {"delta": G0, "pass": bool(len(G0) == 2 and all(abs(v) <= 0.7 for d in G0.values() for v in d.values()))}
    L.append("- G0 (tp_l0 s0 − expB naq_feat s0): " + ", ".join(f"{SHORT[ds]} Δ1s={d['test1_lr']:+.2f} Δ5s={d['test5_lr']:+.2f}" for ds, d in G0.items())
             + f" → pass={S['G0']['pass']}")
    # 2. A
    A = json.load(open("results/TP01A_diag.json"))
    L += ["", "## 2. A — token diagnostics", "", open("results/TP01A_diag.md").read().split("\n", 2)[2]]
    # 3. B1
    L += ["## 3. B1 — seed 0", "", "| dataset | arm | best_valid | 1s LR ± CI | 5s LR ± CI | 1s proto | best ep | shortfall | attn H/logT | mass same-label / uniform |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    B1 = {}
    for ds in DATASETS:
        for arm in ARMS:
            r = load(ds, arm, 0)
            if not r:
                continue
            B1.setdefault(ds, {})[arm] = r
            ts, ad = r.get("token_stats", {}), r.get("attn_diag", {})
            L.append(f"| {SHORT[ds]} | {arm} | {r['best_valid'] * 100:.2f} | {r['test']['test1_lr']['mean']:.2f} ± {r['test']['test1_lr']['ci95']:.2f} | "
                     f"{r['test']['test5_lr']['mean']:.2f} ± {r['test']['test5_lr']['ci95']:.2f} | {r['test']['test1_proto']['mean']:.2f} | {r['best_episode']} | "
                     f"{f(ts.get('shortfall_rate'), 3)} | {f(ad.get('norm_entropy'), 3)} | {f(ad.get('mass_anchor_label'), 3)} / {f(ad.get('uniform_anchor_label'), 3)} |")
    S["B1"] = {ds: {a: {"best_valid": r["best_valid"], "test": r["test"], "best_episode": r["best_episode"],
                        "token_stats": r.get("token_stats"), "attn_diag": r.get("attn_diag")} for a, r in v.items()} for ds, v in B1.items()}
    # 4. diffs
    L += ["", "## 4. B1 − tp_l0 (seed 0, %p)", "", "| arm | " + " | ".join(f"{SHORT[d]} 1s | {SHORT[d]} 5s" for d in DATASETS) + " |",
          "|---|" + "---|" * 6]
    D = {}
    for arm in ARMS[1:]:
        row = []
        for ds in DATASETS:
            b, r = B1.get(ds, {}).get("tp_l0"), B1.get(ds, {}).get(arm)
            if b and r:
                d = {k: r["test"][k]["mean"] - b["test"][k]["mean"] for k in ("test1_lr", "test5_lr")}
                D.setdefault(arm, {})[ds] = d
                row += [f"{d['test1_lr']:+.2f}", f"{d['test5_lr']:+.2f}"]
            else:
                row += ["—", "—"]
        L.append(f"| {arm} | " + " | ".join(row) + " |")
    S["B1_diff"] = D
    # 5. selection
    L += ["", "## 5. Selection (mean best_valid over 3 datasets, arms 2–7)", "", "| arm | mean best_valid |", "|---|---|"]
    sel = {}
    for arm in ARMS[1:]:
        v = [B1[ds][arm]["best_valid"] for ds in DATASETS if arm in B1.get(ds, {})]
        if len(v) == 3:
            sel[arm] = float(np.mean(v))
            L.append(f"| {arm} | {sel[arm] * 100:.3f} |")
    cstar = max(sel, key=lambda a: (sel[a], -ARMS.index(a))) if sel else None   # ties -> earlier arm
    S["selection"] = {"mean_best_valid": sel, "C_star": cstar}
    L.append(f"\nC* = `{cstar}`")
    # 6. B2
    L += ["", "## 6. B2 — seeds", ""]
    B2 = {}
    if cstar:
        L += ["| dataset | arm | s0 1s | s1 1s | s2 1s | mean ± sd 1s | s0 5s | s1 5s | s2 5s | mean ± sd 5s |", "|---|---|---|---|---|---|---|---|---|---|"]
        for ds in DATASETS:
            for arm in ("tp_l0", cstar):
                rs = {s: load(ds, arm, s) for s in (0, 1, 2)}
                cells = []
                for k in ("test1_lr", "test5_lr"):
                    v = [rs[s]["test"][k]["mean"] if rs[s] else None for s in (0, 1, 2)]
                    vv = [x for x in v if x is not None]
                    sd = np.std(vv, ddof=1) if len(vv) > 1 else None
                    cells += [f(x) for x in v] + [f"{np.mean(vv):.2f} ± {f(sd)}"]
                    B2.setdefault(ds, {}).setdefault(arm, {})[k] = v
                L.append(f"| {SHORT[ds]} | {arm} | " + " | ".join(cells) + " |")
        L += ["", "Paired differences C* − tp_l0 (same seed):", "", "| dataset | metric | s0 | s1 | s2 | mean | sd | #pos / #neg |", "|---|---|---|---|---|---|---|---|"]
        for ds in DATASETS:
            for k in ("test1_lr", "test5_lr"):
                a, b = B2[ds][cstar][k], B2[ds]["tp_l0"][k]
                d = [x - y if x is not None and y is not None else None for x, y in zip(a, b)]
                dd = [x for x in d if x is not None]
                B2[ds].setdefault("paired", {})[k] = {"per_seed": d, "mean": float(np.mean(dd)) if dd else None,
                                                     "sd": float(np.std(dd, ddof=1)) if len(dd) > 1 else None,
                                                     "n_pos": int(sum(x > 0 for x in dd)), "n_neg": int(sum(x < 0 for x in dd))}
                p = B2[ds]["paired"][k]
                L.append(f"| {SHORT[ds]} | {k} | " + " | ".join(f(x) if x is None else f"{x:+.2f}" for x in d)
                         + f" | {f(p['mean'])} | {f(p['sd'])} | {p['n_pos']} / {p['n_neg']} |")
    S["B2"] = B2
    # 7. reference
    L += ["", "## 7. tp_l0 vs existing naq_feat (expB)", "", "| dataset | seed | naq_feat 1s / 5s | tp_l0 1s / 5s |", "|---|---|---|---|"]
    for ds, seeds in (("Amazon_eletronics", (0, 1, 2)), ("dblp", (0,))):
        for s in seeds:
            ref, r = json.load(open(f"results/expB/{ds}_naq_feat_s{s}.json")), load(ds, "tp_l0", s)
            L.append(f"| {SHORT[ds]} | {s} | {ref['test']['test1_lr']['mean']:.2f} / {ref['test']['test5_lr']['mean']:.2f} | "
                     + (f"{r['test']['test1_lr']['mean']:.2f} / {r['test']['test5_lr']['mean']:.2f}" if r else "—") + " |")
    # 8. preregistration (arithmetic only)
    P = []
    P.append(("TPP001", f"Gb1 {gb['gb1_max_abs_diff']:.2e}, Gb2 {gb['gb2_max_abs_diff']:.2e}, Gb3 {gb['gb3_max_abs_diff']:.2e} (finite={gb['gb3_finite']}), Gb4 {gb['gb4_abs_diff']:.2e}; threshold 1e-5"))
    P.append(("TPP002", "; ".join(f"{SHORT[ds]} |Δ1s|={abs(d['test1_lr']):.2f}, |Δ5s|={abs(d['test5_lr']):.2f}" for ds, d in G0.items()) + "; threshold 0.7"))
    P.append(("TPP003", "; ".join(f"{SHORT[ds]} sgc m8 token purity {A[ds]['sgc_m8']['token_purity']:.3f} vs query purity {A[ds]['query_purity']:.3f}"
                                  for ds in ("Amazon_eletronics", "dblp"))))
    P.append(("TPP004", "; ".join(f"{SHORT[ds]} overlap@8 {A[ds]['pprrow_ppr_overlap8']:.3f}, pprrow hop≤2 {A[ds]['pprrow_m8']['hop']['le2']:.3f}" for ds in DATASETS)
              + "; thresholds 0.5 / 0.7"))
    t5 = []
    for ds in DATASETS:
        if all(a in B1.get(ds, {}) for a in A2SGC + ["tp_l0"]):
            ba = max(A2SGC, key=lambda a: (B1[ds][a]["best_valid"], -ARMS.index(a)))
            t5.append(f"{SHORT[ds]} {ba} Δ1s={B1[ds][ba]['test']['test1_lr']['mean'] - B1[ds]['tp_l0']['test']['test1_lr']['mean']:+.2f}")
    P.append(("TPP005", "; ".join(t5) + "; threshold +0.5, ≥2 datasets"))
    g = lambda a, ds: D.get(a, {}).get(ds, {}).get("test1_lr")
    P.append(("TPP006", "tp_a2_sgc_l50_m8 Δ1s: " + ", ".join(f"{SHORT[ds]} {f(g('tp_a2_sgc_l50_m8', ds))}" for ds in DATASETS)))
    v1 = lambda a, ds: B1[ds][a]["test"]["test1_lr"]["mean"] if a in B1.get(ds, {}) else None
    P.append(("TPP007", "1s LR b vs a2 (l50 m8): " + ", ".join(f"{SHORT[ds]} {f(v1('tp_b_sgc_l50_m8', ds))} vs {f(v1('tp_a2_sgc_l50_m8', ds))}" for ds in DATASETS)))
    P.append(("TPP008", "1s LR sgc vs pprrow (a2 l50 m8): " + ", ".join(f"{SHORT[ds]} {f(v1('tp_a2_sgc_l50_m8', ds))} vs {f(v1('tp_a2_pprrow_l50_m8', ds))}" for ds in DATASETS)))
    P.append(("TPP009", "tp_a2_sgc_l50_m8 norm entropy: " + ", ".join(
        f"{SHORT[ds]} {f((B1[ds]['tp_a2_sgc_l50_m8'].get('attn_diag') or {}).get('norm_entropy'), 3)}" for ds in DATASETS if 'tp_a2_sgc_l50_m8' in B1.get(ds, {})) + "; threshold 0.9"))
    if cstar and B2:
        P.append(("TPP010", f"C*={cstar}: " + "; ".join(f"{SHORT[ds]} mean Δ1s={f(B2[ds]['paired']['test1_lr']['mean'])}, #pos={B2[ds]['paired']['test1_lr']['n_pos']}/"
                                                        f"{sum(x is not None for x in B2[ds]['paired']['test1_lr']['per_seed'])}" for ds in DATASETS) + "; threshold +0.5 and ≥2 positive, ≥2 datasets"))
    L += ["", "## 8. Preregistration — arithmetic comparison (no verdicts)", "", "| ID | actual values |", "|---|---|"] + [f"| {i} | {t} |" for i, t in P]
    S["prereg"] = dict(P)
    open("results/TP01_summary.md", "w").write("\n".join(L) + "\n")
    json.dump(S, open("results/TP01_summary.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
