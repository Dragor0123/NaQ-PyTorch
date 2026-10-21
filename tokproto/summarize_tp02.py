"""TP02 summary -> results/TP02_summary.{md,json}."""
import json
import os
import subprocess

import numpy as np

from cape.data import DATASETS

SHORT = {"Amazon_clothing": "clothing", "Amazon_eletronics": "eletronics", "dblp": "dblp"}
DAG = lambda a: a + " †" if "sgcora" in a else a
KS = ("test1_lr", "test5_lr")


def load(ds, arm, s):
    p = f"results/tp/{ds}_{arm}_s{s}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def f(x, d=2, sign=False):
    return "—" if x is None else (f"{x:+.{d}f}" if sign else f"{x:.{d}f}")


def acc(ds, arm, s, k):
    r = load(ds, arm, s)
    return r["test"][k]["mean"] if r else None


def paired(ds, a, b, k):
    d = [None if acc(ds, a, s, k) is None or acc(ds, b, s, k) is None else acc(ds, a, s, k) - acc(ds, b, s, k) for s in (0, 1, 2)]
    v = [x for x in d if x is not None]
    return {"per_seed": d, "mean": float(np.mean(v)) if v else None, "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
            "n_pos": int(sum(x > 0 for x in v)), "n": len(v)}


def main():
    S, L = {}, ["# TP02 summary", ""]
    reg = json.load(open("results/tp/TP02_regression.json"))
    grep = subprocess.run(["grep", "-rn", "oracle_filter", "tokproto/"], capture_output=True, text=True).stdout
    grep = "\n".join(l for l in grep.splitlines() if not l.startswith("Binary"))
    S["regression"], S["grep_oracle_filter"] = reg, grep
    L += ["## 1. Regression check and oracle_filter call sites", ""]
    for ds in DATASETS:
        r = reg[ds]
        L.append(f"- {SHORT[ds]}: tokens_equal={r['tokens_equal']}, |L_new − L_TP01|={r['abs_diff']:.2e}, |mean − softmax| (identical tokens)={r['mean_vs_softmax_identical_tokens']:.2e}")
    L += [f"- pass={reg['pass']}", "", "`grep -rn oracle_filter tokproto/`:", "", "```", grep, "```", ""]
    L += ["## 2. Oracle-token diagnostics", "", open("results/TP02A_oracle_tokens.md").read().split("\n", 2)[2]]
    # 3. three seeds
    ARM3 = ["tp_l0", "tp_a2_sgc_l25_m8", "tp_a2_sgc_l25_m8_mean"]
    L += ["## 3. Three seeds (LR %)", "", "| dataset | arm | s0 1s | s1 1s | s2 1s | mean ± sd 1s | s0 5s | s1 5s | s2 5s | mean ± sd 5s |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    S["seeds"] = {}
    for ds in DATASETS:
        for arm in ARM3:
            cells = []
            for k in KS:
                v = [acc(ds, arm, s, k) for s in (0, 1, 2)]
                vv = [x for x in v if x is not None]
                S["seeds"].setdefault(ds, {}).setdefault(arm, {})[k] = v
                cells += [f(x) for x in v] + [f"{np.mean(vv):.2f} ± {f(np.std(vv, ddof=1) if len(vv) > 1 else None)}" if vv else "—"]
            L.append(f"| {SHORT[ds]} | {arm} | " + " | ".join(cells) + " |")
    # 4. paired
    PAIRS = [("tp_a2_sgc_l25_m8", "tp_l0"), ("tp_a2_sgc_l25_m8_mean", "tp_l0"), ("tp_a2_sgc_l25_m8", "tp_a2_sgc_l25_m8_mean")]
    L += ["", "## 4. Paired differences (same seed, %p)", "", "| comparison | dataset | metric | s0 | s1 | s2 | mean | sd | #pos / n |",
          "|---|---|---|---|---|---|---|---|---|"]
    S["paired"] = {}
    for a, b in PAIRS:
        for ds in DATASETS:
            for k in KS:
                p = paired(ds, a, b, k)
                S["paired"].setdefault(f"{a} - {b}", {}).setdefault(ds, {})[k] = p
                L.append(f"| {a} − {b} | {SHORT[ds]} | {k} | " + " | ".join(f(x, sign=True) for x in p["per_seed"])
                         + f" | {f(p['mean'], sign=True)} | {f(p['sd'])} | {p['n_pos']} / {p['n']} |")
    # 5. seed 0
    ARM0 = ["tp_l0", "tp_a2_sgc_l25_m8", "tp_a2_sgc_l50_m8", "tp_a2_sgcora_l25_m8", "tp_a2_sgcora_l50_m8", "tp_a2_sgc_l25_m8_t02"]
    L += ["", "## 5. Seed 0", "", "| dataset | arm | 1s LR ± CI | 5s LR ± CI | best ep | token purity | mean #tok | attn H/logT | mass same-label / uniform |",
          "|---|---|---|---|---|---|---|---|---|"]
    S["seed0"] = {}
    for ds in DATASETS:
        for arm in ARM0:
            r = load(ds, arm, 0)
            if not r:
                L.append(f"| {SHORT[ds]} | {DAG(arm)} | — | — | — | — | — | — | — |"); continue
            ts, ad = r.get("token_stats") or {}, r.get("attn_diag") or {}
            S["seed0"].setdefault(ds, {})[arm] = {"test": r["test"], "best_episode": r["best_episode"], "best_valid": r["best_valid"],
                                                  "token_stats": ts, "attn_diag": ad}
            L.append(f"| {SHORT[ds]} | {DAG(arm)} | {r['test']['test1_lr']['mean']:.2f} ± {r['test']['test1_lr']['ci95']:.2f} | "
                     f"{r['test']['test5_lr']['mean']:.2f} ± {r['test']['test5_lr']['ci95']:.2f} | {r['best_episode']} | {f(ts.get('token_purity'), 3)} | "
                     f"{f(ts.get('mean_tokens'))} | {f(ad.get('norm_entropy'), 3)} | {f(ad.get('mass_anchor_label'), 3)} / {f(ad.get('uniform_anchor_label'), 3)} |")
    DIFF0 = [("tp_a2_sgcora_l25_m8", "tp_a2_sgc_l25_m8"), ("tp_a2_sgcora_l50_m8", "tp_a2_sgc_l50_m8"),
             ("tp_a2_sgcora_l25_m8", "tp_l0"), ("tp_a2_sgcora_l50_m8", "tp_l0"), ("tp_a2_sgc_l25_m8_t02", "tp_a2_sgc_l25_m8")]
    L += ["", "Seed 0 differences (%p):", "", "| comparison | " + " | ".join(f"{SHORT[d]} 1s | {SHORT[d]} 5s" for d in DATASETS) + " |", "|---|" + "---|" * 6]
    S["seed0_diff"] = {}
    for a, b in DIFF0:
        row = []
        for ds in DATASETS:
            for k in KS:
                x, y = acc(ds, a, 0, k), acc(ds, b, 0, k)
                d = None if x is None or y is None else x - y
                S["seed0_diff"].setdefault(f"{a} - {b}", {}).setdefault(ds, {})[k] = d
                row.append(f(d, sign=True))
        L.append(f"| {DAG(a)} − {DAG(b)} | " + " | ".join(row) + " |")
    # 6. prereg (arithmetic only)
    pd = S["paired"]; sd = S["seed0_diff"]
    ent = lambda ds, a: ((S["seed0"].get(ds, {}).get(a) or {}).get("attn_diag") or {}).get("norm_entropy")
    P = [("TPP011", "sgc_l25_m8 − tp_l0, 3-seed mean 1s (#pos): " + ", ".join(
              f"{SHORT[ds]} {f(pd['tp_a2_sgc_l25_m8 - tp_l0'][ds]['test1_lr']['mean'], sign=True)} ({pd['tp_a2_sgc_l25_m8 - tp_l0'][ds]['test1_lr']['n_pos']}/{pd['tp_a2_sgc_l25_m8 - tp_l0'][ds]['test1_lr']['n']})" for ds in DATASETS)
              + "; criterion: datasets with mean ≥ +0.3 and #pos ≥ 2 is ≤ 1"),
         ("TPP012", "|sgc_l25_m8 − sgc_l25_m8_mean|, 3-seed mean 1s: " + ", ".join(
              f"{SHORT[ds]} {f(None if pd['tp_a2_sgc_l25_m8 - tp_a2_sgc_l25_m8_mean'][ds]['test1_lr']['mean'] is None else abs(pd['tp_a2_sgc_l25_m8 - tp_a2_sgc_l25_m8_mean'][ds]['test1_lr']['mean']))}" for ds in DATASETS) + "; threshold ≤ 0.3 all"),
         ("TPP013", "t02 norm entropy: " + ", ".join(f"{SHORT[ds]} {f(ent(ds, 'tp_a2_sgc_l25_m8_t02'), 3)}" for ds in DATASETS) + "; threshold ≤ 0.8 all"),
         ("TPP014", "t02 − sgc_l25_m8 (s0, 1s): " + ", ".join(f"{SHORT[ds]} {f(sd['tp_a2_sgc_l25_m8_t02 - tp_a2_sgc_l25_m8'][ds]['test1_lr'], sign=True)}" for ds in DATASETS) + "; criterion: ≤ +0.3 in ≥ 2 datasets"),
         ("TPP015", "sgcora_l50_m8† − tp_l0 (s0, 1s): " + ", ".join(f"{SHORT[ds]} {f(sd['tp_a2_sgcora_l50_m8 - tp_l0'][ds]['test1_lr'], sign=True)}" for ds in DATASETS) + "; criterion: datasets ≥ +1.0 is ≤ 1"),
         ("TPP016", "sgcora_l50_m8† − sgc_l50_m8 (s0, 1s): " + ", ".join(f"{SHORT[ds]} {f(sd['tp_a2_sgcora_l50_m8 - tp_a2_sgc_l50_m8'][ds]['test1_lr'], sign=True)}" for ds in DATASETS) + "; criterion: largest in eletronics")]
    L += ["", "## 6. Preregistration — arithmetic comparison (no verdicts)", "", "| ID | actual values |", "|---|---|"] + [f"| {i} | {t} |" for i, t in P]
    S["prereg"] = dict(P)
    open("results/TP02_summary.md", "w").write("\n".join(L) + "\n")
    json.dump(S, open("results/TP02_summary.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
