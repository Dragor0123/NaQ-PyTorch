"""T09 aggregation -> results/T09_summary.{md,json}. Usage: --source_check | (default) summarize.
Labels/split are read only through stored episode stats (analysis only)."""
import argparse
import inspect
import json
import os
import re

import numpy as np
from scipy.stats import spearmanr

from cape import generators, novelty, novelty2
from cape.pool import log

E, D, C = "Amazon_eletronics", "dblp", "Amazon_clothing"
DSS = [E, D, C]
BEST = {E: "naq_feat", D: "naq_diff", C: "naq_feat"}
SEEDS = [0, 1, 2, 3, 4]
NOV, MIX, NMX, SUP, UNL, NOVC = "nov2_t1_r0.5", "mix_alt_r0.5", "naq_mix_r0.5", "sup_only_r0.5", "naq_unl", "nov2c_t1_r0.5"
LAB = {NOV: "nov", MIX: "mix_alt ‡", NMX: "naq_mix ‡", SUP: "sup_only ‡", UNL: "naq_unl †", NOVC: "nov2c_t1"}
SH = (("test1_lr", "1s"), ("test5_lr", "5s"))


def planned():
    """The 47 T09 runs in queue order (same as cape/run_T09.sh)."""
    q = []
    q += [("P1", C, a, s) for s in (0, 1, 2) for a in ("naq_feat", NOV, MIX, NMX)]
    q += [("P2", ds, a, s) for s in (2, 3, 4) for ds in (E, D) for a in (MIX, NMX)]
    q += [("P3", ds, a, s) for s in (3, 4) for ds, a in ((E, "naq_feat"), (E, NOV), (D, "naq_diff"), (D, NOV))]
    q += [("P4", C, a, s) for s in (0, 1, 2) for a in (UNL, SUP, "naq_diff")]
    q += [("P5", ds, SUP, s) for s in (2, 3, 4) for ds in (E, D)]
    return q


def source_check():
    pat = re.compile(r"(?<!\.)\b(split|base|valid|test|load_split)\b")   # same checker as T07/T08
    funcs = [novelty2.compute_estimators, novelty2.known_struct, novelty2.pu_novelty, novelty.novelty_scores,
             generators.make_mix, generators.make_mix_alt, generators.make_nov2, generators.make_scarce_gen,
             generators.make_topk_gen, generators.top_in_pool]
    hits = {f"{f.__module__}.{f.__name__}": [l.strip() for l in inspect.getsource(f).splitlines() if pat.search(l)] for f in funcs}
    imp = {m.__name__: [l for l in inspect.getsource(m).splitlines() if l.startswith(("import", "from")) and "load_split" in l] for m in (novelty, novelty2)}
    out = {"function_hits": hits, "module_imports_load_split": imp, "passed": not any(hits.values()) and not any(imp.values())}
    json.dump(out, open("results/T09_source_check.json", "w"), indent=1)
    log(json.dumps(out))
    if not out["passed"]:
        raise SystemExit("source check FAILED")


def load(ds, arm, s):
    p = f"results/expB/{ds}_{arm}_s{s}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def acc(r, k="test1_lr"):
    return r["test"][k]["mean"]


def nfrac(r):
    st = r["episode_stats"]
    return st.get("anchor_novel_frac", st["anchor_group_frac"]["valid"] + st["anchor_group_frac"]["test"])


def sd(v):
    return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")


def seeds_of(ds, *arms):
    return [s for s in SEEDS if all(load(ds, a, s) for a in arms)]


def paired(ds, a, b, k="test1_lr", seeds=None):
    seeds = seeds_of(ds, a, b) if seeds is None else seeds
    return {s: acc(load(ds, a, s), k) - acc(load(ds, b, s), k) for s in seeds}


f2 = lambda x: "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.2f}"


def main_summary():
    J = {"main": {}, "paired": {}, "decomp": {}, "dose": {}, "G9": {}, "V": {}, "runs": {}}
    L = ["# T09 summary — 5-seed expansion (Electronics, dblp) + Amazon_clothing (r = 0.5)", "",
         "‡ = L의 레이블로 만든 지도 에피소드를 포함하는 arm. † = 분석용 상한(split 이용, 방법 아님). "
         "NaQ-best: Electronics naq_feat, dblp naq_diff, Clothing naq_feat. 모든 짝지은 차이는 두 arm의 공통 seed에서 계산.", "",
         "## 1. 주 표 (seed 평균 ± sd)", "",
         "| dataset | arm | n | seeds | 1s LR | 5s LR | q_purity | anchor_novel_frac | best ep |", "|---|---|---|---|---|---|---|---|---|"]
    for ds in DSS:
        arms = [BEST[ds]] + (["naq_diff"] if ds == C else []) + [NOV, NMX, MIX, SUP, UNL]
        for a in arms:
            rs = {s: load(ds, a, s) for s in SEEDS if load(ds, a, s)}
            if not rs:
                L.append(f"| {ds} | {LAB.get(a, a)} | 0 | - | 미실행 | | | | |"); continue
            v1 = [acc(r) for r in rs.values()]; v5 = [acc(r, "test5_lr") for r in rs.values()]
            row = {"seeds": list(rs), "1s": [float(np.mean(v1)), sd(v1)], "5s": [float(np.mean(v5)), sd(v5)],
                   "q_purity": float(np.mean([r["episode_stats"]["query_purity"] for r in rs.values()])),
                   "anchor_novel_frac": float(np.mean([nfrac(r) for r in rs.values()])),
                   "best_ep": float(np.mean([r["best_episode"] for r in rs.values()]))}
            J["main"][f"{ds}|{a}"] = row
            L.append(f"| {ds} | {LAB.get(a, a)} | {len(rs)} | {','.join(map(str, rs))} | {row['1s'][0]:.2f} ± {row['1s'][1]:.2f} | "
                     f"{row['5s'][0]:.2f} ± {row['5s'][1]:.2f} | {row['q_purity']:.3f} | {row['anchor_novel_frac']:.3f} | {row['best_ep']:.0f} |")

    def comps(ds):
        B = BEST[ds]
        c = [(MIX, B), (MIX, NMX), (NMX, B), (NOV, B), (MIX, NOV), (MIX, SUP), (SUP, B), (UNL, B)]
        return c + ([("naq_diff", "naq_feat")] if ds == C else [])
    L += ["", "## 2. 짝지은 차이", "", "| dataset | 비교 | shot | s0 | s1 | s2 | s3 | s4 | 평균 | sd | 음수/양수 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    M = {}
    for ds in DSS:
        for a, b in comps(ds):
            for k, sh in SH:
                d = paired(ds, a, b, k)
                name = f"{LAB.get(a, a)} − {LAB.get(b, b)}"
                if not d:
                    L.append(f"| {ds} | {name} | {sh} | - | - | - | - | - | 미실행 | | |"); continue
                v = np.array(list(d.values()))
                M[(ds, a, b, sh)] = (float(v.mean()), int((v > 0).sum()), len(v))
                J["paired"][f"{ds}|{a}-{b}|{sh}"] = {"per_seed": d, "mean": float(v.mean()), "sd": sd(v), "neg": int((v < 0).sum()), "pos": int((v > 0).sum())}
                L.append(f"| {ds} | {name} | {sh} | " + " | ".join(f"{d[s]:+.2f}" if s in d else "-" for s in SEEDS) +
                         f" | {v.mean():+.2f} | {sd(v):.2f} | {(v < 0).sum()}/{(v > 0).sum()} |")
    m = lambda ds, a, b, sh="1s": M[(ds, a, b, sh)][0] if (ds, a, b, sh) in M else None
    L += ["", "회수율 (1s; 분자·분모는 각자의 공통 seed 평균 차이):", "", "| dataset | naq_unl† − NaQ-best | R = nov 몫 / 상한 | R_mix = mix_alt 몫 / 상한 |", "|---|---|---|---|"]
    for ds in DSS:
        up, n1, mx = m(ds, UNL, BEST[ds]), m(ds, NOV, BEST[ds]), m(ds, MIX, BEST[ds])
        R = None if up is None or n1 is None else n1 / up
        Rm = None if up is None or mx is None else mx / up
        J["paired"][f"{ds}|recovery"] = {"upper": up, "R": R, "R_mix": Rm}
        L.append(f"| {ds} | {f2(up)} | {'-' if R is None else f'{R:.2f}'} | {'-' if Rm is None else f'{Rm:.2f}'} |")

    L += ["", "## 3. 분해 (1s; 네 arm NaQ-best·naq_mix·nov·mix_alt의 공통 seed만 사용)", "",
          "| dataset | seeds | NaQ→naq_mix (지도 30%) | naq_mix→mix_alt (novel 앵커) | 합 A | NaQ→nov (novel 앵커) | nov→mix_alt (지도 30%) | 합 B |", "|---|---|---|---|---|---|---|---|"]
    for ds in DSS:
        B = BEST[ds]; ss = seeds_of(ds, B, NMX, MIX, NOV)
        if not ss:
            L.append(f"| {ds} | - | 미실행 | | | | | |"); continue
        mm = lambda a, b: float(np.mean(list(paired(ds, a, b, seeds=ss).values())))
        a1, a2, b1, b2 = mm(NMX, B), mm(MIX, NMX), mm(NOV, B), mm(MIX, NOV)
        J["decomp"][ds] = {"seeds": ss, "A": [a1, a2], "B": [b1, b2]}
        L.append(f"| {ds} | {','.join(map(str, ss))} | {a1:+.2f} | {a2:+.2f} | {a1 + a2:+.2f} | {b1:+.2f} | {b2:+.2f} | {b1 + b2:+.2f} |")

    L += ["", "## 4. 용량-반응 (1s)", "",
          "pseudo 앵커 novel 비율: 지도 에피소드가 없는 arm은 그 arm의 anchor_novel_frac seed 평균. 지도 혼합 arm은 pseudo 부분이 같은 무혼합 arm의 값을 쓴다 "
          "(mix_alt → nov2_t1). Δ = arm − NaQ-best (공통 seed 평균). nov2c_t1은 T08 run(Electronics·dblp만).", "",
          "| dataset | arm | pseudo 앵커 novel 비율 | Δ 1s | seeds |", "|---|---|---|---|---|"]
    for ds in DSS:
        B = BEST[ds]; pts = []
        fr = lambda a: J["main"].get(f"{ds}|{a}", {}).get("anchor_novel_frac") if a != NOVC else (
            float(np.mean([nfrac(load(ds, a, s)) for s in seeds_of(ds, a)])) if seeds_of(ds, a) else None)
        for a, src in ((B, B), (NOVC, NOVC), (NOV, NOV), (MIX, NOV), (UNL, UNL)):
            x = fr(src)
            if a == B:
                y, ss = 0.0, [s for s in SEEDS if load(ds, B, s)]
            else:
                d = paired(ds, a, B); y, ss = (float(np.mean(list(d.values()))) if d else None), list(d)
            if x is None or y is None:
                continue
            pts.append((x, y))
            L.append(f"| {ds} | {LAB.get(a, a)} | {x:.3f} | {y:+.2f} | {','.join(map(str, ss))} |")
        rho = float(spearmanr([p[0] for p in pts], [p[1] for p in pts])[0]) if len(pts) >= 5 else None
        J["dose"][ds] = {"points": pts, "spearman": rho}
        L.append(f"| {ds} | **Spearman ρ** ({len(pts)}점) | | {'-' if rho is None else f'{rho:.3f}'} | |")

    pos = lambda ds, a, b: (M[(ds, a, b, "1s")][1], M[(ds, a, b, "1s")][2]) if (ds, a, b, "1s") in M else (None, None)
    pf = lambda t: "-" if t[0] is None else f"{t[0]}/{t[1]}"
    g9a = {ds: {"mean": m(ds, MIX, BEST[ds]), "pos": pf(pos(ds, MIX, BEST[ds]))} for ds in DSS}
    g9b = {"mean": m(E, MIX, NMX), "pos": pf(pos(E, MIX, NMX))}
    g9c = {ds: m(ds, MIX, SUP) for ds in DSS}
    J["G9"] = {"a": g9a, "b": g9b, "c": g9c}
    L += ["", "## 5. G9 계산값 (판정은 사람이 한다)", "",
          "- G9-a (기준: 세 데이터셋 `mix_alt − NaQ-best` 1s 평균 ≥ +1.0; 양수 seed Electronics·dblp ≥ 4/5, Clothing 3/3): " +
          ", ".join(f"{ds} {f2(g9a[ds]['mean'])} (양수 {g9a[ds]['pos']})" for ds in DSS),
          f"- G9-b (기준: Electronics `mix_alt − naq_mix` 1s 평균 ≥ +0.7, 양수 ≥ 4/5): {f2(g9b['mean'])} (양수 {g9b['pos']})",
          "- G9-c (기준: 세 데이터셋 `mix_alt − sup_only` 1s > 0): " + ", ".join(f"{ds} {f2(g9c[ds])}" for ds in DSS)]

    rho_e = J["dose"].get(E, {}).get("spearman")
    V = [("V1", "Electronics mix_alt − naq_mix ≥ +0.7, 양수 ≥ 4/5", f"{f2(m(E, MIX, NMX))} (양수 {pf(pos(E, MIX, NMX))})"),
         ("V2", "dblp naq_mix − NaQ-best ≥ +1.5, |mix_alt − naq_mix| ≤ 1.0", f"{f2(m(D, NMX, BEST[D]))}; {f2(m(D, MIX, NMX))}"),
         ("V3", "mix_alt − NaQ-best: Electronics ≥ +1.3, dblp ≥ +2.0", f"Electronics {f2(m(E, MIX, BEST[E]))}, dblp {f2(m(D, MIX, BEST[D]))}"),
         ("V4", "Clothing nov2_t1 − naq_feat ≥ +1.0, 3/3 양수", f"{f2(m(C, NOV, 'naq_feat'))} (양수 {pf(pos(C, NOV, 'naq_feat'))})"),
         ("V5", "Clothing mix_alt − naq_mix ≥ +0.7", f2(m(C, MIX, NMX))),
         ("V6", "Clothing naq_unl† − naq_feat ≥ +1.5", f2(m(C, UNL, "naq_feat"))),
         ("V7", "Clothing sup_only − naq_feat ≤ −2.0", f2(m(C, SUP, "naq_feat"))),
         ("V8", "Electronics Spearman ρ ≥ 0.8 (§4의 5점)", "-" if rho_e is None else f"{rho_e:.3f}"),
         ("V9", "Clothing naq_feat − naq_diff (1s) > 0", f2(None if m(C, "naq_diff", "naq_feat") is None else -m(C, "naq_diff", "naq_feat")))]
    J["V"] = {v[0]: {"criterion": v[1], "value": v[2]} for v in V}
    L += ["", "## 6. 사전 등록 대조표 (계산값만)", "", "| ID | 기준 | 계산값 |", "|---|---|---|"] + [f"| {a} | {b} | {c} |" for a, b, c in V]

    L += ["", "## 7. T09 실행 범위 (계획 47 run)", "", "| 우선순위 | 계획 | 결과 JSON 있음 | 없음(미실행·제외) | run 시간 합 (분) |", "|---|---|---|---|---|"]
    P = planned()
    for p in ("P1", "P2", "P3", "P4", "P5"):
        rows = [q for q in P if q[0] == p]
        have = [q for q in rows if load(q[1], q[2], q[3])]
        miss = [f"{q[1]} {q[2]} s{q[3]}" for q in rows if not load(q[1], q[2], q[3])]
        t = sum(load(q[1], q[2], q[3])["time"]["total_s"] for q in have) / 60
        J["runs"][p] = {"planned": len(rows), "done": len(have), "missing": miss, "run_min": t}
        L.append(f"| {p} | {len(rows)} | {len(have)} | {'; '.join(miss) if miss else '-'} | {t:.1f} |")
    open("results/T09_summary.md", "w").write("\n".join(L) + "\n")
    json.dump(J, open("results/T09_summary.json", "w"), indent=1, default=str)
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source_check", action="store_true")
    a = ap.parse_args()
    if a.source_check:
        source_check()
    else:
        main_summary()


if __name__ == "__main__":
    main()
