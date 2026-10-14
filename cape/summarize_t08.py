"""T08 aggregation -> results/T08_summary.md (controls for mixing, calibrated PU, decomposition, NMI, G8, U1-U7)."""
import json
import os

import numpy as np

from cape.data import load_graph, load_split
from cape.summarize_t07 import nmi_rows

DSS = ["Amazon_eletronics", "dblp"]
BEST = {"Amazon_eletronics": "naq_feat", "dblp": "naq_diff"}
SEEDS = [0, 1, 2]
ARMS = ["naq_mix_r0.5", "sup_only_r0.5", "mix_alt_r0.5", "mix_alt_p10_r0.5", "mix_alt_p50_r0.5", "nov2_t1_r0.5", "nov2c_t1_r0.5"]
SUP = {"naq_mix_r0.5", "sup_only_r0.5", "mix_alt_r0.5", "mix_alt_p10_r0.5", "mix_alt_p50_r0.5"}


def load(ds, arm, s, tag=""):
    p = f"results/expB/{ds}_{arm}_s{s}{tag}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def acc(r, k="test1_lr"):
    return r["test"][k]["mean"]


def sd(a):
    return float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")


def paired(ds, a, b, k="test1_lr"):
    return {s: acc(load(ds, a, s), k) - acc(load(ds, b, s), k) for s in SEEDS if load(ds, a, s) and load(ds, b, s)}


def main():
    A = json.load(open("results/T08A_calibration.json"))
    E2 = A["E2_star"]
    L = ["# T08 summary — mixing controls and PU calibration (r = 0.5)", "",
         "‡ = L의 레이블로 만든 지도 에피소드를 포함하는 arm (NaQ보다 정보량 많음).", "",
         f"## 1. A′ (상세: results/T08A_calibration.md). E2* = `{E2}` — {A['rule_applied']}", "",
         "| dataset | variant | AUROC | 보정 오차 (부호) | 적합 시간 s |", "|---|---|---|---|---|"]
    for ds, d in A["per_ds"].items():
        for e, v in d["variants"].items():
            L.append(f"| {ds} | {e} | {v['auroc']:.3f} | {v['calib_err']:+.3f} | {v['fit_s']:.1f} |")
    L += ["", "## 2. 결과", "", "| dataset | arm | seed | 1s LR | 5s LR | q_purity | anchor_novel_frac | anchor_ent | best ep |", "|---|---|---|---|---|---|---|---|---|"]
    for ds in DSS:
        for a in [BEST[ds]] + ARMS:
            for s in SEEDS:
                r = load(ds, a, s)
                if r:
                    st = r["episode_stats"]
                    L.append(f"| {ds} | {a}{' ‡' if a in SUP else ''} | {s} | {acc(r):.2f} | {acc(r, 'test5_lr'):.2f} | {st['query_purity']:.3f} | "
                             f"{st.get('anchor_novel_frac', st['anchor_group_frac']['valid'] + st['anchor_group_frac']['test']):.3f} | "
                             f"{st['anchor_class_entropy']:.3f} | {r['best_episode']} |")
    comps = lambda ds: [("naq_mix_r0.5", BEST[ds]), ("mix_alt_r0.5", "naq_mix_r0.5"), ("naq_mix_r0.5", "nov2_t1_r0.5"),
                        ("sup_only_r0.5", BEST[ds]), ("mix_alt_r0.5", "sup_only_r0.5"), ("mix_alt_p10_r0.5", BEST[ds]),
                        ("mix_alt_r0.5", BEST[ds]), ("mix_alt_p50_r0.5", BEST[ds]), ("nov2c_t1_r0.5", BEST[ds]), ("nov2c_t1_r0.5", "nov2_t1_r0.5")]
    L += ["", "## 3. 짝지은 차이 (같은 seed)", "", "| dataset | 비교 | shot | s0 | s1 | s2 | 평균 | sd | 음수/양수 |", "|---|---|---|---|---|---|---|---|---|"]
    D = {}
    for ds in DSS:
        for a, b in comps(ds):
            for k, sh in (("test1_lr", "1s"), ("test5_lr", "5s")):
                d = paired(ds, a, b, k)
                if not d:
                    L.append(f"| {ds} | {a} − {b} | {sh} | - | - | - | 미실행 | | |"); continue
                v = np.array(list(d.values())); D[(ds, a, b, sh)] = d
                L.append(f"| {ds} | {a} − {b} | {sh} | " + " | ".join(f"{d[s]:+.2f}" if s in d else "-" for s in SEEDS) +
                         f" | {v.mean():+.2f} | {sd(v):.2f} | {(v < 0).sum()}/{(v > 0).sum()} |")
    mean = lambda ds, a, b, sh="1s": None if (ds, a, b, sh) not in D else float(np.mean(list(D[(ds, a, b, sh)].values())))
    f = lambda x: "-" if x is None else f"{x:+.2f}"

    L += ["", "## 4. 분해 (1s, 각 단계는 해당 두 arm의 공통 seed에 대한 짝지은 차이의 평균)", "",
          "| dataset | 경로 A: NaQ→+지도30%(naq_mix) | →+novel 앵커(mix_alt) | 합 A | 경로 B: NaQ→+novel 앵커(nov2_t1) | →+지도30%(mix_alt) | 합 B | 합 A − 합 B | 직접 mix_alt − NaQ |",
          "|---|---|---|---|---|---|---|---|---|"]
    dec = {}
    for ds in DSS:
        a1, a2 = mean(ds, "naq_mix_r0.5", BEST[ds]), mean(ds, "mix_alt_r0.5", "naq_mix_r0.5")
        n1 = mean(ds, "nov2_t1_r0.5", BEST[ds]) if (ds, "nov2_t1_r0.5", BEST[ds], "1s") in D else None
        if n1 is None:
            d = paired(ds, "nov2_t1_r0.5", BEST[ds]); n1 = float(np.mean(list(d.values())))
        d2 = paired(ds, "mix_alt_r0.5", "nov2_t1_r0.5"); n2 = float(np.mean(list(d2.values())))
        tot = mean(ds, "mix_alt_r0.5", BEST[ds])
        sa = None if a1 is None or a2 is None else a1 + a2
        dec[ds] = (sa, n1 + n2)
        L.append(f"| {ds} | {f(a1)} | {f(a2)} | {f(sa)} | {n1:+.2f} (s0–2) | {n2:+.2f} (s0–1) | {n1 + n2:+.2f} | {f(None if sa is None else sa - (n1 + n2))} | {f(tot)} |")
    L.append("\n같은 seed 집합이면 두 경로 합은 대수적으로 같다 (= mix_alt − NaQ). 차이는 nov2_t1 단계가 seed 0–2를 쓰고 나머지가 seed 0–1을 쓰기 때문에 생긴다.")

    # NMI
    L += ["", "## 5. 메커니즘 NMI (seed 0, best checkpoint 임베딩; T07 행 재계산 포함)", "", "| dataset | run | NMI_test | NMI_base | ΔNMI_test vs NaQ | ΔNMI_base vs NaQ |", "|---|---|---|---|---|---|"]
    for ds, nq in (("Amazon_eletronics", "naq_feat_s0_emb"), ("dblp", "naq_diff_s0_emb")):
        y, split = load_graph(ds)["y"], load_split(ds)
        base = None
        for name, lab in ((nq, "NaQ (P3, T07)"), ("nov2_t1_r0.5_s0", "nov2_t1 (T07)"), ("mix_alt_r0.5_s0", "mix_alt ‡ (T07)"),
                          ("naq_mix_r0.5_s0", "naq_mix ‡"), ("sup_only_r0.5_s0", "sup_only ‡"), ("nov2c_t1_r0.5_s0", "nov2c_t1")):
            r = nmi_rows(ds, f"{ds}_{name}", y, split)
            if r is None:
                L.append(f"| {ds} | {lab} | 미실행 | | | |"); continue
            base = base or r
            L.append(f"| {ds} | {lab} | {r['test']:.4f} | {r['base']:.4f} | {r['test'] - base['test']:+.4f} | {r['base'] - base['base']:+.4f} |")

    E, Db = "Amazon_eletronics", "dblp"
    g8a = D.get((E, "mix_alt_r0.5", "naq_mix_r0.5", "1s"), {})
    pv = lambda ds, a: (lambda r: None if r is None else acc(r))(load(ds, a, 0))
    L += ["", "## 6. G8 계산값 (판정은 사람이 한다)", "",
          f"- G8-a: eletronics mix_alt − naq_mix (1s) = 평균 {f(mean(E, 'mix_alt_r0.5', 'naq_mix_r0.5'))}; seed별 " + ", ".join(f"s{s} {v:+.2f}" for s, v in g8a.items()) + " (기준 ≥ +0.7, 2 seed 모두 양수)",
          f"- G8-b: dblp mix_alt − sup_only (1s) = {f(mean(Db, 'mix_alt_r0.5', 'sup_only_r0.5'))} (기준 ≥ +0.7)",
          f"- G8-c: E2* ({E2}) 보정 오차 = " + ", ".join(f"{ds} {A['per_ds'][ds]['variants'][E2]['calib_err']:+.3f}" for ds in DSS) + " (기준: 절댓값 ≤ 0.15, 두 데이터셋)"]
    t07 = {ds: A["per_ds"][ds]["t07"]["pu_feat"]["auroc"] for ds in DSS}
    pf2 = {ds: A["per_ds"][ds]["variants"]["pu_feat2"] for ds in DSS}
    p = {ds: {k: pv(ds, a) for k, a in (("p10", "mix_alt_p10_r0.5"), ("p30", "mix_alt_r0.5"), ("p50", "mix_alt_p50_r0.5"))} for ds in DSS}
    fp = lambda d: " / ".join("-" if d[k] is None else f"{d[k]:.2f}" for k in ("p10", "p30", "p50"))
    ev = [v for v in p[E].values() if v is not None]
    L += ["", "## 7. 사전 등록 대조표 (계산값만)", "", "| ID | 기준 | 계산값 |", "|---|---|---|",
          "| U1 | pu_feat2 보정 오차 ≤ 0.15 (둘 다), AUROC 하락 ≤ 0.03 (T07 pu_feat 대비) | " +
          ", ".join(f"{ds} 오차 {pf2[ds]['calib_err']:+.3f}, AUROC {pf2[ds]['auroc']:.3f} vs {t07[ds]:.3f} ({pf2[ds]['auroc'] - t07[ds]:+.3f})" for ds in DSS) + " |",
          f"| U2 | elec nov2c_t1 − NaQ ≥ +1.3 | {f(mean(E, 'nov2c_t1_r0.5', BEST[E]))} |",
          f"| U3 | elec mix_alt − naq_mix ≥ +0.7, 2 seed 모두 양수 | {f(mean(E, 'mix_alt_r0.5', 'naq_mix_r0.5'))}; " + ", ".join(f"s{s} {v:+.2f}" for s, v in g8a.items()) + " |",
          f"| U4 | dblp abs(mix_alt − naq_mix) ≤ 1.0 | {f(mean(Db, 'mix_alt_r0.5', 'naq_mix_r0.5'))} |",
          f"| U5 | elec sup_only − NaQ ≤ −2.0; dblp sup_only − NaQ ≤ +1.0 | elec {f(mean(E, 'sup_only_r0.5', BEST[E]))}, dblp {f(mean(Db, 'sup_only_r0.5', BEST[Db]))} |",
          f"| U6 | dblp p50 > p30 > p10 (s0); elec 범위 ≤ 1.0 | dblp p10/p30/p50 = {fp(p[Db])}; elec = {fp(p[E])} (범위 {max(ev) - min(ev):.2f}) |" if len(ev) == 3 else
          f"| U6 | dblp p50 > p30 > p10 (s0); elec 범위 ≤ 1.0 | dblp {fp(p[Db])}; elec {fp(p[E])} |",
          f"| U7 | 두 경로 총합 차이 ≤ 0.7 (elec) | 합 A {f(dec[E][0])}, 합 B {f(dec[E][1])}, 차이 {f(None if dec[E][0] is None else dec[E][0] - dec[E][1])} |"]
    open("results/T08_summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
