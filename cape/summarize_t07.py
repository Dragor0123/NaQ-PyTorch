"""T07 aggregation -> results/T07_summary.md, T07_dose_response.png, T07_auroc_vs_gain.png, mechanism (NMI) table."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score

from cape.data import load_graph, load_split

DSS = ["Amazon_eletronics", "dblp", "Amazon_clothing"]
BEST = {"Amazon_eletronics": "naq_feat", "dblp": "naq_diff", "Amazon_clothing": "naq_feat"}
SEEDS = [0, 1, 2]


def load(ds, arm, s, tag=""):
    p = f"results/expB/{ds}_{arm}_s{s}{tag}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def acc(r, k="test1_lr"):
    return r["test"][k]["mean"]


def nf(r):
    st = r["episode_stats"]
    return st.get("anchor_novel_frac", st["anchor_group_frac"]["valid"] + st["anchor_group_frac"]["test"])


def sd(a):
    return float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")


def nmi_rows(ds, emb_name, y, split):
    p = f"cache/emb/{emb_name}.npy"
    if not os.path.exists(p):
        return None
    Z = np.load(p).astype(np.float32)
    out = {}
    for g in ("test", "base"):
        m = np.isin(y, split[g])
        km = KMeans(n_clusters=len(split[g]), n_init=3, random_state=0).fit(Z[m])
        out[g] = float(normalized_mutual_info_score(y[m], km.labels_))
    return out


def main():
    A = json.load(open("results/T07A_estimators.json"))
    E = A["E_star"]
    L = ["# T07 summary — PU novelty estimator, proportional anchoring, mixing", "",
         "† naq_unl / r=1.0: 표준 설정에서는 '레이블 없음 = novel'이라 추정이 자명해짐 (분석용). ‡ mix_alt는 L의 레이블로 지도 에피소드를 씀 (NaQ보다 정보량 많음).", "",
         f"## 1. A 요약 (상세: results/T07A_estimators.md). E* = `{E}`", "",
         "| dataset | r | " + " | ".join(A["mean_auroc_r0.5"]) + " | best | E* τ1 / τ2 기대 novel 비율 | T06 하드 컷 |", "|---|---|" + "---|" * (len(A["mean_auroc_r0.5"]) + 3)]
    for ds in DSS:
        for r in ("0.5", "0.2"):
            d = A["per_ds"][ds][r]
            L.append(f"| {ds} | {r} | " + " | ".join(f"{d['auroc'][e]:.3f}" for e in A["mean_auroc_r0.5"]) + f" | {A['best_per_ds'][ds][r]} | "
                     f"{d['expected_novel_frac'][E]['tau1']:.3f} / {d['expected_novel_frac'][E]['tau2']:.3f} | {d['t06_hardcut_novel_frac']:.3f} |")
    L.append("| mean r=0.5 | | " + " | ".join(f"{v:.3f}" for v in A["mean_auroc_r0.5"].values()) + " | | | |")
    L += ["", "PU 보정 절대 오차 (r=0.5): " + "; ".join(f"{ds} " + ", ".join(f"{e} {A['per_ds'][ds]['0.5']['calib_abs_err'][e]:.3f}" for e in ("pu_feat", "pu_struct", "pu_both")) for ds in DSS)]

    arms_all = lambda ds: [BEST[ds], "nov_r0.5", "unl_uni_r0.5", "nov2_t1_r0.5", "nov2_t2_r0.5", "mix_alt_r0.5", "nov2_t1_r0.2", "naq_unl", "nov2_t1_r1.0"]
    L += ["", "## 2. B 결과", "", "| dataset | arm | seed | 1s LR | 5s LR | q_purity | anchor_novel_frac | anchor_ent | best ep |", "|---|---|---|---|---|---|---|---|---|"]
    agg = {}
    for ds in DSS:
        for a in arms_all(ds):
            rs = {s: load(ds, a, s) for s in SEEDS}
            rs = {s: r for s, r in rs.items() if r}
            mark = " †" if a in ("naq_unl", "nov2_t1_r1.0") else (" ‡" if a.startswith("mix_alt") else "")
            for s, r in rs.items():
                st = r["episode_stats"]
                L.append(f"| {ds} | {a}{mark} | {s} | {acc(r):.2f} | {acc(r, 'test5_lr'):.2f} | {st['query_purity']:.3f} | {nf(r):.3f} | "
                         f"{st['anchor_class_entropy']:.3f} | {r['best_episode']} |")
            if rs:
                v = [acc(r) for r in rs.values()]
                agg[(ds, a)] = {"nf": float(np.mean([nf(r) for r in rs.values()])), "m": float(np.mean(v)), "sd": sd(v), "n": len(rs)}

    L += ["", "## 3. 짝지은 차이 (같은 seed)", "", "| dataset | 비교 | shot | s0 | s1 | s2 | 평균 | sd | 음수/양수 |", "|---|---|---|---|---|---|---|---|---|"]
    D = {}
    for ds in DSS:
        b = BEST[ds]
        comps = [("nov2_t1_r0.5", b), ("nov2_t2_r0.5", b), ("nov2_t1_r0.5", "nov2_t2_r0.5"), ("nov2_t1_r0.5", "nov_r0.5"),
                 ("mix_alt_r0.5", "nov2_t1_r0.5"), ("mix_alt_r0.5", b), ("nov2_t1_r0.2", b), ("nov2_t1_r1.0", b), ("naq_unl", b)]
        for a, ref in comps:
            for k, sh in (("test1_lr", "1s"), ("test5_lr", "5s")):
                d = {s: acc(load(ds, a, s), k) - acc(load(ds, ref, s), k) for s in SEEDS if load(ds, a, s) and load(ds, ref, s)}
                if not d:
                    continue
                v = np.array(list(d.values())); D[(ds, a, ref, sh)] = d
                L.append(f"| {ds} | {a} − {ref} | {sh} | " + " | ".join(f"{d[s]:+.2f}" if s in d else "-" for s in SEEDS) +
                         f" | {v.mean():+.2f} | {sd(v):.2f} | {(v < 0).sum()}/{(v > 0).sum()} |")

    def m(ds, a, ref, sh="1s"):
        d = D.get((ds, a, ref, sh)); return None if not d else float(np.mean(list(d.values())))
    fmt = lambda x: "-" if x is None else f"{x:+.2f}"

    L += ["", "## 4. 회수율 (nov2_t1_r0.5 − NaQ) / (naq_unl − NaQ), seed 평균 차이, 1s", ""]
    for ds in DSS:
        n, u = m(ds, "nov2_t1_r0.5", BEST[ds]), m(ds, "naq_unl", BEST[ds])
        L.append(f"- {ds}: {fmt(n)} / {fmt(u)} = " + (f"{n / u:.2f}" if n is not None and u else "-"))

    # figures
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, ds in zip(axes, DSS):
        for a in [BEST[ds], "unl_uni_r0.5", "nov_r0.5", "nov2_t1_r0.5", "nov2_t2_r0.5", "naq_unl"]:
            if (ds, a) in agg:
                g = agg[(ds, a)]
                ax.errorbar(g["nf"], g["m"], yerr=0 if g["n"] < 2 else g["sd"], fmt="o", capsize=3, label=f"{a} (n={g['n']})")
        ax.set_title(ds); ax.set_xlabel("anchor_novel_frac"); ax.set_ylabel("1-shot LR acc (%)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig("results/T07_dose_response.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    for ds in DSS:
        g = m(ds, "nov2_t1_r0.5", BEST[ds])
        if g is not None:
            x = A["per_ds"][ds]["0.5"]["auroc"][E]
            ax.scatter(x, g); ax.annotate(ds, (x, g), fontsize=8)
    ax.axhline(0, color="gray", lw=0.5); ax.set_xlabel(f"E* ({E}) AUROC, r=0.5"); ax.set_ylabel("nov2_t1_r0.5 − NaQ-best, 1s (%p)")
    fig.tight_layout(); fig.savefig("results/T07_auroc_vs_gain.png", dpi=120); plt.close(fig)
    L += ["", "![dose](T07_dose_response.png) ![auroc-gain](T07_auroc_vs_gain.png)"]

    # mechanism: NMI on stored embeddings
    L += ["", "## 5. 메커니즘: best-checkpoint 임베딩의 k-means NMI (test / base 클래스 노드, 레이블은 분석용)", "",
          "| dataset | run | NMI_test | NMI_base | ΔNMI_test vs NaQ | ΔNMI_base vs NaQ |", "|---|---|---|---|---|---|"]
    mech = {}
    for ds, runs in (("Amazon_eletronics", [("naq_feat_s0_emb", "NaQ (naq_feat, P3)"), ("nov2_t1_r0.5_s0", "nov2_t1"), ("nov2_t2_r0.5_s0", "nov2_t2"),
                                            ("mix_alt_r0.5_s0", "mix_alt ‡"), ("naq_unl_s0_emb", "naq_unl †")]),
                     ("dblp", [("naq_diff_s0_emb", "NaQ (naq_diff, P3)"), ("nov2_t1_r0.5_s0", "nov2_t1"), ("nov2_t2_r0.5_s0", "nov2_t2"),
                               ("mix_alt_r0.5_s0", "mix_alt ‡")])):
        y, split = load_graph(ds)["y"], load_split(ds)
        base_row = None
        for name, label in runs:
            r = nmi_rows(ds, f"{ds}_{name}", y, split)
            if r is None:
                L.append(f"| {ds} | {label} | 미실행 | | | |"); continue
            mech[(ds, label)] = r
            if base_row is None:
                base_row = r
            L.append(f"| {ds} | {label} | {r['test']:.4f} | {r['base']:.4f} | {r['test'] - base_row['test']:+.4f} | {r['base'] - base_row['base']:+.4f} |")

    # G7 and preregistration
    g7a = A["per_ds"]["dblp"]["0.5"]["auroc"][E]
    g7b = []
    for ds in DSS:
        d = D.get((ds, "nov2_t1_r0.5", BEST[ds], "1s"), {})
        if d:
            g7b.append(f"{ds}: 평균 {np.mean(list(d.values())):+.2f}, 양수 {sum(v > 0 for v in d.values())}/{len(d)}")
    L += ["", "## 6. G7 계산값 (판정은 사람이 한다)", "", f"- G7-a: E* ({E})의 dblp AUROC (r=0.5) = {g7a:.3f} (기준 ≥ 0.65)",
          "- G7-b: nov2_t1_r0.5 − NaQ-best (1s) — " + "; ".join(g7b) + " (기준: 평균 ≥ +1.0, seed 모두 양수; clothing은 seed 2개)"]
    P = A["per_ds"]
    E_ = "Amazon_eletronics"
    s10 = [mech.get((E_, k)) for k in ("NaQ (naq_feat, P3)", "nov2_t1", "naq_unl †")]
    s10s = ("NMI_test NaQ / nov2_t1 / naq_unl = " + " / ".join(f"{x['test']:.4f}" for x in s10) +
            f"; |ΔNMI_base| {abs(s10[1]['base'] - s10[0]['base']):.4f} vs |ΔNMI_test| {abs(s10[1]['test'] - s10[0]['test']):.4f} (nov2_t1 vs NaQ)") if all(s10) else "임베딩 일부 없음"
    L += ["", "## 7. 사전 등록 대조표 (계산값만)", "", "| ID | 기준 | 계산값 |", "|---|---|---|",
          "| S1 | r=0.5 pu_both − cent_sgc AUROC ≥ +0.05 (3개 중 ≥2) | " + ", ".join(f"{ds} {P[ds]['0.5']['auroc']['pu_both'] - P[ds]['0.5']['auroc']['cent_sgc']:+.3f}" for ds in ["Amazon_clothing", "Amazon_eletronics", "dblp"]) + " |",
          f"| S2 | E* dblp AUROC (r=0.5) ≥ 0.65 | {g7a:.3f} |",
          f"| S3 | elec nov2_t1_r0.5 − NaQ ≥ +1.3 | {fmt(m(E_, 'nov2_t1_r0.5', BEST[E_]))} |",
          f"| S4 | elec nov2_t2 − nov2_t1 > 0 | {fmt(None if m(E_, 'nov2_t1_r0.5', 'nov2_t2_r0.5') is None else -m(E_, 'nov2_t1_r0.5', 'nov2_t2_r0.5'))} |",
          f"| S5 | dblp nov2_t1 − NaQ (S2 성립 시 ≥ +0.5, 아니면 abs < 0.7) | {fmt(m('dblp', 'nov2_t1_r0.5', BEST['dblp']))} (S2 계산값 {g7a:.3f}) |",
          f"| S6 | mix_alt − NaQ ≥ 0 (둘 다), elec mix_alt − nov2_t1 ∈ [−0.5, +1.5] | elec {fmt(m(E_, 'mix_alt_r0.5', BEST[E_]))}, dblp {fmt(m('dblp', 'mix_alt_r0.5', BEST['dblp']))}; elec mix − nov2_t1 {fmt(m(E_, 'mix_alt_r0.5', 'nov2_t1_r0.5'))} |",
          f"| S7 | abs((nov2_t1_r1.0 − NaQ) − (naq_unl − NaQ)) ≤ 0.7 (elec) | nov2_t1_r1.0 {fmt(m(E_, 'nov2_t1_r1.0', BEST[E_]))} vs naq_unl {fmt(m(E_, 'naq_unl', BEST[E_]))} (같은 seed 기준) |",
          f"| S8 | elec nov2_t1_r0.2 − NaQ ≥ +0.7 (seed 2개) | {fmt(m(E_, 'nov2_t1_r0.2', BEST[E_]))} |",
          f"| S9 | clothing nov2_t1_r0.5 − naq_feat ≥ +1.0, 모두 양수 | {fmt(m('Amazon_clothing', 'nov2_t1_r0.5', 'naq_feat'))}; seed별 " +
          ", ".join(f"{v:+.2f}" for v in D.get(("Amazon_clothing", "nov2_t1_r0.5", "naq_feat", "1s"), {}).values()) + " |",
          f"| S10 | elec NMI_test NaQ < nov2_t1 < naq_unl, abs(ΔNMI_base) < abs(ΔNMI_test) | {s10s} |"]
    open("results/T07_summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
