"""Final aggregation: Exp B table, gate evidence (G1-G3), reproduction check, figure -> results/summary.md."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cape.generators import ARMS

DSS = ["Amazon_eletronics", "dblp"]
NAQ_PAPER = {"Amazon_eletronics": {"naq_feat": (76.46, 88.72), "naq_diff": (68.56, 84.88)},
             "dblp": {"naq_feat": (71.38, 82.34), "naq_diff": (73.89, 82.24)}}
NOISE = 1.5


def nz(x):
    return f"{x:+.2f}" + (" (잡음 범위)" if abs(x) < NOISE else "")


def main():
    R = {ds: {a: json.load(open(f"results/expB/{ds}_{a}_s0.json")) for a in ARMS
              if os.path.exists(f"results/expB/{ds}_{a}_s0.json")} for ds in DSS}
    acc = lambda ds, a, k="test1_lr": R[ds][a]["test"][k]["mean"]
    L = ["# CaPE validation summary (T03)", "", "1-shot LR test accuracy (%) unless noted. seed 0, 500 test tasks; "
         f"differences < {NOISE}%p are within noise.", "", "## (a) Exp B", "",
         "| dataset | arm | q_purity | way_coll | q_overlap | anchor_uniq | anchor_ent | base/valid/test anchors | 1s LR | 5s LR | 1s proto | best ep / total | time (s) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds in DSS:
        for a, r in R[ds].items():
            s, t = r["episode_stats"], r["test"]
            gf = s["anchor_group_frac"]
            L.append(f"| {ds} | {a} | {s['query_purity']:.3f} | {s['way_collision']:.3f} | {s['query_overlap']:.3f} | "
                     f"{s['anchor_unique_frac']:.3f} | {s['anchor_class_entropy']:.3f} | {gf['base']:.2f}/{gf['valid']:.2f}/{gf['test']:.2f} | "
                     f"{t['test1_lr']['mean']:.2f} ± {t['test1_lr']['ci95']:.2f} | {t['test5_lr']['mean']:.2f} ± {t['test5_lr']['ci95']:.2f} | "
                     f"{t['test1_proto']['mean']:.2f} | {r['best_episode']} / {r['config']['episodes']} | {r['time']['total_s']:.0f} |")

    # gates
    exA = {ds: json.load(open(f"results/expA/{ds}.json"))["G1"] for ds in ["Amazon_clothing"] + DSS}
    L += ["", "## (b) Gates", "", "### G1 (from results/expA)", "", "| dataset | R* | Δ_lr | Spearman | Δ≥0.03 | ρ≥0.3 |", "|---|---|---|---|---|---|"]
    for ds, g in exA.items():
        L.append(f"| {ds} | {g['R_star']} | {g['delta_lr']:+.3f} | {g['spearman_test']:.3f} | {'PASS' if g['pass_delta'] else 'FAIL'} | {'PASS' if g['pass_spearman'] else 'FAIL'} |")
    L.append(f"\nG1 Δ 조건 통과 {sum(g['pass_delta'] for g in exA.values())}/3 (기준 ≥2)")
    best = {ds: max(("naq_feat", "naq_diff"), key=lambda a: acc(ds, a)) for ds in DSS}
    nb = {ds: acc(ds, best[ds]) for ds in DSS}
    d_or = {ds: acc(ds, "oracle") - nb[ds] for ds in DSS}
    d_cl = {ds: acc(ds, "cape_lite") - nb[ds] for ds in DSS}
    m_or, m_cl = sum(d_or.values()) / 2, sum(d_cl.values()) / 2
    L += ["", "### G2 / G3", "", "| dataset | NaQ-best | NaQ-best acc | oracle − NaQ-best | cape_lite − NaQ-best | (oracle − NaQ-best)/3 | cape_lite ≥ NaQ-best − 0.5 |",
          "|---|---|---|---|---|---|---|"]
    for ds in DSS:
        L.append(f"| {ds} | {best[ds]} | {nb[ds]:.2f} | {nz(d_or[ds])} | {nz(d_cl[ds])} | {d_or[ds] / 3:+.2f} | {'PASS' if d_cl[ds] >= -0.5 else 'FAIL'} |")
    L.append(f"| mean | | | {nz(m_or)} | {nz(m_cl)} | {m_or / 3:+.2f} | |")
    g3a = m_cl >= m_or / 3
    g3b = all(d_cl[ds] >= -0.5 for ds in DSS)
    L += ["", f"- G2: mean(oracle − NaQ-best) = {m_or:+.2f} (기준 ≥ +2.0) → {'PASS' if m_or >= 2.0 else 'FAIL'}",
          f"- G3: mean(cape_lite − NaQ-best) = {m_cl:+.2f} vs mean(oracle − NaQ-best)/3 = {m_or / 3:+.2f} → {'PASS' if g3a else 'FAIL'}; "
          f"per-dataset ≥ −0.5: {'PASS' if g3b else 'FAIL'} → G3 {'PASS' if g3a and g3b else 'FAIL'}",
          "", "### 보조 비교 (게이트 아님)", "", "| dataset | lr_only − NaQ-best | cape_lite − lr_only | oracle_base − NaQ-best | oracle − oracle_base |",
          "|---|---|---|---|---|"]
    aux = {ds: (acc(ds, "lr_only") - nb[ds], acc(ds, "cape_lite") - acc(ds, "lr_only"),
                acc(ds, "oracle_base") - nb[ds], acc(ds, "oracle") - acc(ds, "oracle_base")) for ds in DSS}
    for ds in DSS:
        L.append(f"| {ds} | " + " | ".join(nz(x) for x in aux[ds]) + " |")
    L.append("| mean | " + " | ".join(nz(sum(aux[ds][i] for ds in DSS) / 2) for i in range(4)) + " |")
    L.append("\noracle_base는 G2·G3 계산에 쓰지 않았다.")

    # reproduction
    L += ["", "## (c) 재현 점검 (NaQ Table 15/16, ProtoNet LR probing)", "",
          "| dataset | arm | ours 1s | paper 1s | Δ1s | ours 5s | paper 5s | Δ5s |", "|---|---|---|---|---|---|---|---|"]
    for ds in DSS:
        for a in ("naq_feat", "naq_diff"):
            o1, o5 = acc(ds, a), acc(ds, a, "test5_lr")
            p1, p5 = NAQ_PAPER[ds][a]
            flag = lambda d: f"{d:+.2f}" + (" **>5%p**" if abs(d) > 5 else "")
            L.append(f"| {ds} | {a} | {o1:.2f} | {p1:.2f} | {flag(o1 - p1)} | {o5:.2f} | {p5:.2f} | {flag(o5 - p5)} |")
    for ds in DSS:
        L.append(f"\n{ds}: feat − diff = ours {acc(ds, 'naq_feat') - acc(ds, 'naq_diff'):+.2f}, paper "
                 f"{NAQ_PAPER[ds]['naq_feat'][0] - NAQ_PAPER[ds]['naq_diff'][0]:+.2f}")

    # underfitting
    runs = [(ds, a, r["best_episode"], r["config"]["episodes"]) for ds in DSS for a, r in R[ds].items()]
    late = [x for x in runs if x[2] > 0.8 * x[3]]
    late_main = [x for x in late if x[1] != "oracle_base"]
    n_main = sum(x[1] != "oracle_base" for x in runs)
    L += ["", "## (e) 과소학습 점검", "",
          f"- best 에피소드가 마지막 20%(> {int(0.8 * runs[0][3])})에 있는 run: {len(late)}/{len(runs)} (oracle_base 제외 {len(late_main)}/{n_main})",
          f"- {'**학습 부족 가능성**' if len(late) > len(runs) / 2 else '과반 아님'}",
          "- " + ", ".join(f"{ds}/{a}:{b}" for ds, a, b, _ in runs)]

    # figure
    fig, ax = plt.subplots(figsize=(6, 4.5))
    marks = {"Amazon_eletronics": "o", "dblp": "s"}
    colors = dict(zip(ARMS, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    for ds in DSS:
        for a, r in R[ds].items():
            ax.scatter(r["episode_stats"]["query_purity"], acc(ds, a), marker=marks[ds], color=colors[a], s=50)
    for a in ARMS:
        ax.scatter([], [], color=colors[a], label=a)
    for ds in DSS:
        ax.scatter([], [], marker=marks[ds], color="gray", label=ds)
    ax.set_xlabel("episode query purity"); ax.set_ylabel("5-way 1-shot LR test accuracy (%)")
    ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig("results/expB/purity_vs_acc.png", dpi=120)
    L += ["", "## (d) 그림", "", "![purity vs acc](expB/purity_vs_acc.png)"]
    open("results/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
