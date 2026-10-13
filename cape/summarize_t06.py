"""T06 aggregation -> results/T06_summary.md, results/T06_dose_response.png."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DSS = ["Amazon_eletronics", "dblp"]
BEST = {"Amazon_eletronics": "naq_feat", "dblp": "naq_diff"}
SEEDS = [0, 1, 2]


def load(ds, arm, s):
    p = f"results/expB/{ds}_{arm}_s{s}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def novel_frac(r):
    st = r["episode_stats"]
    return st.get("anchor_novel_frac", st["anchor_group_frac"]["valid"] + st["anchor_group_frac"]["test"])


def acc(r, k="test1_lr"):
    return r["test"][k]["mean"]


def arms(ds):
    return [BEST[ds], "unl_uni_r0.5", "nov_r0.5", "naq_unl", "unl_uni_r0.2", "nov_r0.2"]


def sd(a):
    return float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")


def main():
    A = json.load(open("results/T06A_novelty.json"))
    L = ["# T06 summary — label-scarce setting (r = base label ratio) and novelty anchoring", "",
         "† naq_unl = r=1.0 분석용 상한 (레이블 없음 = novel 구조 이용). NaQ-best는 레이블을 쓰지 않아 r과 무관 (T03/T04 재사용).", "",
         "## 1. A 요약", "", "| dataset | r | U: hidden base / valid / test | AUROC sgc / raw | A_nov novel frac | novel cls covered | gini |",
         "|---|---|---|---|---|---|---|"]
    for ds in DSS:
        for r, d in A[ds].items():
            u = d["U_comp"]
            L.append(f"| {ds} | {r} | {u['hidden_base']:.3f} / {u['valid']:.3f} / {u['test']:.3f} | {d['auroc_sgc']:.3f} / {d['auroc_raw']:.3f} | "
                     f"{d['expected_anchor_novel_frac']['nov']:.3f} | {d['A_nov_novel_class_coverage']:.3f} | {d['A_nov_class_incl_gini']:.3f} |")

    L += ["", "## 2. 결과 (dataset × arm × seed)", "", "| dataset | arm | seed | 1s LR | 5s LR | q_purity | anchor_novel_frac | anchor_ent | best ep |",
          "|---|---|---|---|---|---|---|---|---|"]
    agg = {}
    for ds in DSS:
        for a in arms(ds):
            rs = {s: load(ds, a, s) for s in SEEDS}
            rs = {s: r for s, r in rs.items() if r}
            for s, r in rs.items():
                st = r["episode_stats"]
                L.append(f"| {ds} | {a}{' †' if a == 'naq_unl' else ''} | {s} | {acc(r):.2f} | {acc(r, 'test5_lr'):.2f} | {st['query_purity']:.3f} | "
                         f"{novel_frac(r):.3f} | {st['anchor_class_entropy']:.3f} | {r['best_episode']} |")
            if rs:
                agg[(ds, a)] = {"nf": float(np.mean([novel_frac(r) for r in rs.values()])),
                                "m": float(np.mean([acc(r) for r in rs.values()])), "sd": sd([acc(r) for r in rs.values()]), "n": len(rs)}

    L += ["", "## 3. 짝지은 차이 (같은 seed)", "", "| dataset | 비교 | shot | s0 | s1 | s2 | 평균 | 표준편차 | 음수/양수 |", "|---|---|---|---|---|---|---|---|---|"]
    D = {}
    for ds in DSS:
        b = BEST[ds]
        for a, ref in (("unl_uni_r0.5", b), ("nov_r0.5", b), ("nov_r0.5", "unl_uni_r0.5"), ("naq_unl", b),
                       ("unl_uni_r0.2", b), ("nov_r0.2", b)):
            for k, sh in (("test1_lr", "1s"), ("test5_lr", "5s")):
                d = {s: acc(load(ds, a, s), k) - acc(load(ds, ref, s), k) for s in SEEDS if load(ds, a, s) and load(ds, ref, s)}
                if not d:
                    continue
                v = np.array(list(d.values()))
                D[(ds, a, ref, sh)] = d
                L.append(f"| {ds} | {a} − {ref}{' †' if a == 'naq_unl' else ''} | {sh} | " +
                         " | ".join(f"{d[s]:+.2f}" if s in d else "-" for s in SEEDS) +
                         f" | {v.mean():+.2f} | {sd(v):.2f} | {(v < 0).sum()}/{(v > 0).sum()} |")

    def mean(ds, a, ref, sh="1s"):
        d = D.get((ds, a, ref, sh))
        return None if not d else float(np.mean(list(d.values())))

    L += ["", "## 4. 회수율 R = (nov_r0.5 − NaQ-best) / (naq_unl − NaQ-best), seed 평균 차이로 계산 (1s)", "",
          "| dataset | nov − best | naq_unl − best † | R | 사용 seed |", "|---|---|---|---|---|"]
    for ds in DSS:
        n, u = mean(ds, "nov_r0.5", BEST[ds]), mean(ds, "naq_unl", BEST[ds])
        seeds = sorted(set(D.get((ds, "nov_r0.5", BEST[ds], "1s"), {})) | set(D.get((ds, "naq_unl", BEST[ds], "1s"), {})))
        L.append(f"| {ds} | {n:+.2f} | {u:+.2f} | {n / u:.2f} | nov {sorted(D[(ds, 'nov_r0.5', BEST[ds], '1s')])}, naq_unl {sorted(D[(ds, 'naq_unl', BEST[ds], '1s')])} |"
                 if n is not None and u is not None else f"| {ds} | - | - | - | - |")

    # G6 and preregistration (computed values only)
    E = "Amazon_eletronics"
    g6e = D.get((E, "nov_r0.5", BEST[E], "1s"), {})
    g6d = mean("dblp", "nov_r0.5", BEST["dblp"])
    L += ["", "## 5. G6 계산값 (판정은 사람이 한다)", "",
          f"- eletronics: mean(nov_r0.5 − naq_feat) = {np.mean(list(g6e.values())):+.2f}; seed별 {', '.join(f's{s} {v:+.2f}' for s, v in g6e.items())}; 양수 {sum(v > 0 for v in g6e.values())}/{len(g6e)} (기준: ≥ +1.0, 3개 모두 양수)",
          f"- dblp: mean(nov_r0.5 − naq_diff) = {g6d:+.2f} (기준: ≥ −0.5)"]
    m = lambda ds, a: agg[(ds, a)]["m"] if (ds, a) in agg else float("nan")
    r8a, r8b = (D.get((E, "unl_uni_r0.2", BEST[E], "1s"), {}).get(0), D.get((E, "unl_uni_r0.5", BEST[E], "1s"), {}).get(0))
    L += ["", "## 6. 사전 등록 대조표 (계산값만)", "", "| ID | 기준 | 계산값 |", "|---|---|---|",
          f"| R1 | AUROC ≥ .65 (두 데이터셋, r=0.5) | elec {A[E]['0.5']['auroc_sgc']:.3f}, dblp {A['dblp']['0.5']['auroc_sgc']:.3f} (sgc, 사용 점수) |",
          f"| R2 | eletronics A_nov novel 비율 ≥ .70 | {A[E]['0.5']['expected_anchor_novel_frac']['nov']:.3f} |",
          f"| R3 | elec unl_uni_r0.5 − NaQ-best ∈ [+0.2, +1.6] | {mean(E, 'unl_uni_r0.5', BEST[E]):+.2f} (seed 평균) |",
          f"| R4 | G6 eletronics 조건 | 평균 {np.mean(list(g6e.values())):+.2f}, 양수 {sum(v > 0 for v in g6e.values())}/{len(g6e)} |",
          f"| R5 | elec mean(nov − unl_uni) > 0 | {mean(E, 'nov_r0.5', 'unl_uni_r0.5'):+.2f} |",
          f"| R6 | G6 dblp 조건 | {g6d:+.2f} |",
          f"| R7 | elec seed 평균 NaQ < unl_uni < nov < naq_unl | {m(E, BEST[E]):.2f} / {m(E, 'unl_uni_r0.5'):.2f} / {m(E, 'nov_r0.5'):.2f} / {m(E, 'naq_unl'):.2f} |",
          f"| R8 | elec s0: (unl_uni_r0.2 − NaQ) < (unl_uni_r0.5 − NaQ) | " +
          (f"{r8a:+.2f} vs {r8b:+.2f} |" if r8a is not None else "P3a 미실행 |")]
    L.append("\nNaQ-best·naq_unl의 seed 평균은 사용 가능한 seed 전체(NaQ-best 0–2, naq_unl 0–2)로 계산.")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, ds in zip(axes, DSS):
        for a in arms(ds):
            if (ds, a) in agg:
                g = agg[(ds, a)]
                ax.errorbar(g["nf"], g["m"], yerr=0 if g["n"] < 2 else g["sd"], fmt="o", capsize=3, label=f"{a} (n={g['n']})")
        ax.set_title(ds); ax.set_xlabel("anchor_novel_frac"); ax.set_ylabel("5-way 1-shot LR acc (%)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig("results/T06_dose_response.png", dpi=120)
    L += ["", "![dose response](T06_dose_response.png)"]
    open("results/T06_summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
