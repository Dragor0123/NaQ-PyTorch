"""T04 aggregation: seed robustness (A) and training-length check (B) -> results/T04_summary.md."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEEDS = [0, 1, 2]
BEST = {"Amazon_eletronics": "naq_feat", "dblp": "naq_diff"}
ARMS = {ds: [b, "lr_only", "cape_lite"] for ds, b in BEST.items()}


def load(ds, arm, seed, suffix=""):
    p = f"results/expB/{ds}_{arm}_s{seed}{suffix}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def acc(r, k):
    return r["test"][k]["mean"]


def ms(a):
    a = np.array(a)
    return f"{a.mean():.2f} ± {a.std(ddof=1):.2f}" if len(a) > 1 else f"{a.mean():.2f}"


def main():
    R = {ds: {a: {s: load(ds, a, s) for s in SEEDS} for a in ARMS[ds]} for ds in BEST}
    L = ["# T04 summary — seed and training-length robustness", "",
         "LR test accuracy (%), 500 fixed test tasks. std는 seed 3개의 표본 표준편차(ddof=1).", "",
         "## (a) seed별", "", "| dataset | arm | seed | 1s LR | 5s LR | query_purity | best ep |", "|---|---|---|---|---|---|---|"]
    for ds in BEST:
        for a in ARMS[ds]:
            for s in SEEDS:
                r = R[ds][a][s]
                if r:
                    L.append(f"| {ds} | {a} | {s} | {acc(r, 'test1_lr'):.2f} | {acc(r, 'test5_lr'):.2f} | "
                             f"{r['episode_stats']['query_purity']:.3f} | {r['best_episode']} |")
    L += ["", "## (b) 요약 (seed 평균 ± 표준편차)", "", "| dataset | arm | n | 1s LR | 5s LR |", "|---|---|---|---|---|"]
    for ds in BEST:
        for a in ARMS[ds]:
            rs = [r for r in R[ds][a].values() if r]
            L.append(f"| {ds} | {a} | {len(rs)} | {ms([acc(r, 'test1_lr') for r in rs])} | {ms([acc(r, 'test5_lr') for r in rs])} |")

    L += ["", "## (c) 짝지은 차이 (같은 seed, arm − NaQ-best)", "",
          "| dataset | 비교 | shot | s0 | s1 | s2 | 평균 | 표준편차 | 음수 개수 / 양수 개수 |", "|---|---|---|---|---|---|---|---|---|"]
    diffs = {}
    for ds, b in BEST.items():
        for a in ("cape_lite", "lr_only"):
            for k, sh in (("test1_lr", "1s"), ("test5_lr", "5s")):
                d = np.array([acc(R[ds][a][s], k) - acc(R[ds][b][s], k) for s in SEEDS
                              if R[ds][a][s] and R[ds][b][s]])
                diffs[(ds, a, sh)] = d
                cells = [f"{x:+.2f}" for x in d] + ["-"] * (3 - len(d))
                L.append(f"| {ds} | {a} − {b} | {sh} | " + " | ".join(cells) +
                         f" | {d.mean():+.2f} | {d.std(ddof=1) if len(d) > 1 else float('nan'):.2f} | {(d < 0).sum()} / {(d > 0).sum()} |")

    # B: training length
    ds = "Amazon_eletronics"
    b8 = {a: load(ds, a, 0, "_e8000") for a in ("naq_feat", "cape_lite")}
    b4 = {a: R[ds][a][0] for a in ("naq_feat", "cape_lite")}
    L += ["", "## (d) 학습 길이 점검 (Amazon_eletronics, seed 0)", ""]
    if all(b8.values()):
        L += ["| arm | episodes | 1s LR | 5s LR | best ep |", "|---|---|---|---|---|"]
        for a in b8:
            for n, r in ((4000, b4[a]), (8000, b8[a])):
                L.append(f"| {a} | {n} | {acc(r, 'test1_lr'):.2f} | {acc(r, 'test5_lr'):.2f} | {r['best_episode']} |")
        for n, rr in ((4000, b4), (8000, b8)):
            L.append(f"\n- cape_lite − naq_feat @ {n}: 1s {acc(rr['cape_lite'], 'test1_lr') - acc(rr['naq_feat'], 'test1_lr'):+.2f}, "
                     f"5s {acc(rr['cape_lite'], 'test5_lr') - acc(rr['naq_feat'], 'test5_lr'):+.2f}")
        L += ["", "valid 곡선 (8,000 run, prototype acc, 1,000 에피소드 간격 발췌; 전체는 그림)", "",
              "| ep | " + " | ".join(str(e) for e in range(0, 8001, 1000)) + " |", "|---|" + "---|" * 9]
        for a, r in b8.items():
            cv = dict(r["valid_curve"])
            L.append(f"| {a} | " + " | ".join(f"{cv[e]:.3f}" for e in range(0, 8001, 1000)) + " |")
        fig, ax = plt.subplots(figsize=(6, 4))
        for a, r in b8.items():
            e, v = zip(*r["valid_curve"]); ax.plot(e, v, label=f"{a} (8000)")
        for a, r in b4.items():
            e, v = zip(*r["valid_curve"]); ax.plot(e, v, "--", alpha=0.6, label=f"{a} (4000)")
        ax.set_xlabel("training episodes"); ax.set_ylabel("valid prototype accuracy"); ax.legend(fontsize=8)
        ax.set_title("Amazon_eletronics, seed 0"); fig.tight_layout()
        fig.savefig("results/T04_valid_curves.png", dpi=120)
        L += ["", "![valid curves](T04_valid_curves.png)"]
    else:
        L.append("B 미실행 (사유는 reports/T04_report.md)")

    # criteria (computed, not judged)
    c1_means = {d: diffs[(d, "cape_lite", "1s")].mean() for d in BEST}
    c1_neg = sum((diffs[(d, "cape_lite", "1s")] < 0).sum() for d in BEST)
    c1_n = sum(len(diffs[(d, "cape_lite", "1s")]) for d in BEST)
    c2 = {d: np.abs(diffs[(d, "lr_only", "1s")]).mean() for d in BEST}
    L += ["", "## (e) 판정 근거 (계산값만; 판정은 사람이 한다)", "",
          "| ID | 계산값 | 기준 | 기준 충족 여부(산술) |", "|---|---|---|---|",
          f"| C1 | 평균(cape_lite − best, 1s): elec {c1_means['Amazon_eletronics']:+.2f}, dblp {c1_means['dblp']:+.2f}; 음수 {c1_neg}/{c1_n} | 둘 다 ≤ −0.5 그리고 음수 ≥ 5/6 | "
          f"{'충족' if all(v <= -0.5 for v in c1_means.values()) and c1_neg >= 5 else '미충족'} |",
          f"| C2 | 평균 abs(lr_only − best, 1s): elec {c2['Amazon_eletronics']:.2f}, dblp {c2['dblp']:.2f} | 둘 다 ≤ 1.0 | "
          f"{'충족' if all(v <= 1.0 for v in c2.values()) else '미충족'} |"]
    if all(b8.values()):
        c3 = acc(b8["cape_lite"], "test1_lr") - acc(b8["naq_feat"], "test1_lr")
        c4 = acc(b8["naq_feat"], "test1_lr") - acc(b4["naq_feat"], "test1_lr")
        L += [f"| C3 | cape_lite − naq_feat @8000 (1s) = {c3:+.2f} | ≥ 0 → 결론 바뀔 수 있음, 아니면 유지 | {'≥ 0' if c3 >= 0 else '< 0'} |",
              f"| C4 | naq_feat 8000 − 4000 (1s) = {c4:+.2f} | ≥ +1.5 → 기본 에피소드 수 재검토 | {'≥ +1.5' if c4 >= 1.5 else '< +1.5'} |"]
    else:
        L += ["| C3 | B 미실행 | | 계산 불가 |", "| C4 | B 미실행 | | 계산 불가 |"]
    open("results/T04_summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
