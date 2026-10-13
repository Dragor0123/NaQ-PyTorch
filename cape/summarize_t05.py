"""T05 aggregation: A summary, B table, paired differences, preregistration comparison -> results/T05_summary.md."""
import json
import os

import numpy as np

DSS = ["Amazon_eletronics", "dblp"]
BEST = {"Amazon_eletronics": "naq_feat", "dblp": "naq_diff"}
NOISE = {"Amazon_eletronics": 1.0, "dblp": 1.5}
B_ARMS = ["naq_unl", "lr_soft", "naq_hard", "naq_easy"]


def load(ds, arm, seed):
    p = f"results/expB/{ds}_{arm}_s{seed}.json"
    return json.load(open(p)) if os.path.exists(p) else None


def acc(r, k):
    return r["test"][k]["mean"]


def paired(ds, a, b, seeds, k):
    return {s: acc(load(ds, a, s), k) - acc(load(ds, b, s), k) for s in seeds if load(ds, a, s) and load(ds, b, s)}


def main():
    A = json.load(open("results/T05A_diag.json"))
    L = ["# T05 summary — mechanism diagnosis (A) and disentangling arms (B)", "",
         "## 1. A 요약 (상세: results/T05A_diag.md)", "", "| dataset | ρ(π̂,deg) | ρ(π̂,boundary) | ρ(π̂,class_size) | test frac 기각/전체 | gini | max | #cls>50% base/valid/test |",
         "|---|---|---|---|---|---|---|---|"]
    for ds, r in A.items():
        sp, cc = r["spearman"], r["class_coverage"]
        n, tot = cc["n_classes_reject_gt_50"], cc["n_classes"]
        L.append(f"| {ds} | {sp['deg']:+.3f} | {sp['boundary']:+.3f} | {sp['class_size']:+.3f} | {r['rejected']['test_frac']:.3f}/{r['all']['test_frac']:.3f} | "
                 f"{cc['gini_reject_frac']:.3f} | {cc['max_reject_frac']:.3f} | {n['base']}/{tot['base']}, {n['valid']}/{tot['valid']}, {n['test']}/{tot['test']} |")
    L += ["", "| dataset | set | deg | boundary | class_size | tail | purity@10 lr | base | valid | test |", "|---|---|---|---|---|---|---|---|---|---|"]
    for ds, r in A.items():
        for k in ("rejected", "kept"):
            d = r[k]
            L.append(f"| {ds} | {k} | {d['deg']:.2f} | {d['boundary']:.3f} | {d['class_size']:.0f} | {d['tail_frac']:.3f} | {d['purity10_lr']:.3f} | "
                     f"{d['base_frac']:.3f} | {d['valid_frac']:.3f} | {d['test_frac']:.3f} |")

    L += ["", "## 2. B 결과 (naq_unl은 '레이블된 노드 = base' 정보를 쓰는 분석용 arm, T05 §0 예외)", "",
          "| dataset | arm | seed | 1s LR | 5s LR | q_purity | anchor_uniq | anchor_ent | base/valid/test | best ep | 비고 |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds in DSS:
        for a in [BEST[ds], "lr_only"] + B_ARMS:
            for s in (0, 1):
                r = load(ds, a, s)
                if not r:
                    continue
                st, gf = r["episode_stats"], r["episode_stats"]["anchor_group_frac"]
                note = "†레이블 노드 정보 사용" if a == "naq_unl" else ("재사용(T03/T04)" if a in (BEST[ds], "lr_only") else "")
                if "w_stats" in r:
                    w = r["w_stats"]; note = f"w mean {w['mean']:.2f} sd {w['std']:.3f} [{w['min']:.2f}, {w['max']:.2f}]"
                if "hard_easy" in r:
                    h = r["hard_easy"]
                    note = (f"K={h['n_cand']}, sel loss {h['selected_loss_mean']:.3f} / cand {h['candidate_loss_mean']:.3f}, "
                            f"sel purity {h['selected_query_purity']:.3f} / cand {h['all_candidates_query_purity']:.3f}")
                L.append(f"| {ds} | {a} | {s} | {acc(r, 'test1_lr'):.2f} | {acc(r, 'test5_lr'):.2f} | {st['query_purity']:.3f} | "
                         f"{st['anchor_unique_frac']:.3f} | {st['anchor_class_entropy']:.3f} | {gf['base']:.2f}/{gf['valid']:.2f}/{gf['test']:.2f} | "
                         f"{r['best_episode']} | {note} |")

    L += ["", "## 3. 짝지은 차이 (같은 seed; |평균| < 1.0%p(eletronics) / 1.5%p(dblp) → '잡음 범위 가능')", "",
          "| dataset | 비교 | shot | s0 | s1 | 평균 | 비고 |", "|---|---|---|---|---|---|---|"]
    D = {}
    for ds in DSS:
        b = BEST[ds]
        for a, ref in (("naq_unl", b), ("lr_soft", "lr_only"), ("naq_hard", b), ("naq_easy", b), ("naq_hard", "naq_easy")):
            for k, sh in (("test1_lr", "1s"), ("test5_lr", "5s")):
                d = paired(ds, a, ref, (0, 1), k)
                if not d:
                    L.append(f"| {ds} | {a} − {ref} | {sh} | - | - | - | 미실행 |"); continue
                m = float(np.mean(list(d.values())))
                D[(ds, a, ref, sh)] = m
                L.append(f"| {ds} | {a} − {ref} | {sh} | " + " | ".join(f"{d[s]:+.2f}" if s in d else "-" for s in (0, 1)) +
                         f" | {m:+.2f} | {'잡음 범위 가능' if abs(m) < NOISE[ds] else ''} |")

    def g(ds, a, ref, sh="1s"):
        v = D.get((ds, a, ref, sh))
        return "미실행" if v is None else f"{v:+.2f}"
    E, B_ = "Amazon_eletronics", BEST
    L += ["", "## 4. 사전 등록 대조표 (계산값만; 적중 판정은 사람이 한다)", "", "| ID | 기준 | 계산값 |", "|---|---|---|",
          "| Q1 | 세 데이터셋 모두 ρ(π̂, boundary) ≤ −0.3 | " + ", ".join(f"{ds} {r['spearman']['boundary']:+.3f}" for ds, r in A.items()) + " |",
          "| Q2 | 기각 30%의 test 비율 > 전체 test 비율 (3개 중 ≥2) | " + ", ".join(f"{ds} {r['rejected']['test_frac']:.3f} vs {r['all']['test_frac']:.3f}" for ds, r in A.items()) + " |",
          "| Q3 | 기각 비율 > 50% 클래스가 데이터셋마다 ≥1 | " + ", ".join(f"{ds} {sum(r['class_coverage']['n_classes_reject_gt_50'].values())}" for ds, r in A.items()) + " |",
          f"| Q4 | eletronics naq_unl − NaQ-best ≥ +0.5 (seed 평균, 1s) | {g(E, 'naq_unl', B_[E])} |",
          "| Q5 | 두 데이터셋 모두 abs(lr_soft − lr_only) ≤ 0.7 (1s, seed 평균) | " + ", ".join(f"{ds} {g(ds, 'lr_soft', 'lr_only')}" for ds in DSS) + " |",
          f"| Q6 | eletronics naq_hard − naq_easy ≥ +1.0 (1s) | {g(E, 'naq_hard', 'naq_easy')} |",
          "| Q7 | 두 데이터셋 모두 naq_easy − NaQ-best < 0 (1s) | " + ", ".join(f"{ds} {g(ds, 'naq_easy', B_[ds])}" for ds in DSS) + " |"]
    open("results/T05_summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
