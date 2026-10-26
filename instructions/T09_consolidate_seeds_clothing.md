# T09 — 통합: 5 seed 확장(Electronics·dblp) + 세 번째 데이터셋(Amazon_clothing) (무인 실행 가능)

- 선행 문서: `CLAUDE.md`(2026-10-26 개정, 특히 §1.2), `reports/T08_report.md`, `results/T08_summary.md`
- 무인 실행 전제. 질문하지 말고 규칙에 따라 판단하고 보고서에 기록한다.
- 시간 예산: **130분** (git·점검 10 / 구현 10 / 실행 100 / 집계·보고 10). 시간 규칙은 §4.
- **새 방법은 없다.** 기존 arm 코드(`cape/generators.py`의 `naq_feat`·`naq_diff`·`naq_best`·`naq_unl`·`nov2_t1_r0.5`·`naq_mix_r0.5`·`mix_alt_r0.5`·`sup_only_r0.5`)를 그대로 쓴다. 생성기 코드는 수정하지 않는다.
- 목적
  1. 방법 후보 `mix_alt_r0.5`와 통제 arm의 seed를 5개(0–4)로 늘려 T08의 결론(G8-a: Electronics novel 앵커 몫, dblp 지도 혼합 몫)이 유지되는지 본다.
  2. Amazon_clothing에서 같은 arm을 처음 학습해 세 번째 데이터셋 결과를 얻는다.
  3. 논문 표에 쓸 1-shot·5-shot 수치를 한 표로 만든다.

## 0. git과 브랜치 (가장 먼저, 이 순서대로)

현재 작업 트리는 `token_proto` 브랜치(HEAD `670afac`)에 있다. 아래를 순서대로 하고, 각 단계의 명령 출력 요약을 보고서 "수행 내용"에 적는다.

1. `git status --short` 확인. 추적 파일 변경이 있으면 **중단하고 보고**한다. untracked는 `results/T03_summary_message.md`, `results/T07_message.txt` 외에 있으면 목록을 적고 계속한다(커밋하지 않는다).
2. `git push -u origin token_proto` — token_proto 트랙(TP01–TP02)은 종결이며, 이 push 이후 이 브랜치에는 손대지 않는다. push 실패 시 사유를 적고 계속한다.
3. `git checkout cape-validation` → `git log --oneline -1`이 `8f92e39`인지, `git rev-parse origin/cape-validation`과 같은지 확인. 다르면 **중단하고 보고**한다.
4. 사용자가 저장소 밖 `~/wkspace/T09_drop/`에 둔 두 파일을 복사한다.
   - `~/wkspace/T09_drop/CLAUDE.md` → 저장소 루트 `CLAUDE.md`(덮어쓰기)
   - `~/wkspace/T09_drop/T09_consolidate_seeds_clothing.md` → `instructions/T09_consolidate_seeds_clothing.md`
   - 두 파일이 없으면 **중단하고 보고**한다. 복사한 뒤 **새 CLAUDE.md를 다시 읽는다.**
5. 내용 수정 없이 커밋: `T09: instruction, preregistration, CLAUDE.md update`. 바로 `git push origin cape-validation`.

## 1. 사전 점검 (실행 전)

- 디스크: `df -h .` — 여유 10GB 미만이면 이번 큐에서 `--save_emb`를 끄고(§3) 사유를 적는다.
- GPU: `nvidia-smi` — 다른 사용자 프로세스가 있으면 기록. 메모리가 2 프로세스(각 7GB)를 못 받으면 작업자 1개로 돌리고 §4 시간 규칙을 다시 계산한다.
- Amazon_clothing 캐시: `cache/Amazon_clothing/`에 `tasks.npz`, `pool.npz`, `scores.npz`, `labelmask_r0.5.npz`, `novelty2_r0.5.npz`가 있는지 확인한다.
  - 없는 파일은 기존 코드의 생성 경로(`cape.data`, `cape.pool`, `cape.scorer`, `cape.labelmask`, `cape.novelty2.compute_estimators`)로 만든다. 새 정의를 쓰지 않는다.
  - `novelty2_r0.5.npz`를 새로 만들었다면 `pu_feat`의 test-vs-hidden-base AUROC를 다시 계산해 T07 값(.909)과 ±0.01 안인지 적는다(분석용).
- 소스 검사: T08과 같은 검사기로 생성기·추정기 파일에 split 정보가 없음을 확인하고 `results/T09_source_check.json`에 저장한다.

## 2. arm (모두 기존 정의, r = 0.5)

| 기호 | arm 이름 (`--arm`) | 내용 |
|---|---|---|
| NaQ-best | Electronics `naq_feat`, dblp `naq_diff`, Clothing `naq_feat` | 레이블 없음 |
| NaQ-alt | Clothing `naq_diff` | 참고용(NaQ-Feat/Diff 우열 확인) |
| nov | `nov2_t1_r0.5` | 앵커 ∝ p_novel(pu_feat), 지도 없음 |
| naq_mix ‡ | `naq_mix_r0.5` | NaQ-best 70% + L 지도 30% |
| **mix_alt ‡** | `mix_alt_r0.5` | nov 70% + L 지도 30% — **방법 후보** |
| sup_only ‡ | `sup_only_r0.5` | L 지도 100% |
| naq_unl † | `naq_unl` | 앵커 = 비-base 노드 균등. split 이용, **분석용 상한** |

‡ = L의 레이블을 쓰는 arm. † = 분석용(방법 아님, CLAUDE.md §3).

## 3. 실행 목록 (우선순위 순, 47 run)

`cape/run_T09.sh`를 `cape/run_T08.sh`와 같은 방식(작업자 2개, flock 큐, 실패 시 1회 재시도, DEADLINE)으로 만든다. 학습 명령은 T08과 같다: `python -m cape.train_proto --dataset <ds> --arm <arm> --episodes 4000 --eval_every 250 --seed <s> --save_emb`.
**이미 결과 JSON(`results/expB/{ds}_{arm}_s{seed}.json`)이 있는 (ds, arm, seed)는 큐에 넣지 않는다**(재실행 금지).

| 우선순위 | run | 수 |
|---|---|---|
| P1 | Clothing × {`naq_feat`, `nov2_t1_r0.5`, `mix_alt_r0.5`, `naq_mix_r0.5`} × seed {0, 1, 2} | 12 |
| P2 | {Electronics, dblp} × {`mix_alt_r0.5`, `naq_mix_r0.5`} × seed {2, 3, 4} | 12 |
| P3 | {Electronics, dblp} × {NaQ-best, `nov2_t1_r0.5`} × seed {3, 4} | 8 |
| P4 | Clothing × {`naq_unl`, `sup_only_r0.5`, `naq_diff`} × seed {0, 1, 2} | 9 |
| P5 | {Electronics, dblp} × `sup_only_r0.5` × seed {2, 3, 4} | 6 |

- 큐 파일의 줄 순서 = 위 우선순위, 각 우선순위 안에서는 seed 바깥·arm 안쪽.
- 첫 run의 출력 JSON이 T08 JSON과 같은 필드를 갖는지 확인한다(특히 1s/5s LR, valid 곡선, anchor_novel_frac).

## 4. 시간 규칙

1. 첫 2 run(같은 시각 시작)의 wall time으로 전체 큐 시간을 추정한다. Clothing과 나머지 데이터셋의 시간이 다를 수 있으므로, P2의 첫 run이 끝나면 한 번 더 추정한다.
2. 추정 총 실행 시간이 100분을 넘으면 **뒤 우선순위부터** 큐에서 지운다(P5 → P4 → P3). 지운 run은 보고서에 적는다.
3. `DEADLINE` = 큐 시작 + 105분. 그 뒤로는 새 run을 시작하지 않는다(진행 중인 run은 끝까지).
4. 총 경과 140분을 넘으면 진행 중 run만 마치고 집계한다.

## 5. 집계 — `cape/summarize_t09.py` → `results/T09_summary.md`, `results/T09_summary.json`

T07·T08과 같은 집계 방식. **모든 짝지은 차이는 두 arm의 공통 seed에서** 계산하고, 사용한 seed 목록을 함께 적는다. 레이블·split은 분석용으로만 읽는다.

1. **주 표** (dataset × arm): 1s LR, 5s LR의 seed 평균 ± sd, seed 수, q_purity, anchor_novel_frac, best ep 평균. 기존 run(T03–T08)의 JSON을 포함한다.
2. **짝지은 차이** (1s, 5s; seed별 · 평균 · sd · 음수/양수 개수)
   - `mix_alt − NaQ-best` (주 결과)
   - `mix_alt − naq_mix` (novel 앵커의 몫)
   - `naq_mix − NaQ-best` (지도 혼합의 몫)
   - `nov − NaQ-best`, `mix_alt − nov`
   - `mix_alt − sup_only`, `sup_only − NaQ-best`
   - `naq_unl† − NaQ-best` (상한), 회수율 R = (`nov − NaQ-best`) / (`naq_unl† − NaQ-best`)와 R_mix = (`mix_alt − NaQ-best`) / (`naq_unl† − NaQ-best`)
   - Clothing: `naq_diff − naq_feat`
3. **분해 표** (1s): T08 §4와 같은 두 경로(NaQ → naq_mix → mix_alt / NaQ → nov → mix_alt). 공통 seed만 쓰면 두 경로 합이 같아야 한다.
4. **용량-반응 표** (1s, 데이터셋별): arm마다 "pseudo-episode 앵커의 novel 비율"과 `arm − NaQ-best`.
   - pseudo 앵커 novel 비율 = 지도 에피소드가 없는 arm은 그 arm의 `anchor_novel_frac`. 지도 혼합 arm은 pseudo 부분이 같은 무혼합 arm의 값을 쓴다(`naq_mix` → NaQ-best, `mix_alt` → `nov2_t1`). 표에 이 규칙을 적는다.
   - 포함 arm: NaQ-best, `nov2c_t1_r0.5`(T08, Electronics·dblp만), `nov2_t1_r0.5`, `mix_alt_r0.5`, `naq_unl`†. Spearman ρ(novel 비율, Δ)를 데이터셋별로 적는다(점이 5개 미만이면 값만 적고 ρ는 비운다).
5. **G9 계산값** (판정은 사람)
   - G9-a: 세 데이터셋 모두 `mix_alt − NaQ-best` (1s) 평균 ≥ +1.0, 양수 seed가 Electronics·dblp 5개 중 4개 이상, Clothing 3개 중 3개
   - G9-b: Electronics `mix_alt − naq_mix` (1s) 평균 ≥ +0.7, 양수 seed 5개 중 4개 이상
   - G9-c: 세 데이터셋 모두 `mix_alt − sup_only` (1s) > 0
6. **사전 등록 대조표** (§6)
7. 실행 범위·소요 시간·제외된 run.

## 6. 사전 등록 (실행 전 커밋, 판정은 사람)

| ID | 예측 | 기준 | 근거 | 확신도 |
|---|---|---|---|---|
| V1 | novel 앵커 몫이 5 seed에서도 유지된다 | Electronics `mix_alt − naq_mix` ≥ +0.7, 양수 ≥ 4/5 | T08 2 seed +1.56, sd 0.14 | 0.80 |
| V2 | dblp의 이득은 지도 혼합에서 온다 | dblp `naq_mix − NaQ-best` ≥ +1.5, 그리고 \|`mix_alt − naq_mix`\| ≤ 1.0 | T08 +2.99 / −0.35 | 0.75 |
| V3 | 주 결과가 5 seed에서 유지된다 | `mix_alt − NaQ-best` Electronics ≥ +1.3, dblp ≥ +2.0 | T07·T08 +1.67 / +2.65 | 0.70 |
| V4 | Clothing은 Electronics 레짐이다: novel 앵커만으로 이득 | Clothing `nov2_t1 − naq_feat` ≥ +1.0, 3/3 양수 | Clothing pu_feat AUROC .909(세 데이터셋 중 최고), NaQ query 순도가 Electronics 쪽 | 0.55 |
| V5 | Clothing에서도 novel 앵커의 독립 몫이 있다 | Clothing `mix_alt − naq_mix` ≥ +0.7 | V4와 같은 근거 | 0.55 |
| V6 | Clothing 상한이 크다 | Clothing `naq_unl† − naq_feat` ≥ +1.5 | Electronics +2.59 | 0.50 |
| V7 | Clothing에서 지도 단독은 크게 진다 | Clothing `sup_only − naq_feat` ≤ −2.0 | Electronics −5.73(같은 레짐 가정) | 0.55 |
| V8 | 용량-반응: Electronics에서 pseudo 앵커 novel 비율과 이득이 함께 오른다 | Electronics Spearman ρ ≥ 0.8 (§5-4의 5개 점) | T08 표 | 0.70 |
| V9 | Clothing NaQ-best 선택이 맞다 | Clothing `naq_feat − naq_diff` (1s) > 0 | NaQ 논문의 Amazon 데이터셋 경향, `NAQ_BEST_COL` | 0.65 |

해석 규칙 (사전 명시)
- G9-a·b 통과 → `mix_alt`를 논문 방법으로 동결. 다음은 T10(앵커 규칙 정제 1회: 순위 상위 q% 균등, 혼합 비율의 valid 선택)과 비교 기준선.
- G9-b 실패(Electronics novel 몫이 5 seed에서 사라짐) → novel 앵커링의 독립 기여 주장을 내리고 방향을 재검토.
- V4·V5 실패이면서 V6 통과(상한은 큰데 회수 못함) → 추정기·앵커 규칙의 문제로 보고 T10의 주 대상으로 삼는다. V6도 실패하면 Clothing은 dblp 레짐으로 기록한다.
- V9 실패(Clothing에서 `naq_diff`가 더 좋음) → 이번 Clothing 결과는 `naq_feat` 기준 그대로 보고하고, NaQ-best 재정의는 사람이 정한다(Agent는 정의를 바꾸지 않는다).

## 7. 범위 밖

- 생성기·추정기·학습기 코드 수정, 새 arm, 하이퍼파라미터(에피소드 수, 혼합 비율, τ) 변경
- r ≠ 0.5, TEG, VISION, 보정 추정기(`nov2c_*`)의 추가 run
- NMI 재계산 (T10 이후 필요하면 지시)

## 8. 완료 기준과 보고

- [ ] §0 git 단계 완료(token_proto push, cape-validation 전환, 지시서·CLAUDE.md 커밋 `T09: instruction, preregistration, CLAUDE.md update`, push)
- [ ] `cape/run_T09.sh`, `cape/summarize_t09.py`, `results/T09_source_check.json`
- [ ] `results/expB/` 새 JSON, `results/T09_summary.{md,json}`
- [ ] `reports/T09_report.md`, 커밋 `T09: seed expansion and Amazon_clothing`, `git push origin cape-validation`
- 보고 형식 CLAUDE.md §12. 핵심 수치에 주 표(1s·5s), 짝지은 차이 중 `mix_alt − NaQ-best`·`mix_alt − naq_mix`·`naq_mix − NaQ-best`·`naq_unl† − NaQ-best`, 분해 표, 용량-반응 표, G9, 사전 등록 대조, 제외된 run, 소요 시간.
