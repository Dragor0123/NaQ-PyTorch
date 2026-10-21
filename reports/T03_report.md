[T03 보고] 상태: SUCCESS (12 run 전부 완료. 게이트·P1–P16 판정은 사람이 결정)
소요 시간: 약 40분 (구현 12분 / 스모크 1분 / 본 실행 26분 / 집계 1분)
수행 내용:
- instructions/T03_preregistration.md를 수정 없이 먼저 커밋했다 (946b599 "T03: preregistration").
- cape/generators.py, cape/train_proto.py, cape/run_expB.sh, cape/summarize.py 작성
- 사용자 추가 지시에 따라 oracle_base를 추가해 arm 6개 × 2 데이터셋 = 12 run 실행.
  - oracle_base는 G2·G3에서 제외하고, 보조 비교로만 적었다.
- 스모크 (eletronics cape_lite, 400 ep, 단독 실행)
  - step 27.3ms, 생성기 0.01s/400 ep, test 평가(LR 1,000회 + proto) 1.1s
  - 추정: run당 114s(eletronics) / 127s(dblp). 두 프로세스가 GPU를 완전히 직렬로 쓴다고 보면 6×241s ≈ 24분.
  - 24분 ≤ 30분이므로 4,000 에피소드로 진행했다.
- 실측: 병렬 시 step 60–62ms, run당 약 256s, 총 26분 (07:40–08:06). 2배 초과 run 없음, NaN/OOM 없음.
핵심 수치:
- (a) Exp B 표 (1s LR ± 95% CI)
| ds | arm | q_pur | way_coll | q_ovl | anc_uniq | anc_ent | 1s LR | 5s LR | 1s proto | best ep |
|---|---|---|---|---|---|---|---|---|---|---|
| elec | naq_feat | .646 | .031 | .001 | .798 | .971 | 74.58±1.12 | 87.44 | 70.28 | 4000 |
| elec | naq_diff | .302 | .031 | .186 | .798 | .971 | 68.61±1.18 | 84.83 | 63.76 | 3500 |
| elec | lr_only | .666 | .031 | .001 | .798 | .971 | 74.11±1.15 | 87.26 | 69.45 | 3750 |
| elec | cape_lite | .728 | .028 | .000 | .726 | .953 | 73.26±1.19 | 86.72 | 68.20 | 3250 |
| elec | oracle | 1.00 | .000 | .000 | .796 | .972 | 78.61±1.11 | 89.69 | 74.37 | 4000 |
| elec | oracle_base | 1.00 | .000 | .000 | .679 | .851 | 67.58±1.20 | 84.31 | 62.77 | 2000 |
| dblp | naq_feat | .156 | .041 | .004 | .792 | .967 | 71.72±1.19 | 81.35 | 68.59 | 1750 |
| dblp | naq_diff | .292 | .041 | .001 | .792 | .967 | 73.28±1.20 | 82.35 | 69.28 | 750 |
| dblp | lr_only | .298 | .041 | .001 | .792 | .967 | 73.77±1.17 | 82.40 | 69.53 | 2000 |
| dblp | cape_lite | .339 | .037 | .000 | .718 | .971 | 72.10±1.18 | 82.66 | 67.36 | 4000 |
| dblp | oracle | 1.00 | .000 | .000 | .787 | .967 | 78.00±1.12 | 84.44 | 75.59 | 2750 |
| dblp | oracle_base | 1.00 | .000 | .000 | .668 | .862 | 74.25±1.16 | 82.33 | 70.67 | 4000 |
- 앵커 그룹 비율(base/valid/test): oracle_base 외 arm은 .56–.59 / .17–.19 / .23–.24, oracle_base는 1/0/0.
- (b) 게이트 근거
  - G1 (T02): Δ_lr 조건 0/3 통과, Spearman 2/3 통과.
  - G2: oracle − NaQ-best = eletronics +4.03, dblp +4.73, 평균 +4.38 (기준 ≥ +2.0 → 충족). NaQ-best: eletronics feat, dblp diff.
  - G3: cape_lite − NaQ-best = −1.32 / −1.18, 평균 −1.25.
    - 평균 조건: 기준값 (oracle − NaQ-best)/3 = +1.46 → 미충족
    - 데이터셋별 ≥ −0.5 조건: 둘 다 미충족
  - 보조 (게이트 아님, 1.5%p 미만은 잡음 범위)
| | lr_only−best | cape_lite−lr_only | oracle_base−best | oracle−oracle_base |
|---|---|---|---|---|
| elec | −0.47 | −0.85 | −6.99 | +11.02 |
| dblp | +0.49 | −1.67 | +0.98 | +3.75 |
| 평균 | +0.01 | −1.26 | −3.01 | +7.39 |
- (c) 재현 (1s / 5s, 괄호는 논문값): ±5%p 초과 없음, Feat/Diff 우열 방향 일치.
  - elec feat 74.58 (76.46) / 87.44 (88.72)
  - elec diff 68.61 (68.56) / 84.83 (84.88)
  - dblp feat 71.72 (71.38) / 81.35 (82.34)
  - dblp diff 73.28 (73.89) / 82.35 (82.24)
  - feat−diff 격차: elec +5.96 (논문 +7.90), dblp −1.56 (논문 −2.51)
- (e) 과소학습: best가 3,200 초과인 run 7/12 (oracle_base 제외 6/10) → "학습 부족 가능성" 표시.
  - 모두 eletronics 5개(oracle_base 제외)와 dblp cape_lite, dblp oracle_base다. dblp의 나머지 4개는 750–2,750에서 best.
- 에피소드 통계에서 눈에 띄는 점 (사실)
  - cape_lite: 순도 상승 (.646→.728, .292→.339). anchor_unique_frac 하락 (.798→.726, .792→.718). class entropy 차이는 작다 (.971→.953, .967→.971). 앵커 그룹 비율은 거의 같다.
  - oracle_base: entropy .851/.862, unique .68 (base 클래스 노드만 사용).
  - eletronics naq_diff: query_overlap .186 (다른 arm ≤ .004). 평균 degree 2.06인 희소 그래프에서 PPR 이웃이 겹치는 것으로 보인다.
  - 순도와 정확도: 같은 데이터셋 안에서 순도가 높아도 정확도가 오르지 않는 경우가 있다. cape_lite는 lr_only보다 순도가 높지만 1s 정확도가 낮다.
  - 1-shot proto는 LR보다 2.4–5.1%p 낮다 (모든 run).
계획 대비 차이·임의 결정:
- 사용자 추가 지시: oracle_base 추가 (oracle 뒤), 실행 조건 기준 시간 30분 적용.
- oracle 앵커 샘플링: 노드를 균등하게 뽑고, 이미 나온 레이블이면 다시 뽑는 방식으로 레이블 5종을 만든다. oracle_base도 같은 방식이고, base 노드에서만 뽑는다.
- oracle query 부족분 보충: 같은 레이블 노드에서 앵커와 기존 query를 빼고 비복원으로 뽑았다.
- query_overlap: 같은 에피소드 안에서 다른 way의 query로도 나오는 query 슬롯의 비율. 다른 way의 앵커와 겹치는 경우는 세지 않았다.
- naq_feat / naq_diff / lr_only는 같은 seed 0 rng를 써서 앵커 열이 동일하다 (짝지은 비교가 된다).
- valid proto 정확도: prototype은 support 평균, 거리는 Euclidean. 동률이면 앞선 checkpoint(엄격한 > 비교).
- LR probing: 임베딩을 정규화하지 않았다 (NaQ meta.py fine_tuning과 같음). joblib n_jobs=8.
- matplotlib(그림), joblib(LR 병렬)은 지시서가 요구·허용한 범위에서만 썼다.
- peak_gpu_GB가 약 0.48GB로 기록됐다. T01 측정값(1.2GB)보다 낮으며 원인은 확인하지 않았다. 7GB 목표 안이다.
- 사전 등록 예측 P1–P16은 판정하지 않았다 (사용자 지시).
문제·미해결:
- seed 1개라서 cape_lite/lr_only/NaQ-best 사이 차이(≤1.7%p)는 대부분 잡음 범위다.
- 과소학습 신호가 있다 (eletronics에서 best가 대부분 마지막 구간). 에피소드 수 변경은 지시서 범위 밖이므로 하지 않았다.
변경·생성 파일:
- cape/generators.py, cape/train_proto.py, cape/run_expB.sh, cape/summarize.py
- results/expB/{ds}_{arm}_s0.json ×12, results/expB/purity_vs_acc.png, results/summary.md
- reports/T03_report.md, instructions/T03_expB_downstream.md (T03_preregistration.md는 946b599에서 커밋)
커밋: 3d5b873 (branch cape-validation)
