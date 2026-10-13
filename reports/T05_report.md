[T05 보고] 상태: SUCCESS (A 완료, B 12 run 전부 완료, 축소 없음. Q1–Q7은 계산값만 적었고 판정하지 않았다)
소요 시간: 약 31분 (12:18:48 시작, 12:49 경 종료; 구현 약 1분 / A 1초 미만 / 스모크 약 1분 / B 26.5분 / 집계·보고 약 2분)
- 실행 시각은 `date` 기준이다. 구현 시간이 짧게 찍힌 것은 시스템 시계 기준 그대로 적은 것이다.
수행 내용:
- 지시서를 수정 없이 먼저 커밋했다 (b40fb4e "T05: instruction and preregistration").
- 새 파일: cape/diag_pihat.py, cape/run_T05.sh, cape/summarize_t05.py
- generators.py 확장
  - `naq_best` (내부용): NaQ-best ranker. eletronics = raw, dblp = ppr
  - `naq_unl`: 앵커를 비-base 노드(valid ∪ test 클래스)에서만 비복원 균등으로 뽑는다. §0 예외 적용.
  - `lr_soft`: 에피소드는 lr_only와 같다.
- train_proto.py 확장
  - lr_soft: query별 NLL × w_q
  - naq_hard / naq_easy: step마다 후보 K개를 한 번의 forward로 평가하고, argmax/argmin 손실 하나만 backward한다.
  - 에피소드 통계는 실제로 학습에 쓰인 에피소드로 계산한다. hard/easy와 soft의 추가 통계도 기록한다.
  - 기존 arm의 동작은 바뀌지 않았다 (K=1이고 가중치가 없으면 이전 코드 경로와 같다).
- 스모크 (naq_hard, eletronics, 300 ep, 단독): step 28.9ms. T03 27ms의 2배 이하라서 후보 수 K=4를 유지했다.
- B 추정: T03/T04 병렬 실측(run 쌍당 약 256s) × 6 + hard/easy 여유 ≈ 26.5분 ≤ 28분 → 축소 없음. 실측 26.5분.
- NaN·OOM·재시도 없음.
핵심 수치:
- A: Spearman ρ(π̂, 속성) — 전체 노드
| ds | deg | boundary | class_size |
|---|---|---|---|
| clothing | +0.382 | −0.126 | +0.011 |
| eletronics | +0.091 | −0.054 | +0.202 |
| dblp | −0.229 | −0.325 | −0.288 |
- A: 기각 30% vs 유지 70%
| ds | set | deg | boundary | cls_size | tail | pur@10 | base/valid/test |
|---|---|---|---|---|---|---|---|
| clothing | 기각 | 2.60 | .458 | 487 | .076 | .483 | .469/.193/.337 |
| clothing | 유지 | 9.40 | .345 | 509 | .077 | .725 | .413/.195/.392 |
| elec | 기각 | 1.79 | .644 | 300 | .128 | .514 | .520/.222/.258 |
| elec | 유지 | 2.17 | .608 | 363 | .082 | .731 | .583/.182/.234 |
| dblp | 기각 | 17.96 | .809 | 487 | .059 | .204 | .539/.206/.255 |
| dblp | 유지 | 12.55 | .668 | 373 | .099 | .343 | .592/.175/.233 |
- A: 클래스 커버리지 (클래스별 기각 비율)
  - 지니: .295 / .321 / .493
  - 최대: .658 / .956 / .816
  - 기각 50% 초과 클래스 수 (base/valid/test): clothing 6/4/0, elec 16/9/11, dblp 14/7/5
- B: 짝지은 차이 (1s LR, seed 0 / 1 / 평균). 5s는 results/T05_summary.md에 있다.
| 비교 | eletronics | dblp |
|---|---|---|
| naq_unl† − NaQ-best | +3.15 / +2.50 / **+2.83** (5s +1.32) | +1.33 / +0.35 / +0.84 (5s +0.68) |
| lr_soft − lr_only | +0.03 / −0.02 / +0.01 | −0.03 / +0.01 / −0.01 |
| naq_hard − NaQ-best (s0) | +0.28 | −0.46 |
| naq_easy − NaQ-best (s0) | +0.24 | +0.91 |
| naq_hard − naq_easy (s0) | +0.05 | −1.37 |
  - †naq_unl은 "레이블된 노드 = base" 정보를 쓰는 분석용 arm이다.
  - 잡음 기준(eletronics 1.0, dblp 1.5%p)을 넘는 항목은 eletronics naq_unl뿐이다 (1s, 5s 모두).
- 사전 등록 대조표 (계산값만)
  - Q1 ρ(π̂, boundary): −0.126 / −0.054 / −0.325
  - Q2 기각 30% test 비율 vs 전체 test 비율: clothing .337 vs .376, elec .258 vs .241, dblp .255 vs .240
  - Q3 기각 50% 초과 클래스 수: 10 / 36 / 26
  - Q4 eletronics naq_unl − NaQ-best: +2.83
  - Q5 lr_soft − lr_only: +0.01 / −0.01
  - Q6 eletronics naq_hard − naq_easy: +0.05
  - Q7 naq_easy − NaQ-best: +0.24 / +0.91
- 사실 관찰 (짧게)
  - naq_unl 앵커는 entropy .82/.79, unique .61/.59, test 비율 .55–.57이다 (NaQ는 .97, .80, .24). query 순도는 NaQ와 비슷하거나 약간 낮다 (.611 vs .646, .309 vs .292).
  - lr_soft의 w 분포는 sd .19/.15, 범위 약 [0.02, 1.9]다. 정확도는 lr_only와 소수 첫째 자리까지 거의 같다.
  - naq_hard / naq_easy의 선택 손실과 후보 평균 손실
    - eletronics: hard 0.805 (후보 평균 0.498), easy 0.146 (후보 평균 0.998)
    - dblp: hard 0.284 (0.154), easy 0.049 (0.371)
    - easy로 학습한 모델에서는 후보 평균 손실이 더 높다.
  - naq_hard / naq_easy의 선택된 에피소드 순도: hard .604 / .266, easy .685 / .320 (후보 전체 .646 / .292). 난이도 선택이 순도와도 연동된다.
  - 기각 집합은 데이터셋마다 다르다.
    - clothing: 저 degree, 고 boundary
    - eletronics: 작은 클래스·tail 비율 높음
    - dblp: 고 degree, 고 boundary
계획 대비 차이·임의 결정:
- naq_unl의 "레이블 없는 노드" = split의 valid ∪ test 클래스 노드 (= 비-base). §0 예외로 적용했다.
- lr_soft의 s_lr(anchor, q)는 풀 안의 값을 쓴다. query가 lr top-10이므로 항상 풀 안에 있다.
- lr_soft 정규화는 에피소드별 50개 query 평균으로 했다. 손실은 가중 NLL의 평균이다.
- naq_hard / naq_easy의 후보 생성은 step마다 연속 4개 (같은 seed rng)다.
  - 손실 비교는 학습용 forward(train 모드, dropout 없음)를 공유한다. 선택은 detach 없이 argmax/argmin 인덱스로 했다.
  - valid 곡선 로그의 loss는 선택된 손실이다.
- tail 클래스: 각 그룹(base/valid/test) 안에서 크기 오름차순 ceil(20%)
- boundary는 이웃 중 레이블이 다른 비율이다. degree 0 노드는 없다.
- 십분위는 π̂ 순위 기준 10등분이고, 기각 = π̂ < 30번째 백분위수다.
- run_T05.sh: run이 실패하면 한 번 재시도한다 (§6). 실제 재시도는 없었다.
- 스모크·버그 확인용 짧은 run 결과는 logs/T05_smoke/에만 저장했다.
- (무인 실행) 질문 없이 지시서 규칙대로만 판단했다. 판단 내용은 위 "B 추정" 항목과 같다.
문제·미해결:
- naq_hard / naq_easy는 seed 1개다. 모든 차이가 잡음 기준 미만이다.
- naq_unl의 이득에는 "앵커 = test/valid 클래스 노드" 노출 효과와 앵커 다양성 감소가 섞여 있다. 이 실험만으로 분리되지 않는다.
- results/T03_summary_message.md는 이번에도 커밋하지 않았다.
변경·생성 파일:
- cape/generators.py, cape/train_proto.py (확장)
- cape/diag_pihat.py, cape/run_T05.sh, cape/summarize_t05.py
- results/T05A_diag.md, results/T05A_diag.json, results/T05_summary.md
- results/expB/{ds}_{naq_unl,lr_soft}_s{0,1}.json ×8, {ds}_{naq_hard,naq_easy}_s0.json ×4
- reports/T05_report.md
커밋: e04f26b (branch cape-validation)
