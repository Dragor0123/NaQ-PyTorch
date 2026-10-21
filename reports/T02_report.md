[T02 보고] 상태: SUCCESS (모든 산출물 완료. G1 근거 수치상 Δ_lr 조건은 0/3 통과. 판정은 사람이 결정)
소요 시간: 약 14분 (구현 10분 / 실행 약 1.5분)
수행 내용:
- cape/graph_ops.py, pool.py, scorer.py, expA.py 작성
- 3개 데이터셋 pool.npz, scores.npz, results/expA/{ds}.json, {ds}_scorer.json, {ds}_pool.json, summary.md 생성
핵심 수치:
(a) G1
| dataset | R* | p@10(R*) | p@10(lr) | Δ_lr | Δ_gbdt | Δ_lr(valid, 같은 R*) | Spearman(test) | Δ≥0.03 | ρ≥0.3 |
|---|---|---|---|---|---|---|---|---|---|
| clothing | sgc | 0.718 | 0.734 | +0.015 | +0.002 | +0.013 | 0.418 | FAIL | PASS |
| eletronics | raw | 0.631 | 0.647 | +0.016 | +0.004 | +0.025 | 0.505 | FAIL | PASS |
| dblp | sgc | 0.294 | 0.294 | +0.000 | -0.010 | +0.016 | 0.215 | FAIL | FAIL |
- G1 Δ_lr 조건: 0/3 통과 (기준 2/3). Spearman: 2/3 통과.
(c) class-sim@10 / purity@10 (all 그룹) — NaQ 보고값 재현됨
| | raw | ppr | sgc | lr | NaQ raw/ppr |
|---|---|---|---|---|---|
| clothing | .816/.597 | .790/.568 | .841/.637 | .851/.653 | .816/.788 |
| eletronics | .819/.645 | .527/.300 | .730/.517 | .835/.666 | .819/.527 |
| dblp | .735/.157 | .830/.293 | .841/.300 | .836/.301 | .735/.830 |
(test 그룹 수치는 results/expA/summary.md (c))
(d) 보정 (lr; gbdt는 summary.md)
| | AUC base/test | ECE base/test | gap base/valid/test |
|---|---|---|---|
| clothing | .730/.722 | .180/.045 | +.187/+.224/+.079 |
| eletronics | .805/.842 | .100/.113 | +.081/+.133/+.111 |
| dblp | .721/.701 | .088/.089 | +.188/+.147/+.181 |
(e) 선택적 purity@10_lr (test): f=1.0 / 0.7 π̂ (무작위) / 0.5 π̂ (무작위)
- clothing .734 / .804 (.730) / .845 (.730)
- eletronics .647 / .730 (.648) / .785 (.649)
- dblp .294 / .320 (.294) / .344 (.299)
(f) LR 표준화 계수 [cos_raw, cos_sgc, log_ppr, adj, mknn_raw, neg_euclid_raw]
- clothing +.418 +.618 +.151 -.014 -.027 +.138
- eletronics +.955 +.369 +.018 -.053 -.011 +.357
- dblp +.112 +.149 +1.125 -.269 -.019 -.001
(g) tail(test 하위 20%) purity@10 raw/sgc/ppr/lr: clothing .690/.708/.605/.753, eletronics .462/.393/.220/.486, dblp .137/.282/.276/.296
풀 통계·검사:
- 풀 크기 평균/중앙값/최소: clothing 105.6/105/57, eletronics 119.4/121/57, dblp 113.6/114/64
- top-50 겹침 raw-sgc / raw-ppr / sgc-ppr: clothing .32/.22/.54, eletronics .33/.09/.27, dblp .16/.12/.56
- 검사1 brute top-10 일치: raw 100/100 (3개 ds), sgc 100/100 (clothing은 97 + 동률 3). 검사2 PPR K=100 겹침 .976/.984/1.000.
- 검사3 대칭 상대오차 중앙값 0.0, 최대 9.5e-7 (float32 반올림 수준). 1,000쌍. 모두 PASS.
- 단계 시간 (풀 / scorer / 지표): clothing 7s/5s/1s, eletronics 16s/9s/4s, dblp 16s/8s/8s. 피크 GPU 3.4/5.5/4.7GB. LR은 8–10 iter에 수렴 (경고 없음).
계획 대비 차이·임의 결정:
- 검사1 동률 판정: 정렬된 top-10 값이 atol 1e-5 안에서 같으면 동률로 본다.
- 검사3: 저장된 log_ppr에서 exp를 취한 값(Π+1e-6)으로 상대오차를 쟀다. Π>1e-4 쌍만 따로 본 중앙값도 0.0이다.
- Δ_lr(valid)는 test에서 정한 R*를 그대로 썼다. valid 자체의 best로 계산한 값은 JSON에 있다 (+.013/+.025/+.004).
- 선택적 순도의 k는 round(f·n), 무작위 대조는 seed 0 순열이다. ECE bin은 [0,1] 10등분이다.
- tail 클래스는 test 클래스 크기 오름차순으로 정렬해 ceil(20%)개를 택했다. 동률은 클래스 id 순이다.
- CLAUDE.md(사용자가 §6 갱신)의 변경분을 이번 커밋에 포함했다.
문제·미해결 / 이상치 (사실 위주):
- G1 Δ가 작다: lr이 R*보다 +0.0–1.6%p 높은 데 그친다. dblp에서는 lr ≈ sgc ≈ ppr이고 gbdt는 R*보다 낮다.
- π̂ 과대추정: test 보정 격차가 +.08~+.18이다. 원인 후보(확인한 사실): 학습 쌍은 v∈base로 제한되어 양성 비율이 .618/.521/.283이다. 반면 실제 전체 풀 쌍의 양성 비율은 base 앵커 .403/.364/.181, test 앵커 .567/.330/.178이다. 이 차이에서 생기는 선택 편향이 intercept를 높인 것으로 보인다. 그래서 학습 분포 안(in-sample)인 base 그룹의 ECE가 clothing에서 오히려 크다(.180).
- adj 계수가 세 데이터셋 모두 음수다 (dblp -.269). 다른 단서와 공선성이 있는 상태의 조건부 효과로 보인다. mknn_raw 계수도 음수지만 크기는 작다.
- 그럼에도 π̂는 순위 정보로서 유효하다. 선택적 순도가 f=0.5에서 +11/+14/+5%p 오르고, 무작위 대조는 f=1.0과 거의 같다.
- dblp는 raw 순도가 .157로 매우 낮다 (class-sim은 .735로 높다). NaQ class-sim 수치는 4개 셀 모두 소수 셋째 자리까지 재현됐다 (clothing ppr만 .790 vs .788).
- 1.5%p 미만의 차이는 잡음일 수 있다.
변경·생성 파일:
- cape/graph_ops.py, cape/pool.py, cape/scorer.py, cape/expA.py
- results/expA/{ds}.json, {ds}_scorer.json, {ds}_pool.json (각 3개), results/expA/summary.md
- reports/T02_report.md, CLAUDE.md (사용자 수정분), instructions/T02_pool_scorer_expA.md
커밋: 5b318f6 (branch cape-validation)
