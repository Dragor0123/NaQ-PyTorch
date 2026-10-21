[T06 보고] 상태: SUCCESS (P1·P2·P3a 완료, P3b는 시간 규칙에 따라 제외. G6·R1–R8은 계산값만 적었다)
소요 시간: 약 48분 (15:01:32 시작 – 15:49 종료; 구현·점검·A 약 2분 / B 42.5분 / 집계·보고 약 3분)
수행 내용:
- 지시서와 갱신된 CLAUDE.md를 수정 없이 먼저 커밋했다 (1bac211).
- 새 파일: cape/labelmask.py, cape/novelty.py, cape/diag_novelty.py, cape/run_T06.sh, cape/summarize_t06.py
- generators.py 확장
  - make_unl_uni / make_nov / make_scarce_gen 추가. 인자는 top10(풀에서 유도), L_idx, y_L, novelty뿐이다.
  - episode_stats에 anchor_novel_frac 추가 (분석용)
- §1.4 사전 점검: 모두 통과 (results/T06_precheck.json)
  1. r=0.5 마스크에서 클래스별 |L_c|/n_c가 0.5 ± 1/n_c를 벗어난 클래스: 0개 (두 데이터셋)
  2. r=1.0에서 U == valid ∪ test 노드: True (두 데이터셋)
  3. 생성기 소스 점검: make_unl_uni, make_nov, make_scarce_gen, make_topk_gen, top_in_pool의 소스를 inspect.getsource로 읽고 정규식 \b(split|base|valid|test)\b로 검사했다. 일치 행은 모두 [] (0건)이다.
- B 실행 범위 결정 (첫 run 2개 = 각 255–256s, 병렬)
  - P1 25.8분 → +P2 30.1분 → +P3a 34.4분 (≤ 36분) → +P3b 38.7분 (> 36분)
  - 따라서 P1·P2·P3a는 포함하고 **P3b(dblp r=0.2)는 제외**했다. 대기열에서 2줄을 지웠다.
- 실측 B 소요는 42.5분(15:03:08–15:45:38)으로 추정보다 약 8분 길었다.
  - 원인: s1 이후 다른 사용자 프로세스(python, GPU 1.5GB 사용, CPU 400% 이상)와 자원을 같이 썼다.
  - 그 결과 step 시간이 61ms에서 74–78ms로, test 시간이 1s에서 9–15s로 늘었다.
  - 총 경과 시간 60분 기준 안이었다. NaN·OOM·재시도 없음.
핵심 수치:
- A 요약 (U 구성은 숨겨진 base/valid/test 순)
| ds | r | |L| / |U| | U 구성 | AUROC sgc / raw | A_nov novel 비율 | novel 클래스 커버 | gini |
|---|---|---|---|---|---|---|---|
| elec | 0.5 | 11940 / 30378 | .393/.271/.336 | .713 / .784 | .758 | 1.00 | .195 |
| elec | 0.2 | 4771 / 37547 | .509/.219/.272 | .712 / .780 | .648 | 1.00 | .175 |
| dblp | 0.5 | 11708 / 28964 | .405/.259/.337 | .570 / .567 | .644 | 1.00 | .230 |
| dblp | 0.2 | 4683 / 35989 | .521/.208/.271 | .568 / .563 | .523 | 1.00 | .225 |
  - 기대 anchor_novel_frac: NaQ .436/.424 (전체 노드 균등), U 균등 .607/.595 (r=0.5)
- 짝지은 차이 (1s LR, s0 / s1 / s2 → 평균 ± sd, 음수/양수 개수)
| 비교 | eletronics | dblp |
|---|---|---|
| unl_uni_r0.5 − best | +.83/+.58/+.99 → +0.80±.20 (0/3) | +1.55/+.44/+.15 → +0.71±.74 (0/3) |
| nov_r0.5 − best | +1.48/+1.35/+1.19 → **+1.34±.15** (0/3) | +.06/−.11/+.55 → +0.17±.34 (1/2) |
| nov_r0.5 − unl_uni_r0.5 | +.65/+.77/+.20 → +0.54±.30 (0/3) | −1.48/−.55/+.39 → −0.55±.94 (2/1) |
| naq_unl† − best | +3.15/+2.50/+2.12 → +2.59±.52 (0/3) | +1.33/+.35/−.38 → +0.44±.86 (1/2) |
| unl_uni_r0.2 − best (s0) | −0.45 | 미실행 (P3b) |
| nov_r0.2 − best (s0) | +0.99 | 미실행 (P3b) |
  - 5s 평균 (elec / dblp): unl_uni +0.42 / +0.21, nov +1.00 / +0.19, nov − unl_uni +0.58 / −0.02, naq_unl +1.21 / +0.35
- 회수율 R = (nov_r0.5 − best) / (naq_unl − best), seed 평균: elec 1.34/2.59 = **0.52**, dblp 0.17/0.44 = 0.38
  - dblp는 분모 +0.44가 잡음 수준이라 비율 해석에 주의가 필요하다.
- G6 계산값
  - elec: 평균 +1.34, seed별 +1.48/+1.35/+1.19, 양수 3/3 (기준: ≥ +1.0, 모두 양수)
  - dblp: 평균 +0.17 (기준: ≥ −0.5)
- 사전 등록 대조표 (계산값만)
  - R1: AUROC sgc elec .713, dblp .570
  - R2: elec A_nov novel 비율 .758
  - R3: elec unl_uni_r0.5 − best +0.80
  - R4: 평균 +1.34, 양수 3/3
  - R5: elec nov − unl_uni +0.54
  - R6: dblp +0.17
  - R7: elec 평균 NaQ 74.59 / unl_uni 75.39 / nov 75.93 / naq_unl 77.18
  - R8: elec s0 unl_uni_r0.2 −0.45 vs unl_uni_r0.5 +0.83
- 사실 관찰
  - elec에서는 anchor_novel_frac(.43 → .61 → .76 → 1.0)과 1s 평균이 단조 증가한다 (그림 results/T06_dose_response.png).
  - dblp에서는 novelty AUROC가 .57로 낮다. 그래서 A_nov의 novel 비율(.644)이 U 균등(.595)보다 조금만 높다.
  - novelty 앵커는 query 순도가 낮아지고(elec .646 → .606) 엔트로피가 줄었다(.971 → .921). 그럼에도 정확도는 올랐다.
  - novelty_raw의 AUROC(elec .784)가 SGC(.713)보다 높다. raw는 분석용이라 생성기에는 쓰지 않았다.
계획 대비 차이·임의 결정:
- A_nov = U 중 novelty 상위 ceil(|U|/2)개 (안정 정렬). 마스크는 클래스별 round(r·n_c)개, 최소 1개, seed 0이다.
- novelty 캐시: cache/{ds}/novelty_r{r}.npz. 로드 함수는 그래프 구조·feature와 L_idx·y_L만 쓴다 (load_graph의 y는 사용하지 않는다).
- r=1.0 마스크(cache/{ds}/labelmask_r1.0.npz)는 점검 2를 위해 분석 코드에서만 만들었다.
- run_T06.sh: 작업자 2개가 대기열 파일(flock)에서 우선순위 순으로 꺼내 실행한다. 60분 기한 이후에는 새 run을 시작하지 않는다 (해당 없음).
  - 그래서 P3a의 eletronics 2 run은 두 작업자가 하나씩 나눠 돌렸다.
- G6/R의 평균은 사용 가능한 seed 전체로 계산했다. NaQ-best는 s0–2, naq_unl은 s0–2(s2는 이번에 실행).
- naq_unl s2는 T05 코드 그대로 실행했다 (†분석용).
문제·미해결:
- P3b(dblp r=0.2)는 시간 규칙에 따라 실행하지 않았다. 그래서 R8과 r=0.2 비교는 elec s0 하나뿐이다.
- 외부 프로세스의 GPU/CPU 경합으로 run 시간이 25–30% 늘었다. 결과 값에는 영향이 없지만 시간 추정은 빗나갔다.
- 스모크·버그 확인용 짧은 run은 logs/T06_smoke/에만 저장했다.
- results/T03_summary_message.md는 여전히 커밋하지 않았다.
변경·생성 파일:
- cape/labelmask.py, cape/novelty.py, cape/diag_novelty.py, cape/run_T06.sh, cape/summarize_t06.py, cape/generators.py
- results/T06_precheck.json, results/T06A_novelty.{md,json}, results/T06_summary.md, results/T06_dose_response.png
- results/expB/ JSON 16개: {unl_uni,nov}_r0.5 ×2 ds ×3 seed, naq_unl s2 ×2, elec {unl_uni,nov}_r0.2 s0
- reports/T06_report.md
커밋: 50f79d3 (branch cape-validation)
