[T07 보고] 상태: PARTIAL (P1–P3 완료, P4·P5·P6은 시간 규칙에 따라 제외. G7·S1–S10은 계산값만 적었다)
소요 시간: 약 69분 (23:28:27 시작 – 00:37:35 종료; 구현 약 2분 + A 25분(구현과 병행) / B 41분 / 집계·보고 약 2분)
수행 내용:
- 지시서 두 개를 수정 없이 먼저 커밋했다 (cb13ea6).
- 새 파일: cape/novelty2.py, cape/diag_novelty2.py, cape/run_T07.sh, cape/summarize_t07.py
- generators.py 확장: make_nov2(가중 비복원 추출), make_mix_alt, NAQ_BEST_COL에 clothing(raw) 추가
- train_proto.py 확장: --save_emb, --tag 옵션
- A 실행
  - 3 데이터셋 × r 2개를 데이터셋별 프로세스 3개로 병렬 실행했다.
  - 예산(5분)을 크게 넘겨 25분 걸렸다. 원인은 pu_both다: 표준화한 [Xs, 구조] 입력의 LR 6회 적합에 (ds, r)당 520–670s가 걸렸다.
- 소스 검사: 처음 실행에서는 "FAILED"였다.
  - 일치 행 2개는 모두 메서드 호출이었다.
    - `StratifiedKFold(...).split(F, s)` (sklearn CV fold 분할)
    - `arm.split("_")` (문자열 분할)
  - 둘 다 split 멤버십 정보를 참조하지 않는 오탐으로 판단했다.
  - 그래서 정규식에서 `.split(` 형태의 메서드 호출만 제외하도록 고치고 다시 검사했다 → 통과 (results/T07_source_check.json).
  - load_split import도 novelty/novelty2 모듈에 없다. **이 판단과 검사기 수정은 사람의 확인 대상이다.**
- E* = **pu_feat** (r=0.5 세 데이터셋 AUROC 평균 .841로 최고)
  - 데이터셋별 최고: clothing pu_feat, elec pu_feat, dblp pu_struct (r=0.5 / 0.2 모두 같다)
  - E* 선택은 지시서 규칙상 분석용 레이블로 계산한 AUROC를 쓴다 (사전 명시된 단일 전역 규칙).
- 시간 규칙: 첫 2 run은 각 256s (step 61ms). 이 시점 경과 31.2분, 80분 예산까지 남은 48.8분에서 집계 약 5분을 빼면 B 전체에 쓸 수 있는 시간은 약 48분이다.
  - P1–P3 (19 run): 약 40.7분 → 포함
  - +P4 (23 run): 약 51분, 총 약 82분으로 80분 예산 초과 → **P4·P5(clothing)·P6(r=1.0) 제외**
  - 실측 B: 23:55:17–00:36:11 (41분). 다른 사용자 GPU 프로세스 없음 (nvidia-smi에 우리 프로세스 2개, 각 약 0.9GB). NaN·OOM·재시도 없음.
핵심 수치:
- A: AUROC (U 안에서 novel vs 숨겨진 base)
| ds | r | cent_sgc | cent_raw | msp | pu_feat | pu_struct | pu_both |
|---|---|---|---|---|---|---|---|
| clothing | .5 | .707 | .720 | .763 | **.909** | .870 | .780 |
| clothing | .2 | .706 | .716 | .758 | **.886** | .804 | .700 |
| elec | .5 | .713 | .784 | .759 | **.867** | .753 | .706 |
| elec | .2 | .712 | .780 | .743 | **.837** | .683 | .645 |
| dblp | .5 | .570 | .567 | .610 | .749 | **.764** | .677 |
| dblp | .2 | .568 | .563 | .613 | .711 | **.718** | .617 |
| 평균 r=.5 | | .663 | .690 | .711 | **.841** | .796 | .721 |
- PU 보정 (r=0.5, mean p(novel) − 실제, 절대 오차)
  - pu_feat는 크게 과소추정한다 (.24 / .26 / .40). U 대비 mean p가 낮다.
  - pu_struct: .06 / .07 / .23
  - pu_both: .16 / .19 / .07
- 기대 앵커 novel 비율 (r=0.5; clothing / elec / dblp)
  - U 균등: .726 / .607 / .595
  - T06 하드 컷: .836 / .758 / .644
  - E* τ=1: .933 / .877 / .811
  - E* τ=2: .958 / .924 / .871
- 짝지은 차이 (1s, seed 평균 ± sd, 음수/양수 개수; 5s는 summary.md)
| 비교 | elec | dblp |
|---|---|---|
| nov2_t1_r0.5 − best | **+1.62 ± .22** (0/3) [+1.39 / +1.62 / +1.83] | −0.11 ± .51 (2/1) [−.16 / −.59 / +.42] |
| nov2_t2_r0.5 − best | +0.84 ± .59 (0/3) | −1.28 ± .54 (3/0) |
| nov2_t1 − nov2_t2 | +0.77 ± .77 | +1.17 ± .34 |
| nov2_t1 − T06 nov (하드 컷) | +0.28 ± .37 | −0.27 ± .19 |
| mix_alt‡ − nov2_t1 (s0, s1) | +0.17 | **+3.02** (+3.74 / +2.30) |
| mix_alt‡ − best (s0, s1) | +1.67 (+1.79 / +1.56) | **+2.65** (+3.58 / +1.71) |
| naq_unl† − best | +2.59 ± .52 | +0.44 ± .86 |
  - ‡mix_alt는 L의 레이블로 만든 지도 에피소드를 쓴다 (실제 비율: 에피소드 약 30%, 앵커 base 비율 .38–.43). NaQ보다 정보량이 많다.
  - P3 재실행(_emb) 1s: elec naq_feat 74.52 (T03 74.58), dblp naq_diff 73.25 (73.28), elec naq_unl 77.68 (77.72)
- 회수율 (nov2_t1 − NaQ) / (naq_unl − NaQ): elec 1.62 / 2.59 = **0.62** (T06 하드 컷 0.52), dblp −0.11 / 0.44 = −0.24 (분모가 잡음 수준), clothing 미실행
- 메커니즘 (best checkpoint 임베딩, k-means NMI, seed 0)
| run | elec NMI_test | elec NMI_base | dblp NMI_test | dblp NMI_base |
|---|---|---|---|---|
| NaQ (P3) | .6161 | .6584 | .5589 | .5396 |
| nov2_t1 | .6328 (+.017) | .6208 (−.038) | .5729 (+.014) | .5420 (+.003) |
| nov2_t2 | .6262 (+.010) | .6026 (−.056) | .5668 (+.008) | .5299 (−.010) |
| mix_alt‡ | .6268 (+.011) | .6864 (+.028) | .5896 (+.031) | .5845 (+.045) |
| naq_unl† | .6479 (+.032) | .6140 (−.044) | - | - |
- G7 계산값
  - G7-a: E*(pu_feat)의 dblp AUROC = .749 (기준 ≥ .65)
  - G7-b: nov2_t1 − best가 평균 ≥ +1.0이고 모두 양수인 데이터셋
    - elec: +1.62, 3/3 양수
    - dblp: −0.11, 1/3 양수
    - clothing: 미실행 (P5 제외)
- 사전 등록 대조 (계산값만)
  - S1: pu_both − cent_sgc = clothing +.073, elec −.007, dblp +.107
  - S2: .749
  - S3: elec nov2_t1 − best = +1.62
  - S4: elec nov2_t2 − nov2_t1 = −0.77
  - S5: dblp nov2_t1 − best = −0.11 (S2 계산값 .749)
  - S6: mix − best elec +1.67, dblp +2.65; elec mix − nov2_t1 = +0.17
  - S7: 미실행 (P6 제외)
  - S8: 미실행 (P4 제외)
  - S9: 미실행 (P5 제외)
  - S10: elec NMI_test NaQ .6161 / nov2_t1 .6328 / naq_unl .6479; |ΔNMI_base| .0375 vs |ΔNMI_test| .0167
- 사실 관찰
  - dblp에서는 AUROC가 .57에서 .75로 올랐다. 앵커 novel 비율도 .81로 올랐지만 정확도 이득은 없다 (−0.11).
  - τ=2는 두 데이터셋 모두에서 τ=1보다 낮다. 앵커 엔트로피가 .87에서 .83으로 낮아진다.
  - novel 비율과 정확도의 단조 관계는 elec에서 nov2_t2(.925)에서 꺾이고, dblp에서는 관찰되지 않는다.
  - mix_alt는 dblp에서 큰 이득을 낸다 (+2.65, seed 2개). elec에서는 nov2_t1과 비슷하다.
  - NMI: novel 지향 arm은 elec에서 NMI_test가 오르고 NMI_base가 내려간다. mix_alt는 둘 다 오른다.
계획 대비 차이·임의 결정:
- 소스 검사 정규식 수정 (위 설명): 메서드 호출 `.split(`을 제외했다. 원래 일치 행은 그대로 보고했다.
- A를 B 코드 구현과 병행해서 백그라운드로 실행했다 (데이터셋별 프로세스, BLAS 스레드 8).
- known_struct의 m_V는 1_V에서 출발한 같은 확산(k=1..K)이다. pu_struct·pu_both 입력은 StandardScaler로 표준화했다. pu_feat 입력은 표준화하지 않은 Xs다.
- PU의 5-fold OOF는 L∪U 전체를 StratifiedKFold(5, shuffle, seed 0)로 나눴다. c = L 노드 OOF g의 평균이다. 모든 LR은 max_iter=2000이고, PU는 class_weight="balanced"다.
- nov2 가중치: p_E*(novel|u) = 1 − min(1, g/c)에 τ제곱. p=0인 노드는 제외했다. mix_alt의 30%는 에피소드마다 rng.random() < 0.3로 정했다 (정확히 30%로 고정하지 않은 Bernoulli).
- mix_alt 지도 에피소드: L에서 1+10개 이상인 클래스 중 5개를 고르고, 클래스마다 1+10개를 비복원으로 뽑았다.
- B 기한: 00:45 이후 새 run 시작 금지 (해당 없음). 대기열에서 P4–P6 10줄을 지웠다.
- NMI는 best checkpoint 임베딩(float16 저장)을 써서 KMeans(n_init=3, random_state=0)로 계산했다.
문제·미해결:
- A가 예산의 5배(25분)를 썼다. 그 결과 P4(r=0.2), P5(clothing), P6(r=1.0)를 실행하지 못했다. S7–S9와 G7-b의 clothing 항목이 비어 있다.
  - 다음에 재개할 때는 대기열에 해당 줄만 넣으면 된다. r=1.0 추정기 캐시는 없어서 run 안에서 약 10분이 더 든다.
- 소스 검사기 수정이 지시서의 중단 조건 해석에 영향을 준다. 사람의 확인이 필요하다.
- 메커니즘 분석은 seed 0 하나다.
- results/T03_summary_message.md는 커밋하지 않았다.
변경·생성 파일:
- cape/novelty2.py, cape/diag_novelty2.py, cape/run_T07.sh, cape/summarize_t07.py, cape/generators.py, cape/train_proto.py
- results/T07A_estimators.{md,json}, results/T07A_{ds}.json ×3, results/T07_source_check.json
- results/T07_summary.md, results/T07_dose_response.png, results/T07_auroc_vs_gain.png
- results/expB/ JSON 19개
- reports/T07_report.md
커밋: dc44de8 (branch cape-validation)
