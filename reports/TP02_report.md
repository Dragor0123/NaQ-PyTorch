[TP02 보고] 상태: SUCCESS
소요 시간: 약 48분 (구현 약 2분 / 실행 43분 08:53:53–09:37:01 / 집계·보고 약 3분). 시작 08:52.

수행 내용:
- §1: branch `token_proto`, `git log -1` = `c3f6ea7`, 추적 변경 ` M CLAUDE.md` 하나, `?? instructions/TP02_purity_and_attention.md` 확인. 두 파일 커밋 `bdd39e1` (코드 작성 전). 그 밖의 untracked 파일: `results/T03_summary_message.md` (건드리지 않음).
- 구현: `train_tp.py --attn {softmax,mean}`, `--tok sgcora`, arm 접미사(`_mean`, `_t02`), JSON `attn`·`analysis_only`, `token_stats.count_dist`. 기본 경로의 동작은 바꾸지 않았다.
- 회귀 확인(§2) 통과 → 24 run 전부 실행 (rc 0, NaN 없음). §4 진단, 집계.

회귀 확인 (`results/tp/TP02_regression.json`, 세 데이터셋, seed 0 첫 에피소드, 초기화 직후 GCN, `tp_a2_sgc_l25_m8`):
- 새 코드 vs TP01 코드(`git show c3f6ea7:tokproto/{loss,tokens}.py`): 토큰 집합 동일, |ΔL| = 0.0 (세 데이터셋)
- 동일 토큰일 때 mean vs softmax: |ΔL| = 0.0 (세 데이터셋)

oracle_filter 호출 경로 (`grep -rn oracle_filter tokproto/`):
- `tokproto/tokens.py:23` 정의 (레이블 로드 `tokens.py:26-27`는 이 함수 안)
- `tokproto/train_tp.py:74` 호출 — `if args.tok == "sgcora":` 분기 안에서만
- `tokproto/diag_oracle.py:21` 호출 (§4 진단 스크립트)
- `train_tp.py:68`의 `y = g["y"]`는 TP01부터 있던 분석용(episode_stats, token_stats, attn_diag)이며 토큰 선정·손실·선택에는 들어가지 않는다.

핵심 수치:
- §4 sgcora† (seed 0, 2,000 ep, m=8): 부족률 clothing 0.251 / eletronics 0.311 / dblp 0.451; 토큰 0개 0.063 / 0.083 / 0.071; 평균 토큰 수 6.64 / 6.32 / 5.79. 같은 에피소드 sgc 토큰 순도 0.584 / 0.448 / 0.300 (TP01 값과 같음)
- 3 seed 평균 ± sd (1s LR): clothing tp_l0 84.75±0.64, sgc_l25_m8 84.81±0.42, _mean 84.52±0.43; eletronics 74.52±0.19, 74.47±0.09, 74.40±0.25; dblp 72.48±0.65, 72.33±0.16, 72.04±0.42
- 3 seed 평균 ± sd (5s LR): clothing 92.21 / 92.20 / 92.11; eletronics 87.45 / 87.79 / 87.86; dblp 81.82 / 82.07 / 82.01
- 짝지은 차이 1s (s0, s1, s2 → 평균, 양수 수):
  - sgc_l25_m8 − tp_l0: clothing +0.41, +0.25, −0.47 → +0.06 (2/3); eletronics +0.20, −0.33, +0.005 → −0.04 (2/3); dblp +0.41, −0.28, −0.60 → −0.16 (1/3)
  - _mean − tp_l0: clothing −0.23 (0/3), eletronics −0.12 (2/3), dblp −0.45 (0/3)
  - sgc_l25_m8 − _mean (attention의 몫): clothing +0.61, +0.28, −0.03 → +0.285 (2/3); eletronics −0.01, +0.25, −0.02 → +0.075 (1/3); dblp +0.56, +0.04, +0.27 → +0.29 (3/3)
- 짝지은 차이 5s 평균 (양수 수): sgc_l25_m8 − tp_l0: −0.01 (1/3), +0.34 (3/3), +0.24 (2/3); _mean − tp_l0: −0.10 (1/3), +0.41 (3/3), +0.19 (3/3); sgc − _mean: +0.09 (1/3), −0.07 (1/3), +0.06 (2/3)
- seed 0 (1s LR / 엔트로피) — clothing | eletronics | dblp
  - tp_l0 84.78 | 74.33 | 71.75
  - sgc_l25_m8 85.18 / .971 | 74.53 / .950 | 72.15 / .971
  - sgc_l50_m8 84.37 / .968 | 74.47 / .951 | 72.06 / .975
  - sgcora_l25_m8† 84.41 / .976 | 74.17 / .953 | 72.06 / .960
  - sgcora_l50_m8† 84.93 / .972 | 74.00 / .958 | 71.44 / .958
  - sgc_l25_m8_t02 84.84 / .812 | 74.04 / .692 | 72.20 / .720 (같은 레이블 질량 / 균등: .598/.583, .493/.465, .313/.305)
- seed 0 차이 1s: sgcora† − sgc: l25 −0.77 / −0.36 / −0.09, l50 +0.56 / −0.47 / −0.62; sgcora† − tp_l0: l25 −0.37 / −0.15 / +0.31, l50 +0.15 / −0.33 / −0.31; t02 − τ0.1: −0.34 / −0.49 / +0.05

사전 등록 대조:
- TPP011: sgc_l25_m8 − tp_l0 3 seed 평균 1s: clothing +0.06 (양수 2), eletronics −0.04 (양수 2), dblp −0.16 (양수 1); 기준: 평균 ≥ +0.3이고 양수 ≥ 2인 데이터셋이 1개 이하
- TPP012: |sgc − mean| 3 seed 평균 1s: clothing 0.285, eletronics 0.075, dblp 0.290; 기준 ≤ 0.3 (세 데이터셋 모두)
- TPP013: t02 엔트로피 0.812 / 0.692 / 0.720; 기준 ≤ 0.8 (세 데이터셋 모두)
- TPP014: t02 − sgc_l25_m8 (s0, 1s): −0.34 / −0.49 / +0.05; 기준: ≤ +0.3인 데이터셋이 2개 이상
- TPP015: sgcora_l50_m8† − tp_l0 (s0, 1s): +0.15 / −0.33 / −0.31; 기준: ≥ +1.0인 데이터셋이 1개 이하
- TPP016: sgcora_l50_m8† − sgc_l50_m8 (s0, 1s): clothing +0.56, eletronics −0.47, dblp −0.62; 기준: eletronics에서 가장 큼

계획 대비 차이·임의 결정:
- 시간 규칙 §6: 첫 run 2개(clothing 166초, eletronics 214초, 동시 실행) 종료 시점에 남은 22 run × 약 205초 / 작업자 2 ≈ 38분 ≤ 50분으로 추정 → P1–P4 전부 실행. 실제 전체 실행 43분.
- `oracle_filter(cand, ds)`는 함수 안에서 `load_graph(ds)["y"]`를 읽는다 (레이블 접근을 이 함수 하나에 두기 위해). 필터 후 후보 순서는 sgc 점수 순서를 유지한다.
- `--attn mean`의 arm 이름에는 τ 접미사를 붙이지 않는다 (τ 무시). 토큰 0개인 way의 mean α는 0 → prototype은 앵커 임베딩 (TP01과 같은 규칙).
- sgcora†의 attention 진단에서 같은 레이블 질량/균등 기대값은 정의상 1.000 / 1.000이다.
- seed 0 표의 토큰 순도는 학습 run의 4,000 ep 기준(sgc 0.580/0.448/0.299)이고, §4 진단은 2,000 ep 기준(0.584/0.448/0.300)이다.
- 재사용한 TP01 JSON(`tp_l0` s0–2, `sgc_l25_m8` s0, `sgc_l50_m8` s0)에는 `attn`, `analysis_only` 필드가 없다 (수정하지 않음).
- 양수 개수는 > 0만 센다. eletronics sgc_l25_m8 − tp_l0의 s2 값은 74.520 − 74.515 = +0.005로 양수로 셌다 (요약 표에는 +0.00으로 표시).

문제·미해결:
- 없음

변경·생성 파일:
- 수정: `tokproto/tokens.py` (oracle_filter, count_dist), `tokproto/loss.py` (attn 인자), `tokproto/train_tp.py` (--attn, --tok sgcora, arm 이름, JSON 필드)
- 생성: `tokproto/{test_tp02.py, diag_oracle.py, summarize_tp02.py, run_TP02.sh, queue_tp02.txt}`
- `results/tp/`: 새 run JSON 24개, `TP02_regression.json`
- `results/TP02A_oracle_tokens.{json,md}`, `results/TP02_summary.{json,md}`, `reports/TP02_report.md`
커밋: 6b0df01 (branch token_proto)
