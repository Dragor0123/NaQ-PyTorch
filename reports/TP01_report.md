[TP01 보고] 상태: SUCCESS
소요 시간: 약 70분 (구현 약 10분 / 실행 약 56분: pprrow 캐시 4분, G0 4분, B1 34분, B2 22분; 집계·보고 약 4분). 시작 07:08, 마지막 run 종료 08:14.

수행 내용:
- §1: branch `cape-validation`, HEAD `8f92e39…`, 추적 변경 ` M CLAUDE.md` 하나, `?? instructions/TP01_token_proto_first.md` 확인. `token_proto` 생성, 두 파일 커밋 `9658520` (코드 작성 전).
  - 그 밖의 untracked 파일: `results/T03_summary_message.md` (건드리지 않음, 커밋하지 않음).
- Ga: 세 데이터셋의 data/*, pool.npz, tasks.npz, split.json, 5way_1/5shot.pkl 모두 존재. 생성한 것 없음.
- pprrow 캐시 3개 생성·검증(통과). Gb 통과. A 진단(앵커 1,000개, 15초). G0 통과.
- B1 21 run, B2 12 run (seed 1, 2) 전부 실행. 실패·NaN 없음.

핵심 수치:
- Gb: (1) max|Δ| 9.5e-7 (a2·b, 5 ep) (2) 9.5e-7 (3) 0.0, 손실·그래디언트 finite (4) eletronics 첫 ep |Δ| 2.4e-7
- pprrow 검증 max_abs_err: elec 3.6e-7 / dblp 3.0e-7 / cloth 4.2e-7, 무작위 노드 > 50번째 점수: 0건. top-50 점수 ≤ 0 비율: elec 0.426, dblp 0.0005, cloth 0.079
- G0 (tp_l0 s0 − naq_feat s0): eletronics Δ1s −0.25, Δ5s +0.10; dblp Δ1s +0.03, Δ5s +0.06
- A (m=8): 토큰 순도 sgc / pprrow / query 순도
  - clothing 0.584 / 0.531 / 0.600, eletronics 0.448 / 0.259 / 0.647, dblp 0.300 / 0.296 / 0.157
  - 부족률 전 설정 0.000. pprrow–ppr 겹침@8: cloth 0.831, elec 0.825, dblp 0.893
  - hop ≤ 2 (sgc / pprrow / query): cloth 0.832 / 0.851 / 0.487, elec 0.556 / 0.621 / 0.185, dblp 0.969 / 0.999 / 0.332
  - 연결 안 됨 (sgc / pprrow / query): cloth 0.075 / 0.070 / 0.112, elec 0.349 / 0.260 / 0.638, dblp 0 / 0 / 0
- B1 차이 (arm − tp_l0, 1s / 5s LR, %p) — clothing | eletronics | dblp
  - a2_sgc_l25_m4: +0.48/−0.06 | +0.03/+0.28 | −0.20/+0.14
  - a2_sgc_l25_m8: +0.41/−0.01 | +0.20/+0.21 | +0.41/+0.69
  - a2_sgc_l50_m4: +0.07/+0.01 | −1.23/−0.23 | +0.24/+0.86
  - a2_sgc_l50_m8: −0.41/−0.06 | +0.14/+0.15 | +0.31/+0.91
  - b_sgc_l50_m8: −0.33/−0.19 | −1.21/−0.02 | −0.66/+0.64
  - a2_pprrow_l50_m8: −0.90/−0.07 | −0.52/+0.10 | +0.30/+0.67
  - tp_l0 s0 1s/5s: cloth 84.78/92.14, elec 74.33/87.54, dblp 71.75/81.41
- 선택: 세 데이터셋 평균 best_valid — pprrow_l50_m8 71.783, sgc_l25_m8 71.767, sgc_l25_m4 71.717, sgc_l50_m4 71.517, sgc_l50_m8 71.367, b_sgc_l50_m8 70.350 → **C\* = `tp_a2_pprrow_l50_m8`** (동률 없음)
- B2 짝지은 차이 C\* − tp_l0 (s0, s1, s2 / 평균 ± sd / 양수 수)
  - clothing 1s: −0.90, −0.70, −1.57 / −1.06 ± 0.45 / 0 of 3; 5s: −0.12 ± 0.05 / 0 of 3
  - eletronics 1s: −0.52, −0.76, −0.82 / −0.70 ± 0.16 / 0 of 3; 5s: +0.23 ± 0.14 / 3 of 3
  - dblp 1s: +0.30, +0.03, +0.30 / +0.21 ± 0.15 / 3 of 3; 5s: +0.57 ± 0.22 / 3 of 3
- attention (a2_sgc_l50_m8, 정규화 엔트로피): cloth 0.968, elec 0.951, dblp 0.975. 같은 레이블 질량 / 균등 기대값: 0.589/0.583, 0.477/0.465, 0.308/0.305
- tp_l0 vs 기존 naq_feat 1s: elec s0 74.33 vs 74.58, s1 74.70 vs 74.69, s2 74.52 vs 74.50; dblp s0 71.75 vs 71.72
- peak GPU 0.49 GB/프로세스. run당 평균 213초 (2개 동시 실행).

사전 등록 대조:
- TPP001: Gb1 9.5e-7, Gb2 9.5e-7, Gb3 0, Gb4 2.4e-7; 기준 ≤ 1e-5
- TPP002: elec |Δ1s| 0.25, |Δ5s| 0.10; dblp |Δ1s| 0.03, |Δ5s| 0.06; 기준 ≤ 0.7
- TPP003: elec 토큰 0.448 vs query 0.647; dblp 토큰 0.300 vs query 0.157
- TPP004: 겹침@8 0.831 / 0.825 / 0.893 (기준 ≥ 0.5); pprrow hop ≤ 2 0.851 / 0.621 / 0.999 (기준 ≥ 0.7) — cloth / elec / dblp
- TPP005: A2-sgc 중 best_valid 최고 설정과 Δ1s: cloth l25_m8 +0.41, elec l25_m4 +0.03, dblp l50_m8 +0.31; 기준 +0.5 이상인 데이터셋이 2개 이상
- TPP006: a2_sgc_l50_m8 Δ1s: cloth −0.41, elec +0.14, dblp +0.31
- TPP007: 1s b vs a2: cloth 84.45 vs 84.37, elec 73.12 vs 74.47, dblp 71.08 vs 72.06
- TPP008: 1s sgc vs pprrow: cloth 84.37 vs 83.88, elec 74.47 vs 73.81, dblp 72.06 vs 72.05
- TPP009: 엔트로피 0.968 / 0.951 / 0.975; 기준 ≤ 0.9
- TPP010: C\* 평균 Δ1s cloth −1.06 (양수 0/3), elec −0.70 (0/3), dblp +0.21 (3/3); 기준: +0.5 이상이고 양수 2개 이상인 데이터셋이 2개 이상

계획 대비 차이·임의 결정:
- 시간 규칙 §7: G0 두 run(동시 실행)이 4분 18초 걸렸다. 남은 31 run × 257초 / 작업자 2 ≈ 66분 ≤ 75분으로 추정하고 전부 실행했다. 실제로는 B1+B2에 56분이 걸렸다.
- G0 run과 A 진단(CPU, 15초)이 함께 돌았다.
- attention 진단의 "같은 레이블"은 **앵커의 레이블**로 정했다 (토큰 순도와 같은 기준). query 레이블 기준의 질량(`mass_query_label`)도 JSON에 함께 남겼다. 정규화 엔트로피는 |T| ≥ 2인 경우만 평균했다 (부족률이 0이라 모든 경우 해당).
- 토큰이 없는 way는 softmax 점수를 −1e9로 마스크한 뒤 α에 마스크를 곱하고, prototype을 앵커 임베딩으로 둔다 (`t = h_a + λ·[|T|>0]·(Σα h − h_a)`). 학습 run에서는 토큰 0개인 경우가 생기지 않았다 (empty_rate 0).
- pprrow 후보: 정의대로 점수 내림차순 top-50을 쓰고 점수 0인 후보도 빼지 않았다. eletronics는 top-50의 42.6%가 점수 0이다 (작은 연결 요소. 점수 0 동률 중 어떤 노드가 들어가는지는 `torch.topk` 결과를 따르고 캐시에 고정됨). 앵커 1,000개의 pprrow top-8 중 점수 ≤ 0인 비율은 `results/TP01A_diag.json`의 `pprrow_zero_score_frac_top8`에 있다.
- pprrow 검증의 "무작위 노드 1,000개"에서는 자기 자신과 top-50에 이미 든 노드를 뺐고, 허용 오차 1e-4를 두었다.
- 에피소드 생성은 `cape.generators.load_inputs` 대신 `pool.npz`를 직접 읽고 `make_topk_gen(top_in_pool(nbr, cue[:,:,0], 10))`로 했다 (load_inputs는 rng를 쓰지 않음). seed 0의 episode_stats(query_purity 0.64627/0.15567, anchor_unique_frac)와 valid 곡선 초기값(ep 0: 0.416 / 0.5345)이 expB와 같다.
- 대기열은 `xargs -P 2`로 돌렸다. 큐 파일 `tokproto/queue_{g0,b1,b2}.txt`를 추가했다.

문제·미해결:
- 없음

변경·생성 파일:
- tokproto/: `__init__.py pprrow.py tokens.py loss.py test_equiv.py diag_tokens.py train_tp.py run_TP01.sh summarize_tp01.py queue_g0.txt queue_b1.txt queue_b2.txt`
- results/tp/: run JSON 33개, `TP01_Gb.json`, `{ds}_pprrow_build.json` 3개
- results/: `TP01A_diag.{json,md}`, `TP01_summary.{json,md}`
- cache/{ds}/pprrow_top50.npz 3개 (gitignore), logs/TP01_* (gitignore)
- reports/TP01_report.md
커밋: f1a5f7d (branch token_proto)
