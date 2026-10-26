[T09 보고] 상태: SUCCESS
소요 시간: 약 82분 (git·점검 약 1분 / 구현 약 2분 (큐 실행 중 집계 코드 작성) / 실행 78분 05:17:04–06:35:04 / 집계·보고 약 3분). 시작 05:16.

수행 내용:
- §0 git
  - (1) `git status --short`: 추적 파일 변경 없음. untracked는 `results/T03_summary_message.md` 하나 (`results/T07_message.txt`는 이미 없음).
  - (2) `git push -u origin token_proto`: 성공 (`[new branch] token_proto -> token_proto`).
  - (3) `git checkout cape-validation`: HEAD `8f92e39` = `origin/cape-validation` `8f92e39`.
  - (4) `~/wkspace/T09_drop/`의 CLAUDE.md → 저장소 루트, T09 지시서 → `instructions/`. `cmp`로 원본과 같음을 확인했고, 새 CLAUDE.md를 다시 읽음.
  - (5) 커밋 `f58175d` "T09: instruction, preregistration, CLAUDE.md update" 후 push 성공 (`8f92e39..f58175d`).
- §1 사전 점검
  - 디스크 여유 325GB → `--save_emb` 유지.
  - GPU: 다른 프로세스 없음 (2 MiB 사용). 작업자 2개로 실행. run당 최대 peak 0.49GB.
  - Clothing 캐시 5종(tasks, pool, scores, labelmask_r0.5, novelty2_r0.5)이 모두 이미 있었다. 새로 만든 파일이 없으므로 AUROC 재계산은 해당 없음.
  - 소스 검사: T07/T08과 같은 정규식 검사기로 `cape.summarize_t09 --source_check`를 실행해 통과 (`results/T09_source_check.json`).
- §3 큐: `cape/run_T09.sh` (T08과 같은 방식: 작업자 2, flock, 1회 재시도, DEADLINE = 시작 + 105분 = 07:02:05).
  - 기존 JSON과 겹치는 run은 0개라 47 run을 모두 큐에 넣었다.
  - 첫 run의 JSON 필드(test1/5_lr·proto, valid 곡선 17점, anchor_novel_frac)가 T08과 같음을 확인했다.
- §4 시간 추정
  - 1차: Clothing 첫 2 run이 각 127초 → 전체 약 78분.
  - 2차: P2 Electronics 첫 run 4.05분 → 06:33 종료 예상, 큐 전체 약 76분.
  - 둘 다 100분 이하라 제외한 run은 없다.
- 결과: 47/47 완료. FAILED·RETRY·DEADLINE skip 없음. 큐 실행 78분.
- 집계: `cape/summarize_t09.py` → `results/T09_summary.{md,json}`. 표 전체는 summary.md에 있다.

핵심 수치 (1s LR / 5s LR, seed 평균 ± sd; n = seed 수):

| dataset | NaQ-best | nov | naq_mix ‡ | **mix_alt ‡** | sup_only ‡ | naq_unl † |
|---|---|---|---|---|---|---|
| Electronics (n=5; †3) | 74.72±.19 / 87.52±.13 | 76.02±.42 / 88.21±.15 | 74.56±.52 / 87.35±.21 | **75.87±.54 / 87.84±.23** | 68.42±.53 / 83.70±.30 | 77.18±.56 / 88.63±.22 |
| dblp (n=5; †3) | 73.78±.54 / 82.86±.30 | 73.59±.56 / 82.89±.27 | 76.75±.30 / 83.58±.07 | **76.50±.54 / 83.66±.17** | 74.29±.28 / 81.88±.32 | 74.16±.68 / 83.14±.32 |
| Clothing (n=3) | 84.88±.43 / 92.25±.15 | 84.84±1.15 / 92.10±.17 | 84.76±.37 / 92.23±.15 | **85.38±.34 / 92.23±.24** | 76.59±.80 / 89.64±.52 | 84.58±.42 / 92.18±.09 |

- Clothing `naq_diff`: 80.72±.51 / 91.10±.27.
- NaQ-best: Electronics `naq_feat`, dblp `naq_diff`, Clothing `naq_feat`.

짝지은 차이 (1s 평균 (음수/양수) | 5s 평균):

| 비교 | Electronics | dblp | Clothing |
|---|---|---|---|
| mix_alt − NaQ-best | +1.15 (0/5) \| +0.32 | +2.73 (0/5) \| +0.80 | +0.50 (1/2) \| −0.02 |
| mix_alt − naq_mix | +1.31 (0/5) \| +0.49 | −0.25 (3/2) \| +0.07 | +0.62 (0/3) \| +0.00 |
| naq_mix − NaQ-best | −0.16 (3/2) \| −0.17 | +2.98 (0/5) \| +0.73 | −0.12 (1/2) \| −0.02 |
| nov − NaQ-best | +1.31 (0/5) | −0.19 (4/1) | −0.04 (2/1) |
| naq_unl† − NaQ-best | +2.59 (0/3) \| +1.21 | +0.44 (1/2) \| +0.35 | −0.31 (2/1) \| −0.07 |
| 회수율 R / R_mix | 0.51 / 0.45 | −0.43 / 6.25 | 0.14 / −1.64 |

- Electronics `mix_alt − NaQ-best` seed별 값: +1.79, +1.56, +1.28, +1.09, +0.05.
- Clothing `naq_diff − naq_feat` = −4.16 (3/0).

분해 (1s, 네 arm의 공통 seed; 경로 A: NaQ→naq_mix→mix_alt / 경로 B: NaQ→nov→mix_alt):

| dataset | seeds | A 1단계 | A 2단계 | 합 | B 1단계 | B 2단계 | 합 |
|---|---|---|---|---|---|---|---|
| Electronics | 0–4 | −0.16 | +1.31 | +1.15 | +1.31 | −0.15 | +1.15 |
| dblp | 0–4 | +2.98 | −0.25 | +2.73 | −0.19 | +2.92 | +2.73 |
| Clothing | 0–2 | −0.12 | +0.62 | +0.50 | −0.04 | +0.54 | +0.50 |

용량-반응 (1s; pseudo 앵커 novel 비율 → Δ vs NaQ-best):
- Electronics: NaQ .433→0, nov2c .676→+0.52, nov .878→+1.31, mix_alt (.878)→+1.15, naq_unl† 1.00→+2.59. ρ = 0.975.
- dblp: .427→0, .711→+0.86, .809→−0.19, (.809)→+2.73, 1.00→+0.44. ρ = 0.103.
- Clothing: .570→0, .935→−0.04, (.935)→+0.50, 1.00→−0.31. 점이 4개라 ρ는 비움.

G9 계산값 (판정은 사람이 한다):
- G9-a: mix_alt − NaQ-best (1s) = Electronics +1.15 (양수 5/5), dblp +2.73 (5/5), Clothing +0.50 (2/3)
- G9-b: Electronics mix_alt − naq_mix = +1.31 (양수 5/5)
- G9-c: mix_alt − sup_only = Electronics +7.45, dblp +2.22, Clothing +8.79

사전 등록 대조 (계산값만):

| ID | 기준 | 계산값 |
|---|---|---|
| V1 | Electronics mix_alt − naq_mix ≥ +0.7, 양수 ≥ 4/5 | +1.31, 5/5 |
| V2 | dblp naq_mix − NaQ-best ≥ +1.5, \|mix_alt − naq_mix\| ≤ 1.0 | +2.98; −0.25 |
| V3 | mix_alt − NaQ-best: Electronics ≥ +1.3, dblp ≥ +2.0 | +1.15; +2.73 |
| V4 | Clothing nov − naq_feat ≥ +1.0, 3/3 양수 | −0.04, 양수 1/3 |
| V5 | Clothing mix_alt − naq_mix ≥ +0.7 | +0.62 |
| V6 | Clothing naq_unl† − naq_feat ≥ +1.5 | −0.31 |
| V7 | Clothing sup_only − naq_feat ≤ −2.0 | −8.29 |
| V8 | Electronics ρ ≥ 0.8 | 0.975 |
| V9 | Clothing naq_feat − naq_diff > 0 | +4.16 |

제외된 run: 없음 (P1 12, P2 12, P3 8, P4 9, P5 6 = 47 완료. run 시간 합: P1 25.5분, P2 51.3분, P3 34.0분, P4 19.6분, P5 22.7분).

계획 대비 차이·임의 결정:
- 소스 검사기는 T08 검사기와 같은 정규식이다. 검사 대상 함수 목록은 T09에서 쓰는 경로에 맞췄다.
  - 보정 추정기 `compute_estimators3`, `pu_novelty2` 대신 `compute_estimators`(pu_feat)를 넣었다.
  - 별도 파일을 만들 수 없어서 `summarize_t09.py`의 `--source_check` 옵션으로 구현했다.
- 이전 run의 JSON(T03–T05) 중 `anchor_novel_frac` 필드가 없는 것은 anchor_group_frac의 valid + test로 계산했다 (T08과 같은 대체 방식).
- 태그가 붙은 파일(`_emb`, `_e8000`)은 집계에서 뺐다 (T08과 같다).
- Electronics·dblp의 `naq_unl†`은 T09 계획에 없어 seed 0–2(3개)뿐이다. 그래서 상한과 회수율은 seed 3개 기준이다.
- 회수율 R은 분자와 분모를 각자의 공통 seed 평균으로 계산했다.
- 용량-반응의 nov2c_t1 점은 T08 run(seed 0–2)을 썼다.
- mix_alt는 규칙에 따라 nov와 같은 x값을 쓴다. 그래서 Spearman 계산에 동순위(tie)가 들어 있다.
- `run_T09.sh`에 `SAVE_EMB` 환경변수(기본 `--save_emb`)를 두었다. 디스크가 충분해 기본값으로 실행했다.

문제·미해결:
- Clothing은 seed 3개(0–2)뿐이다 (지시서 계획대로).
- dblp 회수율은 상한(+0.44)이 작아서 R_mix = 6.25이고, Clothing은 상한이 음수(−0.31)여서 R_mix = −1.64이다. 둘 다 분모가 0에 가깝거나 음수인 비율이다.
- `results/T03_summary_message.md`는 커밋하지 않았다.

변경·생성 파일:
- `CLAUDE.md`, `instructions/T09_consolidate_seeds_clothing.md` (사용자 파일 복사, 커밋 f58175d)
- `cape/run_T09.sh`, `cape/summarize_t09.py`
- `results/T09_source_check.json`, `results/T09_summary.md`, `results/T09_summary.json`
- `results/expB/` JSON 47개 (Clothing 21, Electronics 13, dblp 13)
- `reports/T09_report.md`
커밋: d9cc9cf (branch cape-validation), push: 성공
