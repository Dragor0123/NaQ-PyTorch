[T01 보고] 상태: SUCCESS
소요 시간: 약 10분 (구현 7분 / 실행 1분 미만)
수행 내용:
- cape-validation 브랜치, .gitignore, cape/ cache/ results/ logs/ reports/ 생성
- cape/data.py (load_graph, norm_adj, to_torch_sparse, load_split, load_tasks), cape/check_data.py, cape/smoke_gpu.py 작성
- 3개 데이터셋 cache/{ds}/split.json, tasks.npz 생성, 검사 1–4 전부 통과
- 환경: Python 3.9.25, torch 2.4.1+cu121, CUDA 사용 가능, RTX 4080 SUPER; numpy 1.26.4, scipy 1.13.1, sklearn 1.5.2, matplotlib 3.9.4 설치됨
핵심 수치:
| | clothing | eletronics | dblp |
|---|---|---|---|
| N / E / d | 24919 / 91680 / 9034 | 42318 / 43556 / 8669 | 40672 / 288270 / 7202 |
| 클래스 전체/base/valid/test | 77/40/17/20 | 167/90/37/40 | 137/80/27/30 |
| X nnz/행 | 31.74 (§6: 31.4) | 40.26 (§6: 40.7) | 46.57 (§6: 46.3) |
| 최소 클래스 / Σp² | 101 / 0.0202 | 101 / 0.0081 | 101 / 0.0100 |
| way 오염 5/10/20 | .078/.168/.321 | .032/.071/.144 | .039/.087/.174 |
| homophily (사전값 / NaQ T7) | 0.676 (0.676 / 0.62) | 0.373 (0.373 / 0.38) | 0.258 (0.258 / 0.29) |
| 평균 degree / deg0 | 7.358 / 0 | 2.059 / 0 | 14.175 / 0 |
| 검사 1–4 | PASS | PASS | PASS |
- valid 클래스 파일별 개수: 모든 pkl에서 17/37/27. 예외는 clothing 1way_1shot.pkl로 16이며, 합집합은 17.
- GCN (a): eletronics step 26.5ms, fwd 18.5ms, 피크 1.21GB, 16k 추정 7.1분 / dblp 30.6ms, 21.4ms, 1.18GB, 8.2분 (test LR probing 시간 제외)
- 풀 블록 (b), B=2048 (eletronics / dblp):
  - cos_raw: 블록 0.053s ×21 ≈ 1.1s, 피크 2.78GB / 0.045s ×20 ≈ 0.9s, 피크 2.25GB
  - PPR(51 term): 블록 0.60s ×21 ≈ 12.6s, 피크 1.67GB / 0.64s ×20 ≈ 12.9s, 피크 1.62GB
  - X_sgc: 0.41s, 텐서 1.37GB, 피크 5.52GB / 0.06s, 1.09GB, 피크 4.43GB
계획 대비 차이·임의 결정:
- X nnz/행이 §6 값과 약 1% 다르다 (검사 1 대상 아님). explicit zero는 0개이며 원인은 미확인이다. 이 값으로 split이나 수치를 바꾸지 않았다.
- 시간 측정 에피소드는 노드 55개를 무작위로 뽑아 5개는 support, 50개는 query로 썼다. 손실은 -‖q−p‖² logits에 CE를 적용했다.
- forward ms는 20회 평균이다. cos 블록 topk는 self를 포함한 51개다.
- X_sgc 피크(4.4–5.5GB)는 sparse(Â·X) → dense 변환 후 Â를 곱하는 순서에서 나온 값이다. 단일 프로세스 12GB 목표 안이다.
- OOM 없음. 1024 블록 재시도는 하지 않았다.
문제·미해결:
- 없음
변경·생성 파일:
- .gitignore, cape/__init__.py, cape/data.py, cape/check_data.py, cape/smoke_gpu.py
- results/T01_data_stats.json, results/T01_timing.json, reports/T01_report.md (+ CLAUDE.md, instructions/)
커밋: dd35aa1 (branch cape-validation)
