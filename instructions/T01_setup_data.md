# T01 — 환경·데이터 기반 구축과 시간 측정

- 선행 문서: `CLAUDE.md` (특히 §3, §5, §6, §10, §12)
- 시간 예산: **15분** (구현 약 8분, 실행 약 3분, 보고 약 4분)
- 이번 지시서의 범위: 브랜치·디렉터리 준비, 데이터 로더, split과 고정 태스크 캐시, 데이터 무결성 검사, GPU 시간 측정
- 범위 밖: 후보 풀, 단서, scorer, 에피소드 생성기, 학습 (T02 이후)

## 1. 저장소 준비

1. `.gitignore`를 만들고 다음을 넣는다: `cache/`, `logs/`, `__pycache__/`, `*.pyc`
2. 디렉터리를 만든다: `cape/`, `cache/`, `results/`, `logs/`, `reports/`
   - `cape/__init__.py`는 빈 파일로 만든다.
3. 사용자가 넣어 둔 `CLAUDE.md`와 `instructions/`는 수정하지 않는다. 커밋에는 포함한다.

## 2. 환경 확인

`taskns` 환경에서 다음을 확인하고 결과를 보고서에 적는다.
- python, torch, CUDA 사용 가능 여부, GPU 이름
- numpy, scipy, scikit-learn 버전
- matplotlib 설치 여부 (설치하지 말고 여부만 적는다)

CUDA를 쓸 수 없으면 중단한다.

## 3. `cape/data.py` 구현

아래 함수만 만든다. 단순하게 유지한다.

**`load_graph(ds, root="data") -> dict`**
- 반환 키
  - `A`: scipy CSR, float32. 대칭 이진 행렬이며 중복 edge와 self-loop를 제거한다.
  - `X`: scipy CSR, float32, [N, d]. `train.mat`과 `test.mat`의 `Attributes`를 각 `Index` 위치에 배치한다 (NaQ `data.py`와 같은 방식).
  - `y`: np.int64 [N]
  - `train_classes`, `test_classes`: 정렬된 list
- N = network 파일의 최대 노드 id + 1
- 모든 노드가 feature와 레이블을 하나씩 받았는지 확인한다.

**`norm_adj(A) -> scipy CSR`**
- Â = D^−1/2 (A + I) D^−1/2

**`to_torch_sparse(mat, device)`**
- scipy sparse를 torch sparse float32 tensor로 바꾼다.

**`load_split(ds) -> dict`**
- `cache/{ds}/split.json`이 있으면 읽고, 없으면 만든다.
- valid 클래스 = `save/downstream_tasks/{ds}/*.pkl` **모든 파일**의 valid pool에 등장하는 노드들의 실제 레이블 합집합이다.
  - 파일별로 찾은 valid 클래스 수도 함께 기록한다.
- base 클래스 = train 클래스 − valid 클래스. test 클래스 = `test_classes`.
- 저장 형식: `{"base": [...], "valid": [...], "test": [...], "valid_per_file": {"5way_1shot.pkl": k, ...}, "source": "union of NaQ save/downstream_tasks/{ds}/*.pkl valid pools"}`

**`load_tasks(ds) -> dict of np.ndarray`**
- `cache/{ds}/tasks.npz`가 있으면 읽고, 없으면 만든다.
- 변환 대상
  - `5way_1shot.pkl`의 valid pool → `valid_spt [50,5]`, `valid_yspt`, `valid_qry [50,40]`, `valid_yqry`
  - `5way_1shot.pkl`의 test pool → `test1_spt [500,5]`, `test1_yspt`, `test1_qry [500,40]`, `test1_yqry`
  - `5way_5shot.pkl`의 test pool → `test5_spt [500,25]`, `test5_yspt`, `test5_qry [500,40]`, `test5_yqry`
- 모든 배열은 int64다. NaQ의 y 값은 순서 가정 없이 그대로 저장한다.
- pickle 파일은 `(valid_pool, test_pool)` 튜플이다. 각 원소는 `(id_spt: np.ndarray, y_spt: LongTensor, id_query: np.ndarray, y_query: LongTensor)`이다.
  - torch를 import한 상태에서 `pickle.load`로 읽는다.
  - 읽기에 실패하면 태스크를 새로 만들지 말고 중단한다.

## 4. `cape/check_data.py` — 무결성 검사와 통계

`python -m cape.check_data`로 세 데이터셋(`Amazon_clothing`, `Amazon_eletronics`, `dblp`)을 모두 처리한다.

**통계** → `results/T01_data_stats.json`과 콘솔 표
- N, 무방향 edge 수, d, X의 행당 nnz
- 클래스 수 (전체 / base / valid / test)
- 전 노드 레이블 여부, 레이블 연속성, 최소 클래스 크기
- Σp²와 way 오염률 (N = 5, 10, 20)
- edge homophily (같은 레이블을 잇는 edge 비율)
- 평균 degree, degree 0인 노드 수

**검사** (하나라도 실패하면 중단하고, 무엇이 어떻게 다른지 보고한다. split이나 수치를 맞추려고 고치지 않는다)
1. N, 무방향 edge 수, d, 클래스 수(전체/base/valid/test)가 `CLAUDE.md` §6 표와 일치한다.
2. base ∩ valid = ∅, base ∪ valid = train 클래스, test ∩ train = ∅
3. 모든 태스크에 대해:
   - 배열 shape이 위 명세와 같다.
   - 태스크 안에서 support와 query가 겹치지 않는다.
   - 같은 y ↔ 같은 실제 레이블이고, 한 태스크에 실제 레이블이 정확히 5종이다.
   - valid 태스크의 클래스 ⊂ valid 클래스, test 태스크의 클래스 ⊂ test 클래스
4. 각 pkl 파일의 valid 클래스 ⊆ `split.json`의 valid 클래스
   - 합집합이므로 항상 성립해야 한다. 이 검사는 코드 오류를 잡기 위한 것이다.

**사전 계산값과 비교** (게이트 아님, 크게 다르면 원인을 보고)
- edge homophily: 0.676 / 0.373 / 0.258
- 평균 degree: 7.36 / 2.06 / 14.18
- degree 0인 노드: 0개
- NaQ Table 7의 homophily(0.62 / 0.38 / 0.29)는 정의가 다른 것으로 보이니 참고로만 나란히 적는다.

## 5. `cape/smoke_gpu.py` — 시간 측정

`python -m cape.smoke_gpu`로 `Amazon_eletronics`와 `dblp`를 처리하고, 결과를 `results/T01_timing.json`에 저장한다.
모든 시간은 `torch.cuda.synchronize()`를 사용해 측정한다.

**(a) GCN 학습 step**
- 설정: X(sparse)와 Â(sparse)를 GPU에 올린다. 2-layer GCN(d→512→256, ReLU, xavier), Adam lr 1e-4.
- 한 step의 구성
  - 전 그래프 forward
  - 임의의 5-way 1-shot, 10-query 에피소드에 대한 ProtoNet 손실 (squared Euclidean)
  - backward, step
- warmup 5회 후 50 step 평균 ms/step을 잰다.
- 함께 기록할 것
  - `torch.cuda.max_memory_allocated`
  - forward만(eval, no_grad) 했을 때의 ms
- 추정치 보고
  - 16,000 에피소드 학습 시간 = step 시간 × 16,000 + forward 시간 × 40 (검증 횟수)
  - 이 값에 test LR probing 시간은 포함되지 않는다는 점을 명시한다.

**(b) 후보 풀 계산 블록** (T02 대비)
- `cos_raw` 블록
  - 행 정규화한 dense X(float32, GPU)에서 [N, d] @ [d, 2048] → `topk(51)`
  - 블록 1개 시간을 재고, 전체 블록 수(⌈N/2048⌉)를 곱한 추정치를 보고한다.
- PPR 블록
  - [N, 2048] dense 행렬에 Â를 51번 곱하는 power iteration → `topk(51)`
  - 블록 1개 시간과 전체 추정치를 보고한다.
- 두 측정 모두 피크 GPU 메모리를 기록한다.
- `X_sgc`(= Â² X, dense) 생성 시간과 메모리도 1회 측정한다.

## 6. 완료 기준

- [ ] `cape-validation` 브랜치, `.gitignore`, 디렉터리 생성
- [ ] `cape/data.py`, `cape/check_data.py`, `cape/smoke_gpu.py` 작성
- [ ] 세 데이터셋 모두 `cache/{ds}/split.json`, `cache/{ds}/tasks.npz` 생성
- [ ] §4의 검사 1–3 모두 통과
- [ ] `results/T01_data_stats.json`, `results/T01_timing.json` 생성
- [ ] `reports/T01_report.md` 작성, 커밋 (`T01: setup, data loader, task cache, timing`)

## 7. 중단 조건

- `taskns`에서 CUDA를 쓸 수 없다.
- pickle을 읽지 못한다.
- valid 클래스 합집합의 크기가 17/37/27과 다르다. 이 경우 파일별 클래스 수를 보고한다.
- §4의 검사 1–3 중 하나라도 실패한다.
- 단일 측정(블록 1개 등)에서 GPU OOM이 발생한다.
  - 이 경우 블록 크기를 1024로 한 번만 줄여 다시 시도한다. 그래도 실패하면 중단한다.

## 8. 보고에 반드시 포함할 항목

- 환경 버전 (§2)
- 데이터셋 3개의 통계 표 (homophily와 NaQ 값 비교 포함), 검사 통과 여부
- (a)의 ms/step, forward ms, 피크 메모리, 16,000 에피소드 추정 시간 (데이터셋별)
- (b)의 블록당 시간, 전체 추정 시간, 피크 메모리, `X_sgc` 생성 시간과 메모리
- 계획 대비 차이와 임의로 내린 결정
