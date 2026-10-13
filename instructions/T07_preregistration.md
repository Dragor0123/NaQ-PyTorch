# T07 사전 등록 (실행 전 커밋, 결과 확인 후 수정 금지, 2026-10-14)

- 판정은 T07 보고 수신 후 사람이 한다. Agent는 계산값만 적는다.
- 근거: T06 결과(elec nov +1.34, dblp +0.17; AUROC cent_sgc elec .713 / dblp .570, cent_raw elec .784), NaQ 논문 Table 15(clothing ProtoNet Feat 83.77 / Diff 78.64), CLAUDE.md §6 데이터 통계
- 정확도는 1-shot LR(%), 차이는 "arm − NaQ-best", seed 평균

| ID | 예측 | 적중 기준 | 확신도 | 근거 |
|---|---|---|---|---|
| S1 | PU 추정기가 centroid보다 낫다 | r=0.5에서 `pu_both` AUROC − `cent_sgc` AUROC ≥ +0.05인 데이터셋이 3개 중 2개 이상 | 0.60 | PU는 U의 분포를 직접 쓰고, 구조 채널이 추가됨 |
| S2 | dblp도 어느 정도 가를 수 있게 된다 | E\*의 dblp AUROC(r=0.5) ≥ 0.65 | 0.40 | dblp는 homophily .26, feature 채널이 약함. 구조 채널에 기대 |
| S3 | 비례 샘플링이 하드 컷과 같거나 낫다 | elec `nov2_t1_r0.5` − NaQ ≥ +1.3 | 0.50 | T06 하드 컷 +1.34. 보정 확률 가중이 novel 비율을 비슷하게 유지하면서 커버리지를 지킴 |
| S4 | 더 강한 집중이 더 낫다 | elec `nov2_t2` − `nov2_t1` > 0 (seed 평균) | 0.55 | 용량-반응이 단조였음. 단, 다양성 감소로 상쇄될 수 있음 |
| S5 | dblp 이득은 AUROC에 달려 있다 | S2가 성립하면 dblp `nov2_t1` − NaQ ≥ +0.5; 성립하지 않으면 \|차이\| < 0.7 | 0.55 | T06: AUROC .57에서 nov ≈ unl_uni |
| S6 | 혼합은 해롭지 않고 약간 돕는다 | 두 데이터셋 모두 `mix_alt` − NaQ ≥ 0, 그리고 elec `mix_alt` − `nov2_t1` ∈ [−0.5, +1.5] | 0.50 | oracle_base 단독은 −7이었으나 30% 혼합은 base 영역 보강으로 작용할 수 있음 |
| S7 | 표준 설정에서 추정기는 상한과 같아진다 | elec `nov2_t1_r1.0` − NaQ 와 `naq_unl` − NaQ의 차이 ≤ 0.7 | 0.70 | r=1.0에서 PU 분류기는 거의 완벽 |
| S8 | 레이블 20%에서도 이득이 남는다 | elec `nov2_t1_r0.2` − NaQ ≥ +0.7 (seed 2개 평균) | 0.55 | T06 r=0.2 하드 컷 +0.99 (seed 1개) |
| S9 | Amazon_clothing에서도 효과가 있다 | clothing `nov2_t1_r0.5` − `naq_feat` ≥ +1.0 (seed 2개 평균, 모두 양수) | 0.50 | homophily .68로 영역 구분이 쉬울 것. 단, 5-way 오염률이 높은 데이터셋 |
| S10 | 영역 노출이 test 영역 임베딩을 바꾼다 | elec에서 NMI_test: NaQ < `nov2_t1` < `naq_unl` (단조), 그리고 NMI_base의 변화량 절댓값 < NMI_test 변화량 | 0.60 | 메커니즘 가설. 정확도 외 독립 증거 |

## 게이트 G7 (판정 기준, 사전 명시)
- G7-a: E\*의 dblp AUROC(r=0.5) ≥ 0.65
- G7-b: `nov2_t1_r0.5` − NaQ-best ≥ +1.0 이고 seed 3개 모두 양수인 데이터셋이 3개 중 2개 이상 (clothing은 seed 2개 기준)
- 해석 규칙
  - G7-a·G7-b 모두 통과 → 방법을 일반 주장으로 진행
  - G7-b만 통과 → "이득은 영역 식별 가능성에 비례한다"는 조건부 주장으로 진행. AUROC–이득 그림이 핵심 결과
  - 둘 다 실패 → 방향 재검토

## 해석상 주의
- `mix_alt`는 L의 레이블을 지도 에피소드에 쓴다. NaQ(레이블 없음)와의 비교에서는 정보량 차이를 표시한다.
- r=1.0 결과는 "레이블 없음 = novel" 구조 때문에 추정이 자명해진다. 방법의 증거는 r ≤ 0.5에서 얻은 것만 쓴다.
- P3 재실행은 비결정성 때문에 T03 값과 ±0.3%p 정도 다를 수 있다. 메커니즘 분석에만 쓴다.
