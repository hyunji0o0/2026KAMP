# KAMP CN7·RG3 사출성형 품질 예측 및 검사 우선순위

## 바로 실행하기

이 ZIP을 풀고 `KAMP_CN7_RG3_Submission/` 폴더에서 다음을 실행합니다.

```bash
python -m pip install -r requirements.txt
python predict_test_data.py
python verify_results.py
```

`test_prediction/` 안의 제품별 CSV 2개와 통합 CSV 1개가 재생성됩니다. 각 열과 판정 기준은 `docs/RESULTS_GUIDE.md`에 설명했습니다.

## 제출 파일

| 위치 | 역할 |
| --- | --- |
| `test_prediction/` | CN7·RG3 제품별 예측 CSV와 두 제품을 섞은 통합 CSV. 각 제품의 내부 순서 보존 |
| `test_data/` | 대회가 제공한 CN7·RG3 공식 무라벨 테스트 CSV. 재현 실행에 사용 |
| `training_data/` | 대회가 제공한 CN7·RG3 labeled 학습 CSV |
| `model/` | 제품 분기·CN7·RG3 모델과 F1.5 임계값을 담은 단일 joblib |
| `predict_test_data.py`, `source_code/f15_model.py` | 자동 제품 분기·제품별 점수·검사순위·0/1 판정·CSV 저장 |
| `verify_results.py` | 입력 피처와 결과, 참고 모델 순위, 0/1 임계값, 통합 파일 일치 검증 |
| `train_model.py`, `rebuild_f15_bundle.py` | 학습용 데이터에서 모델을 다시 학습하고 기존 정책을 연결하는 재현 코드 |
| `requirements.txt` | 실행에 필요한 Python 패키지 버전 |
| `docs/RESULTS_GUIDE.md` | 결과 열, 제품별 수량, F1.5 기준과 검증 한계 설명 |

## AI 모델의 전체 구조

```text
CN7·RG3 혼합 입력 → 공통 공정센서 23개 → ExtraTrees23 제품 분류
                                          ├─ CN7 → XGB + LOG·TEAM Frozen Quota
                                          └─ RG3 → 가중 1-NN + SoftPair
                             → 제품별 검사순위 → 상위 5·10·20% 검사목록
                                              → F1.5 기준 0/1 불량 의심 표시
                             → CN7·RG3 제품별 CSV와 혼합 순서의 통합 CSV
```

두 품질모델의 원점수는 서로 다른 척도이므로 제품 간 위험도를 직접 비교하지 않습니다. 검사순위와 0/1 의심 판정은 각각 별도의 기준으로 산출됩니다. 최종 품질검사 전 **검사 우선순위 결정과 불량 의심 표시**가 목적이며, 출력만으로 합격·불합격을 확정하지 않습니다.

### 1. 제품 분류: ExtraTrees23

공통인 23개 수치형 공정센서만 입력하여 160개 ExtraTrees의 판단을 종합합니다. RG3 분류 점수가 0.5 이상이면 RG3, 그렇지 않으면 CN7 품질모델로 보냅니다. 선택된 제품 점수가 0.8 미만이거나 센서 하나라도 분류기 학습 범위 밖이면 `Product_Review`로 표시하며, 공식 출력 CSV에서는 해당 행을 `REVIEW`로 남기고 품질 결과를 확정하지 않습니다. 분류 점수는 제품 판별용 점수이며 불량확률이 아닙니다.

파일명, `Record_ID`, `PassOrFail`은 **추론 시 제품 분류 피처가 아닙니다**. 분류기 학습에서는 CN7·RG3로 제공된 네 파일의 출처를 제품 정답으로 사용했고, 품질 정답은 사용하지 않았습니다. 공식 테스트의 파일 출처는 모델 입력을 섞은 후 *검증과 제품별 결과 복원*에만 사용합니다. 따라서 제공된 공식 테스트의 분기 일치율은 처음 보는 현장의 독립 정확도가 아닙니다.

### 2. CN7 품질모델: XGB 중심 Frozen Quota

23개 센서에 압력(`Max_Switch_Over_Pressure`), 속도(`Max_Injection_Speed`), 시간(`Filling_Time`)의 곱으로 만든 상호작용 피처를 더합니다. **XGB_SVF_SSL 5개 시드 평균**이 기본 위험순위를 만들고, **LOG_SV_SSL**과 **TEAM_LX**가 XGB 순위에서 뒤에 있지만 보조모델에서는 높게 평가한 후보를 찾습니다. XGB·LOG 학습에는 정답 없는 CN7 데이터에서 선별한 일부 의사라벨도 활용합니다.

Frozen Quota는 검사목록을 상위 5% → 10% → 20%로 넓힐 때, *새로 생긴 자리*의 최대 20% → 30% → 20%만 LOG·TEAM 후보에 허용합니다. 나머지는 XGB 순위로 채우며 이전 단계에서 뽑은 행은 유지합니다. 이 숫자는 불량확률이나 검사대상 비율이 아니라 **보조모델 개입량의 상한**입니다. 최종 순위는 XGB 원점수만 정렬한 결과와 다를 수 있습니다.

### 3. RG3 품질모델: 가중 1-NN + SoftPair

학습된 정상·불량 조건의 센서 차이를 반영한 가중 거리로, 새 기록에서 **기존 불량 공정조건까지의 최근접 거리**를 계산합니다. 거리가 가까울수록 기본 위험점수가 높습니다. 센서 23개가 모두 같고 `Record_ID`가 연속인 두 기록이 **정확히 한 쌍**일 때에만, 학습에서 정한 기록 위치 방향에 따라 SoftPair 보정(계수 0.125)을 더하거나 뺍니다. 조건이 맞지 않는 행에는 이 보정이 없습니다.

`Record_ID`는 **RG3의 Pair 보조신호에만** 쓰이며 제품 자동분류에는 사용되지 않습니다. 기록 순서가 금형 위치나 실제 불량 원인과 연결된다는 물리적 의미는 확인되지 않았습니다. Pair 신호는 두 기록이 함께 입력된 배치에서만 계산할 수 있습니다.

### 4. 순위, 검사목록, 0/1 의심 판정

제품별 최종 모델 점수를 정렬해 `Inspection_Rank` 1~N을 매깁니다. 각 제품 배치 크기의 상위 5·10·20%(`floor(N×비율)`)를 중첩된 검사목록으로 표시합니다. 이 순위를 0~1로 바꾼 `Priority_Score`가 제품별 F1.5 임계값 이상이면 `Predicted_PassOrFail=1`(불량 의심), 미만이면 `0`(정상 예측)입니다. **Top-k 검사목록과 0/1 판정은 서로 다른 결정**이며, 둘 다 실제 불량확률이나 최종 품질검사 결과가 아닙니다.

### 5. 선택 근거와 성능의 범위

동일 공정조건이 학습·검증 양쪽에 놓이지 않도록 나눈 labeled OOF에서, CN7 최종 정책은 1,211행(불량 17건) 기준 AP 0.5055·Top 10% 16/17건으로 Logistic 베이스라인(AP 0.4837·14/17건)과 XGB 단독(AP 0.5005·15/17건)보다 높았습니다. RG3는 1,182행(불량 25건)의 5-fold×5회 그룹 분리 OOF를 행별 평균하여 평가했으며, 일반 1-NN AP 0.0271·Top 20% 5/25건, 1-NN+Pair AP 0.0515·10/25건, 최종 가중 1-NN+Pair AP 0.0552·10/25건입니다. **RG3의 검사량별 포착 개선은 주로 Pair 신호에서 나왔고, 센서 가중치의 추가 이득은 AP에 한정적**입니다.

이 수치는 소수의 동일 개발 라벨로 모델·정책을 선택한 OOF 근거입니다. 공식 테스트는 무라벨이므로 실제 Precision·Recall·F1이나 불량 포착률을 계산할 수 없습니다. 특히 RG3는 동일 센서의 정상·불량이 중첩되고, 새로운 배치에서 Pair 적용률이 달라질 수 있어 새 생산기간의 품질 정답으로 재평가해야 합니다.

## 결과 CSV의 열

원본 25열 중 **세 모델이 실제 사용하는 23개 센서 변수와 기록 ID**만 남겼습니다. 사용하지 않는 `Clamp_Open_Position`을 제외합니다. 센서 피처명은 원본의 **영어 이름 그대로** 유지합니다(예: `Injection_Time`, `Max_Injection_Pressure`, `Mold_Temperature_4`). `Unnamed: 0` 원본 기록 열만 보기 쉬운 `Record_ID`로 표시합니다. 공정 수치와 제품별 행 순서는 바꾸지 않았습니다. 모델의 추가 결과 7열은 아래의 영어 이름을 사용합니다.

| 결과 열 | 해석 |
| --- | --- |
| `Predicted_Product` | 센서값으로 자동 분기한 CN7 또는 RG3. 파일명·제품명은 **모델 입력에 사용하지 않음** |
| `Predicted_PassOrFail` | 동봉된 모델의 제품별 F1.5 임계값을 적용한 평가용 이진 예측. **0=정상 예측, 1=불량 의심** |
| `Predicted_Quality` | `PASS_PREDICTED`=정상 예측, `FAIL_SUSPECT`=불량 의심 |
| `Inspection_Rank` | 제품 내 모델 위험순위 1~N. 1이 가장 먼저 확인할 제품 |
| `Inspection_Tier` | `TOP_5_PCT`=상위 5%, `TOP_5_TO_10_PCT`=5~10%, `TOP_10_TO_20_PCT`=10~20%, `AFTER_TOP_20_PCT`=20% 이후. RG3에서 20% 이후에도 불량 의심이면 `AFTER_TOP_20_PCT_FAIL_SUSPECT` |
| `Model_Score` | CN7 XGB Core 또는 RG3 WeightedSoftPair가 만든 원점수. 제품 간 직접 비교 금지, 불량확률 아님 |
| `Priority_Score` | 모델 순위를 배치 내 0~1로 환산한 점수. 이 값에 F1.5 임계값을 적용하며 **불량확률은 아님** |

`Predicted_PassOrFail`과 `Inspection_Tier`는 **서로 다른 결정**입니다. 품질 이진판정은 동봉 F1.5 임계값을 사용하고, 검사 구간은 Top 5/10/20% 역량을 가정합니다. RG3에서는 임계값에 따른 불량 의심이 Top 20%를 넘어가므로, 해당 행을 숨기거나 정상으로 바꾸지 않고 `AFTER_TOP_20_PCT_FAIL_SUSPECT`로 표시했습니다.

| 제품 | F1.5 임계값 | 공식 테스트의 0 | 공식 테스트의 1 | 이진 불량 의심 비율 |
| --- | ---: | ---: | ---: | ---: |
| CN7 | 0.9851239669421488 | 34,714 | 525 | 1.49% |
| RG3 | 0.7552921253175275 | 27,146 | 8,795 | 24.47% |

이 임계값은 **동봉 ZIP의 그룹 분리 OOF F1.5(β=1.5) 최대 기준**을 그대로 사용합니다. CN7 labeled 검증은 TP 9/17, Precision 47.37%, Recall 52.94%, F1 0.5000입니다. RG3 반복 그룹 검증 기준은 TP 14/25, Precision 4.83%, Recall 56.00%, F1 0.0889입니다. 같은 개발 라벨을 임계값 선택에도 사용했으므로 독립 외부 평가로 해석하면 안 됩니다.

동봉된 모델의 공식 테스트 검사순위·0/1 예측을 재현하도록 XGBoost를 `3.1.3`으로 고정했습니다. `2.1.4`로는 CN7 순위가 달라집니다. `verify_results.py`는 공식 테스트 파일에서 고정된 기준 순위 해시까지 대조합니다.

판정점수는 `1-(검사순위-1)/(제품별 배치크기-1)`입니다. 따라서 **0/1의 전체 개수는 제품별 배치크기와 고정 threshold에 의해 거의 결정되고, 어떤 행이 1인지는 모델의 순위가 결정**합니다. 예측 1의 비율을 테스트의 실제 불량률이나 불량확률로 읽지 마세요.

**중요:** 공식 unlabeled 테스트에는 실제 `PassOrFail` 정답이 없습니다. 위 525건·8,795건의 실제 불량률, 테스트 Precision/Recall/F1은 알 수 없습니다. CN7의 학습·테스트 수치 영역 차이와 RG3의 쌍 적용률 차이도 남아 있습니다. 0을 현장 양품 확정으로, 1을 불량 확정으로 취급하거나 품질검사를 생략해서는 안 됩니다.

## Anaconda / Jupyter 실행

ZIP을 풀고 Anaconda Prompt에서 `KAMP_CN7_RG3_Submission` 폴더로 이동합니다.

```bash
conda create -n kamp_f15 python=3.12 -y
conda activate kamp_f15
python -m pip install -r requirements.txt
python -m pip install notebook
jupyter notebook
```

권장 환경은 Python 3.12, scikit-learn 1.8.0, XGBoost 3.1.3입니다. 의존성을 바꿔 실행할 경우 `verify_results.py`의 참고 결과 대조가 실패할 수 있습니다.

이 폴더에 노트북을 만들어 다음을 순서대로 실행합니다.

```python
%run predict_test_data.py
%run verify_results.py
```

터미널에서는 `python predict_test_data.py`와 `python verify_results.py`를 실행합니다. 다른 폴더로 재생성하려면 `%run predict_test_data.py --output-dir "test_prediction/reproduced"`를 사용하세요. 검증 스크립트는 기본 `test_prediction/` 경로를 확인합니다.

프로그램은 내부적으로 CN7·RG3 데이터를 35~180행 크기의 다양한 묶음으로 섞어 모델에 한 번 전달합니다. 제품별 입력 순서는 보존합니다. **입력 DataFrame에 `Product_Type` 열을 넣지 않으므로**, joblib의 센서 기반 ExtraTrees 제품 분기가 실행됩니다. 출처는 모델 실행 후 제품별 파일에 원래 행을 배치하고 분기를 대조하는 데만 씁니다.

제품 분기 모델은 이 공식 두 테스트 파일의 **출처 정보도 학습**했으므로, 이번 파일의 분기 일치 100%를 처음 보는 생산 데이터의 정확도로 주장하지 않습니다.

## 재학습의 범위

```python
%run train_model.py
%run rebuild_f15_bundle.py
%run predict_test_data.py --model "model/CN7_RG3_retrained_f15.joblib" --output-dir "test_prediction/retrained"
```

`train_model.py`는 동봉 학습 데이터와 CN7 무라벨 pseudo-label pool을 사용합니다. 새로 학습한 모델에 **기존** F1.5 threshold를 연결하면 사용법을 재현할 수 있지만, 재학습 모델에 맞는 임계값을 재검증한 것은 아닙니다. 제출 결과의 정확한 재현은 기본 동봉 모델로 수행하세요.
