# Project Notes

## Why I built this

국립국어원/말평 확신성 추론 과제는 문장 쌍을 보고 확신성 점수를 `1~7` 사이 실수로 예측하는 문제였다. 단순 분류가 아니라 회귀 문제에 가깝고, 공식 평가는 MSE 기반이라 작은 예측 오차도 점수에 바로 영향을 줬다.

이 프로젝트는 하나의 모델을 크게 키우기보다, 여러 seed와 모델 후보를 만들고 dev 점수 기준으로 앙상블과 보정을 반복해보는 방식으로 접근했다.

## What was difficult

가장 어려웠던 부분은 리더보드 점수를 올리는 것과 과적합을 피하는 것 사이의 균형이었다. dev 점수만 보고 계속 조합을 바꾸면 쉽게 dev에만 맞는 앙상블이 될 수 있다. 특히 제출 파일을 여러 번 만들다 보면 어떤 조합이 진짜 안정적인지, 우연히 좋아진 것인지 헷갈릴 수 있었다.

또 하나 어려웠던 점은 출력 범위였다. 모델은 회귀값을 자유롭게 낼 수 있지만, 공식 정답 범위는 `1~7`이었다. 예측값이 범위를 벗어나면 후처리가 필요했고, calibration이 점수를 올릴 수도 있지만 반대로 망칠 수도 있었다.

## Issues I ran into

### 1. 단일 seed 결과가 안정적이지 않았다

Transformer 회귀 모델은 seed에 따라 dev MSE가 달라졌다. 하나의 seed만 보고 제출 파일을 만들면 안정성이 부족했다. 그래서 여러 seed를 학습하고, greedy ensemble로 조합을 선택했다.

### 2. 모델 다양성이 부족했다

비슷한 모델만 앙상블하면 예측이 크게 달라지지 않았다. KLUE-RoBERTa-large 기반 후보뿐 아니라 XLM-RoBERTa-large-XNLI 후보를 추가해 앙상블 다양성을 확보하려고 했다.

### 3. calibration이 항상 좋아지는 것은 아니었다

Polynomial calibration을 적용하면 dev 기준 점수가 좋아질 수 있었지만, 너무 강하게 맞추면 과적합 위험이 있었다. 그래서 제출 후보별로 dev 점수와 예측 분포를 같이 확인했다.

### 4. 로컬 검증 answer 파일 관리가 조심스러웠다

README에도 적어둔 것처럼 `test-answer` 파일은 로컬 검증용이었다. 이 파일을 공식 제출 모델 선정에 직접적으로 과적합시키면 결과 해석이 위험해질 수 있었다. 그래서 데이터 위치와 용도를 README에 명시했다.

## How I fixed them

- 여러 seed의 Transformer regressor를 학습했다.
- KLUE-RoBERTa-large와 XLM-RoBERTa-large-XNLI 계열 후보를 함께 사용했다.
- greedy ensemble로 dev MSE가 좋아지는 후보만 단계적으로 추가했다.
- polynomial calibration을 적용하되, 예측값은 `1~7` 범위로 clipping했다.
- `scripts/score_submission.py`로 제출 파일 형식과 점수를 확인했다.
- 실험 결과와 제출 후보를 `reports/`, `submissions/`로 분리해 관리했다.

## What I learned

이 프로젝트를 하면서 리더보드 점수는 모델 하나의 성능만으로 결정되지 않는다는 걸 배웠다. seed, 앙상블 조합, calibration, clipping 같은 후처리도 점수에 큰 영향을 줬다.

또 점수를 올리는 과정에서도 검증 기준을 계속 의심해야 한다는 걸 느꼈다. dev 점수만 보고 무한히 조합을 바꾸면 실제 일반화 성능을 설명하기 어려워진다. 그래서 어떤 파일이 어떤 용도인지 문서화하는 것이 중요했다.

## What I would improve next

- seed별 결과와 ensemble 선택 과정을 더 보기 쉬운 표로 정리하고 싶다.
- calibration 전후 예측 분포를 시각화하고 싶다.
- 제출 후보별 차이를 자동으로 비교하는 리포트를 만들고 싶다.
- 모델 카드처럼 데이터 사용 범위와 검증 한계를 더 명확히 남기고 싶다.
