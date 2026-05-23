# 확신성 추론

국립국어원/말평 확신성 추론 과제 실험 프로젝트입니다.

입력으로 `context`와 `prompt`가 주어지면, 해당 명제의 확신성 점수를 `1~7` 사이 실수로 예측합니다. 공식 평가는 MSE 기반이며, 리더보드에는 `(-)MSE` 형태로 표시됩니다.

## 현재 최고 제출

- 모델명: `wave8_xlm_roberta_xnli_greedy_calibrated_ensemble`
- 제출 파일: `submissions/malpyeong_certainty_submission_best_wave8.jsonl`
- 공식 리더보드 점수: `-0.0757290`
- 공식 순위: 1위 확인

## 핵심 아이디어

- KLUE-RoBERTa-large 기반 회귀 모델 다중 seed 학습
- XLM-RoBERTa-large-XNLI 후보를 추가해 앙상블 다양성 확보
- dev MSE 기준 greedy ensemble
- dev 기반 polynomial calibration 적용
- 예측값은 공식 범위인 `1~7`로 clipping

## 폴더 구조

- `data/raw/`: 원본 JSONL 데이터
- `data/derived/`: 전처리/파생 데이터
- `docs/`: 과제 기술서와 제출 예시
- `scripts/`: 학습, 앙상블, 보정, 제출 파일 생성 스크립트
- `submissions/`: 제출 후보 JSONL
- `reports/`: 실험/분석 결과
- `runs/`: 일부 로컬 실험 결과

## 주요 스크립트

- `scripts/train_transformer_regressor.py`: transformer 회귀/ordinal class head 학습
- `scripts/greedy_ensemble.py`: dev 기준 greedy ensemble
- `scripts/calibrate_predictions.py`: polynomial calibration
- `scripts/stack_ensemble.py`: weighted/simplex ensemble
- `scripts/score_submission.py`: 로컬 answer 기준 제출 파일 검증/채점
- `scripts/wave*_gpu3_runner.sh`: 서버 GPU3 실험 runner

## 주의

`data/raw/nikluge-2022-nli-test-answer.jsonl`은 로컬 검증용 answer 파일입니다. 공식 제출 모델 선정에 직접 과적합되지 않도록 관리해야 합니다.
