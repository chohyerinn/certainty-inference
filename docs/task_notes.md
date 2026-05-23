# 확신성 추론 작업 노트

## 과제 요약

- 입력: `input.context`, `input.prompt`
- 출력: `output`, 1부터 7 사이 실수
- 평가: `MSE`
- 리더보드 표시 점수: `-100 * MSE`
- 공개된 데이터 규모: 훈련 1,448 / 검증 189 / 시험 180

## 다운로드 페이지 항목

| 구분 | ID | 이름 | 현재 상태 |
| --- | ---: | --- | --- |
| 말뭉치 | 75 | 2023 인공 지능의 언어 능력 평가: NLI | 로그인 후 말뭉치 신청 필요 |
| 답안 제출 양식 | 76 | 답안 제출 양식 | 보관 완료 |
| 정답지 | 216 | 정답지 | 보관 완료 |
| 기타 | 292 | 과제 기술서 | 보관 완료 |

## 현재 로컬 파일

- `data/raw/nikluge-2022-nli-train.jsonl`: 1,448개, 정답 포함
- `data/raw/nikluge-2022-nli-dev.jsonl`: 189개, 정답 포함
- `data/raw/nikluge-2022-nli-test.jsonl`: 180개, 정답 없음
- `data/raw/nikluge-2022-nli-test-answer.jsonl`: 180개 정답 포함 JSONL
- `data/derived/nikluge-2022-nli-test-input.jsonl`: 정답 제거 후 만든 입력 전용 JSONL
- `submissions/constant_6.0.jsonl`: 파이프라인 검증용 상수 제출

공식 `test.jsonl`은 `test-answer.jsonl`에서 `output`만 제거한 입력과 동일함을 확인했다.

## 통계

| split | rows | output mean | output median | context chars mean | prompt chars mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 1,448 | 5.196 | 6.05 | 154.1 | 30.8 |
| dev | 189 | 5.226 | 6.125 | 152.3 | 30.3 |
| test | 180 | - | - | 163.8 | 32.5 |

## 주의

정답 포함 파일은 로컬 평가와 제출 포맷 확인에만 사용한다. 실제 경쟁 제출 모델의 학습 데이터에는 섞지 않는다.
