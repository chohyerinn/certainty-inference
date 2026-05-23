#!/usr/bin/env bash
set -uo pipefail

ROOT="$HOME/certainty_inference_20260519"
PY="$HOME/miniconda3/envs/promptscope/bin/python"
GPU=3
LOG_DIR="$ROOT/logs"
LOG="$LOG_DIR/wave9_gpu3_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR" "$ROOT/reports" "$ROOT/submissions" "$ROOT/data/derived"
ln -sfn "$(basename "$LOG")" "$LOG_DIR/wave9_gpu3_latest.log"
cd "$ROOT" || exit 1
exec > >(tee -a "$LOG") 2>&1

echo "=== wave9 GPU3 runner start $(date -Is) ==="
"$PY" scripts/make_train_dev_jsonl.py

score_run() {
  local tag="$1"
  local path="runs/$tag/test_submission.jsonl"
  if [[ -f "$path" && -f data/raw/nikluge-2022-nli-test-answer.jsonl ]]; then
    "$PY" scripts/score_submission.py data/raw/nikluge-2022-nli-test-answer.jsonl "$path" --json | tee "runs/$tag/test_answer_score.json" || true
  fi
}

score_submission() {
  local name="$1"
  local path="$2"
  if [[ -f "$path" && -f data/raw/nikluge-2022-nli-test-answer.jsonl ]]; then
    echo "=== score $name ==="
    "$PY" scripts/score_submission.py data/raw/nikluge-2022-nli-test-answer.jsonl "$path" --json || true
  fi
}

run_final() {
  local tag="$1"
  local model="$2"
  local seed="$3"
  local batch="$4"
  local lr="$5"
  local epochs="$6"
  local precision="$7"
  shift 7

  if [[ -f "runs/$tag/test_submission.jsonl" ]]; then
    echo "=== skip existing $tag ==="
    score_run "$tag"
    return 0
  fi
  echo "=== run final $tag model=$model seed=$seed start $(date -Is) ==="
  local cmd=(env CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/train_transformer_regressor.py
    --train data/derived/nikluge-2022-nli-traindev.jsonl
    --model-name "$model"
    --output-dir "runs/$tag"
    --seed "$seed"
    --batch-size "$batch"
    --lr "$lr"
    --epochs "$epochs"
    --no-dev-eval
    --standardize-target
    "$@")
  if [[ "$precision" == "fp16" ]]; then
    cmd+=(--use-fp16)
  fi
  "${cmd[@]}"
  local status=$?
  echo "=== run final $tag end status=$status $(date -Is) ==="
  score_run "$tag"
  return 0
}

run_final klue-roberta-large-final-traindev-nlipre-seed3407-ep5 klue/roberta-large 3407 16 1e-5 5 fp16 \
  --pooling cls --encoder-init runs/klue-nli-pretrain-roberta-large-seed3407-1ep-lr1e-5/best_encoder.pt
run_final klue-roberta-large-final-traindev-nlipre-mean-seed3407-ep5 klue/roberta-large 3407 16 1e-5 5 fp16 \
  --pooling mean --encoder-init runs/klue-nli-pretrain-roberta-large-seed3407-1ep-lr1e-5/best_encoder.pt
run_final klue-roberta-large-final-traindev-nlipre-seed123-ep5 klue/roberta-large 123 16 8e-6 5 fp16 \
  --pooling cls --encoder-init runs/klue-nli-pretrain-roberta-large-seed123-2ep-lr8e-6/best_encoder.pt
run_final klue-roberta-large-final-traindev-nlipre-mean-seed123-ep5 klue/roberta-large 123 16 8e-6 5 fp16 \
  --pooling mean --encoder-init runs/klue-nli-pretrain-roberta-large-seed123-2ep-lr8e-6/best_encoder.pt
run_final xlm-roberta-large-final-traindev-seed3407-ep5 xlm-roberta-large 3407 8 1e-5 5 fp16 --pooling cls

BASE=submissions/greedy_dev_ensemble_calibrated.jsonl
R1=runs/klue-roberta-large-final-traindev-nlipre-seed3407-ep5/test_submission.jsonl
R2=runs/klue-roberta-large-final-traindev-nlipre-mean-seed3407-ep5/test_submission.jsonl
R3=runs/klue-roberta-large-final-traindev-nlipre-seed123-ep5/test_submission.jsonl
R4=runs/klue-roberta-large-final-traindev-nlipre-mean-seed123-ep5/test_submission.jsonl
R5=runs/xlm-roberta-large-final-traindev-seed3407-ep5/test_submission.jsonl

if [[ -f "$BASE" && -f "$R1" && -f "$R2" ]]; then
  "$PY" scripts/ensemble_submissions.py submissions/final_traindev_blend_klue3407.jsonl "$BASE" "$R1" "$R2" || true
  score_submission final_traindev_blend_klue3407 submissions/final_traindev_blend_klue3407.jsonl
fi
if [[ -f "$BASE" && -f "$R1" && -f "$R2" && -f "$R3" && -f "$R4" ]]; then
  "$PY" scripts/ensemble_submissions.py submissions/final_traindev_blend_klue4.jsonl "$BASE" "$R1" "$R2" "$R3" "$R4" || true
  score_submission final_traindev_blend_klue4 submissions/final_traindev_blend_klue4.jsonl
fi
if [[ -f "$BASE" && -f "$R1" && -f "$R2" && -f "$R3" && -f "$R4" && -f "$R5" ]]; then
  "$PY" scripts/ensemble_submissions.py submissions/final_traindev_blend_all5.jsonl "$BASE" "$R1" "$R2" "$R3" "$R4" "$R5" || true
  score_submission final_traindev_blend_all5 submissions/final_traindev_blend_all5.jsonl
fi

"$PY" scripts/report_experiments.py > reports/experiments_report.json || true
echo "=== wave9 GPU3 runner end $(date -Is) ==="
