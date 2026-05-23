#!/usr/bin/env bash
set -uo pipefail

ROOT="$HOME/certainty_inference_20260519"
PY="$HOME/miniconda3/envs/promptscope/bin/python"
GPU=3
LOG_DIR="$ROOT/logs"
LOG="$LOG_DIR/wave8_gpu3_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR" "$ROOT/reports" "$ROOT/submissions"
ln -sfn "$(basename "$LOG")" "$LOG_DIR/wave8_gpu3_latest.log"
cd "$ROOT" || exit 1
exec > >(tee -a "$LOG") 2>&1

echo "=== wave8 GPU3 runner start $(date -Is) ==="

refresh_ensembles() {
  "$PY" scripts/greedy_ensemble.py --steps 1500 \
    --exclude-prefix tfidf-knn- \
    --exclude-prefix feature-reg- \
    --output-jsonl submissions/greedy_dev_ensemble_all.jsonl \
    --dev-output-jsonl submissions/greedy_dev_ensemble_all_dev.jsonl \
    --report-json reports/greedy_dev_ensemble_all_report.json || true
  "$PY" scripts/calibrate_predictions.py \
    --dev-pred submissions/greedy_dev_ensemble_all_dev.jsonl \
    --test-pred submissions/greedy_dev_ensemble_all.jsonl \
    --output-jsonl submissions/greedy_dev_ensemble_calibrated.jsonl \
    --report-json reports/greedy_dev_ensemble_calibrated_report.json || true
  "$PY" scripts/report_experiments.py > reports/experiments_report.json || true
  grep -E 'candidate_count|best_step|final_dev_mse|final_test_answer_mse' reports/greedy_dev_ensemble_all_report.json 2>/dev/null || true
  grep -E '"name"|"dev_mse"|"test_answer_mse"' reports/greedy_dev_ensemble_calibrated_report.json 2>/dev/null | head -n 18 || true
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
}

score_run() {
  local tag="$1"
  local path="runs/$tag/test_submission.jsonl"
  if [[ -f "$path" && -f data/raw/nikluge-2022-nli-test-answer.jsonl ]]; then
    "$PY" scripts/score_submission.py data/raw/nikluge-2022-nli-test-answer.jsonl "$path" --json || true
  fi
}

run_reg() {
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
    return 0
  fi
  echo "=== run $tag model=$model seed=$seed start $(date -Is) ==="
  local cmd=(env CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/train_transformer_regressor.py
    --model-name "$model"
    --output-dir "runs/$tag"
    --seed "$seed"
    --batch-size "$batch"
    --lr "$lr"
    --epochs "$epochs"
    --patience 4
    --standardize-target
    "$@")
  if [[ "$precision" == "fp16" ]]; then
    cmd+=(--use-fp16)
  fi
  "${cmd[@]}"
  local status=$?
  echo "=== run $tag end status=$status $(date -Is) ==="
  score_run "$tag"
  refresh_ensembles
  return 0
}

refresh_ensembles

run_reg xlm-roberta-large-xnli-gpu3-cls-seed3407 joeddav/xlm-roberta-large-xnli 3407 8 1e-5 10 fp16 --pooling cls
run_reg xlm-roberta-large-xnli-gpu3-mean-seed3407 joeddav/xlm-roberta-large-xnli 3407 8 1e-5 10 fp16 --pooling mean
run_reg xlm-roberta-large-xnli-gpu3-clsmean-seed3407 joeddav/xlm-roberta-large-xnli 3407 8 1e-5 10 fp16 --pooling cls_mean
run_reg xlm-roberta-large-xnli-anli-gpu3-mean-seed3407 vicgalle/xlm-roberta-large-xnli-anli 3407 8 1e-5 10 fp16 --pooling mean
run_reg xlm-roberta-large-xnli-gpu3-mean-long384-seed42 joeddav/xlm-roberta-large-xnli 42 4 8e-6 8 fp16 --max-length 384 --pooling mean

refresh_ensembles
echo "=== wave8 GPU3 runner end $(date -Is) ==="
