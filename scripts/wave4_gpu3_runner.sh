#!/usr/bin/env bash
set -uo pipefail

ROOT="$HOME/certainty_inference_20260519"
PY="$HOME/miniconda3/envs/promptscope/bin/python"
GPU=3
LOG_DIR="$ROOT/logs"
LOG="$LOG_DIR/wave4_gpu3_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR" "$ROOT/reports" "$ROOT/submissions"
ln -sfn "$(basename "$LOG")" "$LOG_DIR/wave4_gpu3_latest.log"
cd "$ROOT" || exit 1
exec > >(tee -a "$LOG") 2>&1

echo "=== wave4 GPU3 runner start $(date -Is) ==="
echo "root=$ROOT gpu=$GPU py=$PY"

gpu_status() {
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
}

score_run() {
  local tag="$1"
  local path="runs/$tag/test_submission.jsonl"
  if [[ -f "$path" && -f data/raw/nikluge-2022-nli-test-answer.jsonl ]]; then
    "$PY" scripts/score_submission.py data/raw/nikluge-2022-nli-test-answer.jsonl "$path" --json || true
  fi
}

refresh_ensembles() {
  echo "--- refresh ensembles $(date -Is) ---"
  "$PY" scripts/greedy_ensemble.py --steps 900 \
    --output-jsonl submissions/greedy_dev_ensemble_all.jsonl \
    --dev-output-jsonl submissions/greedy_dev_ensemble_all_dev.jsonl \
    --report-json reports/greedy_dev_ensemble_all_report.json || true
  "$PY" scripts/stack_ensemble.py \
    --output-jsonl submissions/stacked_dev_ensemble.jsonl \
    --report-json reports/stacked_dev_ensemble_report.json || true
  "$PY" scripts/calibrate_predictions.py \
    --dev-pred submissions/greedy_dev_ensemble_all_dev.jsonl \
    --test-pred submissions/greedy_dev_ensemble_all.jsonl \
    --output-jsonl submissions/greedy_dev_ensemble_calibrated.jsonl \
    --report-json reports/greedy_dev_ensemble_calibrated_report.json || true
  "$PY" scripts/report_experiments.py > reports/experiments_report.json || true
  echo "--- greedy summary ---"
  grep -E 'candidate_count|best_step|final_dev_mse|final_test_answer_mse' reports/greedy_dev_ensemble_all_report.json 2>/dev/null || true
  echo "--- calibrated summary ---"
  grep -E '"name"|"dev_mse"|"test_answer_mse"' reports/greedy_dev_ensemble_calibrated_report.json 2>/dev/null | head -n 30 || true
  gpu_status || true
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

run_nli_chain() {
  local seed="$1"
  local pre_epochs="$2"
  local lr="$3"
  local suffix="seed${seed}-${pre_epochs}ep-lr${lr//./p}"
  local pre_dir="runs/klue-nli-pretrain-roberta-large-${suffix}"

  if [[ ! -f "$pre_dir/best_encoder.pt" ]]; then
    echo "=== pretrain KLUE-NLI $suffix start $(date -Is) ==="
    env CUDA_VISIBLE_DEVICES="$GPU" "$PY" scripts/pretrain_klue_nli_classifier.py \
      --model-name klue/roberta-large \
      --output-dir "$pre_dir" \
      --seed "$seed" \
      --epochs "$pre_epochs" \
      --batch-size 8 \
      --lr "$lr" \
      --use-fp16 || true
    echo "=== pretrain KLUE-NLI $suffix end $(date -Is) ==="
  fi

  if [[ -f "$pre_dir/best_encoder.pt" ]]; then
    run_reg "klue-roberta-large-nlipre-gpu3-standard-${suffix}" klue/roberta-large "$seed" 8 1e-5 10 fp16 \
      --encoder-init "$pre_dir/best_encoder.pt" --pooling cls
  fi
}

run_encoder_variant() {
  local label="$1"
  local seed="$2"
  local encoder="$3"
  local pooling="$4"
  local max_length="$5"
  local batch="$6"
  local lr="$7"
  local epochs="$8"
  if [[ -f "$encoder" ]]; then
    run_reg "$label" klue/roberta-large "$seed" "$batch" "$lr" "$epochs" fp16 \
      --encoder-init "$encoder" --pooling "$pooling" --max-length "$max_length"
  else
    echo "missing encoder: $encoder"
  fi
}

refresh_ensembles

run_reg klue-roberta-large-gpu3-mean-seed42 klue/roberta-large 42 8 1e-5 10 fp16 --pooling mean
run_reg klue-roberta-large-gpu3-mean-seed2024 klue/roberta-large 2024 8 1e-5 10 fp16 --pooling mean
run_reg klue-roberta-large-gpu3-mean-seed3407 klue/roberta-large 3407 8 1e-5 10 fp16 --pooling mean
run_reg klue-roberta-large-gpu3-mean-seed555 klue/roberta-large 555 8 1e-5 10 fp16 --pooling mean
run_reg klue-roberta-large-gpu3-mean-seed888 klue/roberta-large 888 8 1e-5 10 fp16 --pooling mean

run_reg klue-roberta-large-gpu3-mean-lr7e6-seed123 klue/roberta-large 123 8 7e-6 12 fp16 --pooling mean
run_reg klue-roberta-large-gpu3-mean-lr15e6-seed123 klue/roberta-large 123 8 1.5e-5 10 fp16 --pooling mean
run_reg klue-roberta-large-gpu3-mean-long384-seed123 klue/roberta-large 123 4 8e-6 12 fp16 --pooling mean --max-length 384

ENC3407="runs/klue-nli-pretrain-roberta-large-seed3407-1ep-lr1e-5/best_encoder.pt"
ENC2024="runs/klue-nli-pretrain-roberta-large-seed2024-1ep-lr1e-5/best_encoder.pt"
run_encoder_variant klue-roberta-large-nlipre-gpu3-mean-lr7e6-seed3407 3407 "$ENC3407" mean 256 8 7e-6 12
run_encoder_variant klue-roberta-large-nlipre-gpu3-mean-lr15e6-seed3407 3407 "$ENC3407" mean 256 8 1.5e-5 10
run_encoder_variant klue-roberta-large-nlipre-gpu3-mean-lr7e6-seed2024 2024 "$ENC2024" mean 256 8 7e-6 12
run_encoder_variant klue-roberta-large-nlipre-gpu3-mean-lr15e6-seed2024 2024 "$ENC2024" mean 256 8 1.5e-5 10

run_nli_chain 999 1 1e-5
run_nli_chain 1234 1 1e-5
run_nli_chain 2025 1 1e-5

refresh_ensembles
echo "=== wave4 GPU3 runner end $(date -Is) ==="
