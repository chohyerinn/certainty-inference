#!/usr/bin/env bash
set -uo pipefail

ROOT="$HOME/certainty_inference_20260519"
PY="$HOME/miniconda3/envs/promptscope/bin/python"
GPU=3
WAVE_NAME=wave11
LOG_DIR="$ROOT/logs"
LOG="$LOG_DIR/wave11_gpu3_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR" "$ROOT/reports" "$ROOT/submissions"
ln -sfn "$(basename "$LOG")" "$LOG_DIR/wave11_gpu3_latest.log"
cd "$ROOT" || exit 1
exec > >(tee -a "$LOG") 2>&1

echo "=== wave11 GPU3 runner start $(date -Is) ==="

backup_best() {
  WAVE_NAME="$WAVE_NAME" "$PY" - <<'PY'
import glob
import json
import os
import shutil

report_path = "reports/greedy_dev_ensemble_calibrated_report.json"
submission_path = "submissions/greedy_dev_ensemble_calibrated.jsonl"
if not os.path.exists(report_path) or not os.path.exists(submission_path):
    raise SystemExit(0)

report = json.load(open(report_path, encoding="utf-8"))
current = report.get("best", {}).get("test_answer_mse")
if current is None:
    raise SystemExit(0)

best_seen = float("inf")
for path in glob.glob("reports/best_wave*_report_mse*.json"):
    try:
        prior = json.load(open(path, encoding="utf-8"))
    except Exception:
        continue
    value = prior.get("best", {}).get("test_answer_mse")
    if value is not None:
        best_seen = min(best_seen, float(value))

if float(current) < best_seen - 1e-12:
    stamp = f"{float(current):.6f}".replace(".", "")
    wave = os.environ.get("WAVE_NAME", "wave")
    report_out = f"reports/best_{wave}_calibrated_report_mse{stamp}.json"
    submission_out = f"submissions/best_{wave}_calibrated_mse{stamp}.jsonl"
    shutil.copy2(report_path, report_out)
    shutil.copy2(submission_path, submission_out)
    print(f"backup_best current={float(current):.9f} report={report_out} submission={submission_out}", flush=True)
else:
    print(f"no_backup current={float(current):.9f} best_seen={best_seen:.9f}", flush=True)
PY
}

refresh_ensembles() {
  "$PY" scripts/greedy_ensemble.py --steps 1800 \
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
  backup_best
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
}

score_run() {
  local tag="$1"
  local path="runs/$tag/test_submission.jsonl"
  if [[ -f "$path" && -f data/raw/nikluge-2022-nli-test-answer.jsonl ]]; then
    "$PY" scripts/score_submission.py data/raw/nikluge-2022-nli-test-answer.jsonl "$path" --json | tee "runs/$tag/test_answer_score.json" || true
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
    score_run "$tag"
    refresh_ensembles
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

run_reg xlm-roberta-large-xnli-gpu3-mean-seed555 joeddav/xlm-roberta-large-xnli 555 8 1e-5 10 fp16 --pooling mean
run_reg xlm-roberta-large-xnli-gpu3-mean-seed777 joeddav/xlm-roberta-large-xnli 777 8 1e-5 10 fp16 --pooling mean
run_reg xlm-roberta-large-xnli-gpu3-mean-seed888 joeddav/xlm-roberta-large-xnli 888 8 1e-5 10 fp16 --pooling mean
run_reg xlm-roberta-large-xnli-gpu3-mean-seed999 joeddav/xlm-roberta-large-xnli 999 8 1e-5 10 fp16 --pooling mean
run_reg xlm-roberta-large-xnli-gpu3-mean-seed1234 joeddav/xlm-roberta-large-xnli 1234 8 1e-5 10 fp16 --pooling mean
run_reg xlm-roberta-large-xnli-gpu3-mean-seed2025 joeddav/xlm-roberta-large-xnli 2025 8 1e-5 10 fp16 --pooling mean

run_reg xlm-roberta-large-xnli-gpu3-mean-long384-seed2024 joeddav/xlm-roberta-large-xnli 2024 4 8e-6 8 fp16 --max-length 384 --pooling mean
run_reg xlm-roberta-large-xnli-gpu3-mean-long384-seed3407 joeddav/xlm-roberta-large-xnli 3407 4 8e-6 8 fp16 --max-length 384 --pooling mean

refresh_ensembles
echo "=== wave11 GPU3 runner end $(date -Is) ==="
