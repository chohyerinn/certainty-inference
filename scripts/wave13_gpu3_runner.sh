#!/usr/bin/env bash
set -uo pipefail

ROOT="$HOME/certainty_inference_20260519"
PY="$HOME/miniconda3/envs/promptscope/bin/python"
GPU=3
WAVE_NAME=wave13
LOG_DIR="$ROOT/logs"
LOG="$LOG_DIR/wave13_gpu3_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR" "$ROOT/reports" "$ROOT/submissions"
ln -sfn "$(basename "$LOG")" "$LOG_DIR/wave13_gpu3_latest.log"
cd "$ROOT" || exit 1
exec > >(tee -a "$LOG") 2>&1

echo "=== wave13 GPU3 runner start $(date -Is) ==="

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
  "$PY" scripts/greedy_ensemble.py --steps 2600 \
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

# Partial-input and order-ablation candidates. These are intentionally different
# from normal NLI training so the greedy ensemble can exploit dataset-specific signal.
run_reg klue-roberta-large-nlipre-promptonly-reg-seed123 klue/roberta-large 123 8 1e-5 10 fp16 \
  --encoder-init runs/klue-nli-pretrain-roberta-large-seed123-1ep/best_encoder.pt \
  --input-mode prompt_only --pooling cls
run_reg klue-roberta-large-nlipre-contextonly-reg-seed123 klue/roberta-large 123 8 1e-5 10 fp16 \
  --encoder-init runs/klue-nli-pretrain-roberta-large-seed123-1ep/best_encoder.pt \
  --input-mode context_only --pooling cls
run_reg klue-roberta-large-nlipre-swapped-reg-seed123 klue/roberta-large 123 8 1e-5 10 fp16 \
  --encoder-init runs/klue-nli-pretrain-roberta-large-seed123-1ep/best_encoder.pt \
  --input-mode swapped_pair --pooling cls
run_reg klue-roberta-large-nlipre-marked-reg-seed123 klue/roberta-large 123 8 1e-5 10 fp16 \
  --encoder-init runs/klue-nli-pretrain-roberta-large-seed123-1ep/best_encoder.pt \
  --input-mode marked_concat --pooling cls

run_reg klue-roberta-large-nlipre-promptonly-class-seed123 klue/roberta-large 123 8 1e-5 10 fp16 \
  --encoder-init runs/klue-nli-pretrain-roberta-large-seed123-1ep/best_encoder.pt \
  --input-mode prompt_only --head-type class_expectation --pooling cls
run_reg klue-roberta-large-nlipre-swapped-class-seed3407 klue/roberta-large 3407 8 1e-5 10 fp16 \
  --encoder-init runs/klue-nli-pretrain-roberta-large-seed3407-1ep-lr1e-5/best_encoder.pt \
  --input-mode swapped_pair --head-type class_expectation --pooling cls

run_reg xlm-roberta-large-xnli-promptonly-mean-seed42 joeddav/xlm-roberta-large-xnli 42 8 1e-5 10 fp16 \
  --input-mode prompt_only --pooling mean
run_reg xlm-roberta-large-xnli-swapped-mean-seed42 joeddav/xlm-roberta-large-xnli 42 8 1e-5 10 fp16 \
  --input-mode swapped_pair --pooling mean
run_reg xlm-roberta-large-xnli-marked-mean-seed123 joeddav/xlm-roberta-large-xnli 123 8 1e-5 10 fp16 \
  --input-mode marked_concat --pooling mean

refresh_ensembles
echo "=== wave13 GPU3 runner end $(date -Is) ==="
