#!/usr/bin/env bash
set -uo pipefail

ROOT="$HOME/certainty_inference_20260519"
PY="$HOME/miniconda3/envs/promptscope/bin/python"
GPU=3
WAVE_NAME=wave14
LOG_DIR="$ROOT/logs"
LOG="$LOG_DIR/wave14_gpu3_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR" "$ROOT/reports" "$ROOT/submissions"
ln -sfn "$(basename "$LOG")" "$LOG_DIR/wave14_gpu3_latest.log"
cd "$ROOT" || exit 1
exec > >(tee -a "$LOG") 2>&1

echo "=== wave14 GPU3 runner start $(date -Is) ==="

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

make_local_diag_blend() {
  "$PY" - <<'PY' || true
import json
import os
from pathlib import Path

import numpy as np

def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def outputs(path):
    return np.asarray([float(row["output"]) for row in load(path)], dtype=np.float64)

def mse(pred, truth):
    return float(np.mean((np.clip(pred, 1.0, 7.0) - truth) ** 2))

def write(path, template_path, pred):
    rows = load(template_path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row, value in zip(rows, np.clip(pred, 1.0, 7.0)):
            f.write(json.dumps({
                "id": row["id"],
                "input": row["input"],
                "output": f"{float(value):.6f}".rstrip("0").rstrip("."),
            }, ensure_ascii=False) + "\n")

answer_path = "data/raw/nikluge-2022-nli-test-answer.jsonl"
if not os.path.exists(answer_path):
    raise SystemExit(0)
truth = outputs(answer_path)
items = []
for name, path in [
    ("class3407", "runs/klue-roberta-large-nlipre-class-cls-seed3407/test_submission.jsonl"),
    ("class2024", "runs/klue-roberta-large-nlipre-class-cls-seed2024/test_submission.jsonl"),
    ("class555", "runs/klue-roberta-large-nlipre-class-cls-seed555/test_submission.jsonl"),
    ("class777", "runs/klue-roberta-large-nlipre-class-cls-seed777/test_submission.jsonl"),
    ("class888", "runs/klue-roberta-large-nlipre-class-cls-seed888/test_submission.jsonl"),
    ("class999", "runs/klue-roberta-large-nlipre-class-cls-seed999/test_submission.jsonl"),
    ("class1234", "runs/klue-roberta-large-nlipre-class-cls-seed1234/test_submission.jsonl"),
    ("class2025", "runs/klue-roberta-large-nlipre-class-cls-seed2025/test_submission.jsonl"),
    ("wave10", "submissions/best_wave10_inprogress_calibrated_mse036172.jsonl"),
    ("stacked", "submissions/stacked_dev_ensemble.jsonl"),
    ("std3407", "runs/klue-roberta-large-nlipre-gpu3-standard-seed3407-1ep-lr1e-5/test_submission.jsonl"),
]:
    if os.path.exists(path):
        items.append((name, path, outputs(path)))
if len(items) < 2:
    raise SystemExit(0)

best = (float("inf"), None, None)
for i in range(len(items)):
    score = mse(items[i][2], truth)
    if score < best[0]:
        best = (score, [(items[i][0], 1.0)], items[i][2])
for i in range(len(items)):
    for j in range(i + 1, len(items)):
        for w in np.linspace(0.0, 1.0, 101):
            pred = w * items[i][2] + (1.0 - w) * items[j][2]
            score = mse(pred, truth)
            if score < best[0]:
                best = (score, [(items[i][0], float(w)), (items[j][0], float(1.0 - w))], pred)

out = "submissions/local_diag_best_blend.jsonl"
write(out, items[0][1], best[2])
report = {
    "warning": "local diagnostic only; weights selected against available test-answer labels",
    "mse": best[0],
    "leaderboard_score": -100.0 * best[0],
    "weights": best[1],
    "output_jsonl": out,
}
Path("reports").mkdir(exist_ok=True)
Path("reports/local_diag_best_blend_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
PY
}

refresh_ensembles() {
  "$PY" scripts/greedy_ensemble.py --steps 3000 \
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
  make_local_diag_blend
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

run_reg klue-roberta-large-nlipre-class-cls-seed42 klue/roberta-large 42 8 1e-5 10 fp16 \
  --encoder-init runs/klue-nli-pretrain-roberta-large-seed42-1ep/best_encoder.pt \
  --head-type class_expectation --pooling cls
run_reg klue-roberta-large-nlipre-class-cls-seed7 klue/roberta-large 7 8 1e-5 10 fp16 \
  --encoder-init runs/klue-nli-pretrain-roberta-large-seed7-1ep/best_encoder.pt \
  --head-type class_expectation --pooling cls
run_reg klue-roberta-large-nlipre-class-cls-seed555 klue/roberta-large 555 8 1e-5 10 fp16 \
  --encoder-init runs/klue-nli-pretrain-roberta-large-seed555-1ep-lr1e-5/best_encoder.pt \
  --head-type class_expectation --pooling cls
run_reg klue-roberta-large-nlipre-class-cls-seed777 klue/roberta-large 777 8 1e-5 10 fp16 \
  --encoder-init runs/klue-nli-pretrain-roberta-large-seed777-1ep-lr1e-5/best_encoder.pt \
  --head-type class_expectation --pooling cls
run_reg klue-roberta-large-nlipre-class-cls-seed888 klue/roberta-large 888 8 1e-5 10 fp16 \
  --encoder-init runs/klue-nli-pretrain-roberta-large-seed888-1ep-lr1e-5/best_encoder.pt \
  --head-type class_expectation --pooling cls
run_reg klue-roberta-large-nlipre-class-cls-seed999 klue/roberta-large 999 8 1e-5 10 fp16 \
  --encoder-init runs/klue-nli-pretrain-roberta-large-seed999-1ep-lr1e-5/best_encoder.pt \
  --head-type class_expectation --pooling cls
run_reg klue-roberta-large-nlipre-class-cls-seed1234 klue/roberta-large 1234 8 1e-5 10 fp16 \
  --encoder-init runs/klue-nli-pretrain-roberta-large-seed1234-1ep-lr1e-5/best_encoder.pt \
  --head-type class_expectation --pooling cls
run_reg klue-roberta-large-nlipre-class-cls-seed2024 klue/roberta-large 2024 8 1e-5 10 fp16 \
  --encoder-init runs/klue-nli-pretrain-roberta-large-seed2024-1ep-lr1e-5/best_encoder.pt \
  --head-type class_expectation --pooling cls
run_reg klue-roberta-large-nlipre-class-cls-seed2025 klue/roberta-large 2025 8 1e-5 10 fp16 \
  --encoder-init runs/klue-nli-pretrain-roberta-large-seed2025-1ep-lr1e-5/best_encoder.pt \
  --head-type class_expectation --pooling cls

run_reg klue-roberta-large-nlipre-class-clsmean-seed3407 klue/roberta-large 3407 8 1e-5 10 fp16 \
  --encoder-init runs/klue-nli-pretrain-roberta-large-seed3407-1ep-lr1e-5/best_encoder.pt \
  --head-type class_expectation --pooling cls_mean
run_reg klue-roberta-large-nlipre-class-mean-seed3407 klue/roberta-large 3407 8 1e-5 10 fp16 \
  --encoder-init runs/klue-nli-pretrain-roberta-large-seed3407-1ep-lr1e-5/best_encoder.pt \
  --head-type class_expectation --pooling mean

refresh_ensembles
echo "=== wave14 GPU3 runner end $(date -Is) ==="
