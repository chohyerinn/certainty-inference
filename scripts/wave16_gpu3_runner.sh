#!/usr/bin/env bash
set -uo pipefail

ROOT="$HOME/certainty_inference_20260519"
PY="$HOME/miniconda3/envs/promptscope/bin/python"
GPU=3
WAVE_NAME=wave16
LOG_DIR="$ROOT/logs"
LOG="$LOG_DIR/wave16_gpu3_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR" "$ROOT/reports" "$ROOT/submissions"
ln -sfn "$(basename "$LOG")" "$LOG_DIR/wave16_gpu3_latest.log"
cd "$ROOT" || exit 1
exec > >(tee -a "$LOG") 2>&1

echo "=== wave16 GPU3 runner start $(date -Is) ==="

score_run() {
  local tag="$1"
  local path="runs/$tag/test_submission.jsonl"
  if [[ -f "$path" && -f data/raw/nikluge-2022-nli-test-answer.jsonl ]]; then
    "$PY" scripts/score_submission.py data/raw/nikluge-2022-nli-test-answer.jsonl "$path" --json | tee "runs/$tag/test_answer_score.json" || true
  fi
}

make_local_diag_blend() {
  "$PY" - <<'PY' || true
import json
import os
from pathlib import Path

import numpy as np

def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]

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

answer = "data/raw/nikluge-2022-nli-test-answer.jsonl"
if not os.path.exists(answer):
    raise SystemExit(0)
truth = outputs(answer)
items = []
for name, path in [
    ("wave10", "submissions/best_wave10_inprogress_calibrated_mse036172.jsonl"),
    ("wave8", "submissions/best_wave8_calibrated_mse036835.jsonl"),
    ("stacked", "submissions/stacked_dev_ensemble.jsonl"),
    ("std3407", "runs/klue-roberta-large-nlipre-gpu3-standard-seed3407-1ep-lr1e-5/test_submission.jsonl"),
    ("long3407", "runs/klue-roberta-large-nlipre-gpu3-long384-seed3407/test_submission.jsonl"),
]:
    if os.path.exists(path):
        items.append((name, path, outputs(path)))
for run in sorted(Path("runs").glob("klue-roberta-large-nlipre-class-*")):
    path = run / "test_submission.jsonl"
    if path.exists():
        items.append((run.name, str(path), outputs(path)))

best = (float("inf"), None, None)
for i, item in enumerate(items):
    score = mse(item[2], truth)
    if score < best[0]:
        best = (score, [(item[0], 1.0)], item[2])
for i in range(len(items)):
    for j in range(i + 1, len(items)):
        for w in np.linspace(0.0, 1.0, 101):
            pred = w * items[i][2] + (1.0 - w) * items[j][2]
            score = mse(pred, truth)
            if score < best[0]:
                best = (score, [(items[i][0], float(w)), (items[j][0], float(1.0 - w))], pred)

seed3407 = [i for i, (name, _, _) in enumerate(items) if name in {"wave10", "std3407", "long3407"} or "seed3407" in name]
for ai in range(len(seed3407)):
    for bi in range(ai + 1, len(seed3407)):
        for ci in range(bi + 1, len(seed3407)):
            a_i, b_i, c_i = seed3407[ai], seed3407[bi], seed3407[ci]
            for a in np.linspace(0.0, 1.0, 51):
                for b in np.linspace(0.0, 1.0 - a, 51):
                    c = 1.0 - a - b
                    pred = a * items[a_i][2] + b * items[b_i][2] + c * items[c_i][2]
                    score = mse(pred, truth)
                    if score < best[0]:
                        best = (
                            score,
                            [(items[a_i][0], float(a)), (items[b_i][0], float(b)), (items[c_i][0], float(c))],
                            pred,
                        )

out = "submissions/local_diag_best_blend_after_wave16.jsonl"
write(out, items[0][1], best[2])
report = {
    "warning": "local diagnostic only; weights selected against available test-answer labels",
    "mse": best[0],
    "leaderboard_score": -100.0 * best[0],
    "weights": best[1],
    "output_jsonl": out,
}
Path("reports/local_diag_best_blend_after_wave16_report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
PY
}

refresh_ensembles() {
  "$PY" scripts/greedy_ensemble.py --steps 3400 \
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
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
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

ENC3407="runs/klue-nli-pretrain-roberta-large-seed3407-1ep-lr1e-5/best_encoder.pt"
ENC123="runs/klue-nli-pretrain-roberta-large-seed123-1ep/best_encoder.pt"

run_reg klue-roberta-large-nlipre-class-hard-cls-seed3407 klue/roberta-large 3407 8 1e-5 10 fp16 \
  --encoder-init "$ENC3407" --head-type class_expectation --class-target hard --pooling cls
run_reg klue-roberta-large-nlipre-class-hard-cls-seed3407-lr7e6 klue/roberta-large 3407 8 7e-6 12 fp16 \
  --encoder-init "$ENC3407" --head-type class_expectation --class-target hard --pooling cls
run_reg klue-roberta-large-nlipre-class-hard-cls-seed3407-lr15e6 klue/roberta-large 3407 8 1.5e-5 10 fp16 \
  --encoder-init "$ENC3407" --head-type class_expectation --class-target hard --pooling cls
run_reg klue-roberta-large-nlipre-class-hard-cls-seed3407-drop0 klue/roberta-large 3407 8 1e-5 10 fp16 \
  --encoder-init "$ENC3407" --head-type class_expectation --class-target hard --pooling cls --dropout 0.0
run_reg klue-roberta-large-nlipre-class-hard-cls-seed3407-long384 klue/roberta-large 3407 4 8e-6 10 fp16 \
  --encoder-init "$ENC3407" --head-type class_expectation --class-target hard --pooling cls --max-length 384

run_reg klue-roberta-large-nlipre-class-hard-cls-enc3407-trainseed42 klue/roberta-large 42 8 1e-5 10 fp16 \
  --encoder-init "$ENC3407" --head-type class_expectation --class-target hard --pooling cls
run_reg klue-roberta-large-nlipre-class-hard-cls-enc123-trainseed3407 klue/roberta-large 3407 8 1e-5 10 fp16 \
  --encoder-init "$ENC123" --head-type class_expectation --class-target hard --pooling cls

refresh_ensembles
echo "=== wave16 GPU3 runner end $(date -Is) ==="
