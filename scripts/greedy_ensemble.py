#!/usr/bin/env python
"""Build a greedy ensemble from all runs with dev/test predictions."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--dev-answer", type=Path, default=Path("data/raw/nikluge-2022-nli-dev.jsonl"))
    parser.add_argument("--test-answer", type=Path, default=Path("data/raw/nikluge-2022-nli-test-answer.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("submissions/greedy_dev_ensemble_all.jsonl"))
    parser.add_argument("--dev-output-jsonl", type=Path, default=Path("submissions/greedy_dev_ensemble_all_dev.jsonl"))
    parser.add_argument("--report-json", type=Path, default=Path("reports/greedy_dev_ensemble_all_report.json"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--include-prefix", action="append", default=[])
    parser.add_argument("--exclude-prefix", action="append", default=[])
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def outputs(path: Path) -> np.ndarray:
    return np.asarray([float(row["output"]) for row in load_jsonl(path)], dtype=np.float64)


def mse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((np.clip(pred, 1.0, 7.0) - truth) ** 2))


def write_submission(path: Path, template_path: Path, pred: np.ndarray) -> None:
    rows = load_jsonl(template_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row, value in zip(rows, np.clip(pred, 1.0, 7.0)):
            out = {
                "id": row["id"],
                "input": row["input"],
                "output": f"{float(value):.6f}".rstrip("0").rstrip("."),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def keep_run(name: str, include_prefixes: list[str], exclude_prefixes: list[str]) -> bool:
    if include_prefixes and not any(name.startswith(prefix) for prefix in include_prefixes):
        return False
    return not any(name.startswith(prefix) for prefix in exclude_prefixes)


def main() -> None:
    args = parse_args()
    y_dev = outputs(args.dev_answer)
    y_test = outputs(args.test_answer) if args.test_answer.exists() else None

    candidates = []
    for run in sorted(args.runs_dir.iterdir()):
        if not keep_run(run.name, args.include_prefix, args.exclude_prefix):
            continue
        dev_path = run / "dev_predictions.jsonl"
        test_path = run / "test_submission.jsonl"
        if not dev_path.exists() or not test_path.exists():
            continue
        try:
            dev_pred = outputs(dev_path)
            test_pred = outputs(test_path)
        except Exception:
            continue
        if len(dev_pred) != len(y_dev):
            continue
        candidates.append(
            {
                "run": run.name,
                "dev_path": dev_path,
                "test_path": test_path,
                "dev_pred": dev_pred,
                "test_pred": test_pred,
                "dev_mse": mse(dev_pred, y_dev),
                "test_mse": mse(test_pred, y_test) if y_test is not None and len(test_pred) == len(y_test) else None,
            }
        )

    if not candidates:
        raise SystemExit("no candidates found")

    selected: list[int] = []
    current_dev: np.ndarray | None = None
    current_test: np.ndarray | None = None
    best_dev_score = float("inf")
    best_dev_pred: np.ndarray | None = None
    best_test_pred: np.ndarray | None = None
    best_selected: list[int] = []
    best_step = 0
    history = []

    for step in range(args.steps):
        best = None
        for idx, cand in enumerate(candidates):
            if current_dev is None:
                trial_dev = cand["dev_pred"]
                trial_test = cand["test_pred"]
            else:
                trial_dev = (current_dev * len(selected) + cand["dev_pred"]) / (len(selected) + 1)
                trial_test = (current_test * len(selected) + cand["test_pred"]) / (len(selected) + 1)
            score = mse(trial_dev, y_dev)
            if best is None or score < best[0]:
                best = (score, idx, trial_dev, trial_test)
        assert best is not None
        score, idx, current_dev, current_test = best
        selected.append(idx)
        test_score = mse(current_test, y_test) if y_test is not None and len(current_test) == len(y_test) else None
        history.append(
            {
                "step": step + 1,
                "added": candidates[idx]["run"],
                "dev_mse": score,
                "test_answer_mse": test_score,
            }
        )
        if score < best_dev_score:
            best_dev_score = score
            best_dev_pred = current_dev.copy()
            best_test_pred = current_test.copy()
            best_selected = selected[:]
            best_step = step + 1

    assert best_dev_pred is not None
    assert best_test_pred is not None
    write_submission(args.dev_output_jsonl, args.dev_answer, best_dev_pred)
    write_submission(args.output_jsonl, candidates[best_selected[0]]["test_path"], best_test_pred)

    counts = collections.Counter(candidates[idx]["run"] for idx in best_selected)
    best_test_score = mse(best_test_pred, y_test) if y_test is not None and len(best_test_pred) == len(y_test) else None
    report = {
        "candidate_count": len(candidates),
        "include_prefix": args.include_prefix,
        "exclude_prefix": args.exclude_prefix,
        "single_best_by_dev": sorted(
            [
                {"run": cand["run"], "dev_mse": cand["dev_mse"], "test_answer_mse": cand["test_mse"]}
                for cand in candidates
            ],
            key=lambda row: row["dev_mse"],
        )[:30],
        "selection_counts": counts.most_common(),
        "best_step": best_step,
        "history": history,
        "last_step_dev_mse": history[-1]["dev_mse"],
        "last_step_test_answer_mse": history[-1]["test_answer_mse"],
        "final_dev_mse": best_dev_score,
        "final_test_answer_mse": best_test_score,
        "dev_output_jsonl": str(args.dev_output_jsonl),
        "output_jsonl": str(args.output_jsonl),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
