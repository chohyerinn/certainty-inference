#!/usr/bin/env python
"""Fit simple dev-set calibration transforms and apply them to test predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-answer", type=Path, default=Path("data/raw/nikluge-2022-nli-dev.jsonl"))
    parser.add_argument("--dev-pred", type=Path, default=Path("submissions/greedy_dev_ensemble_all_dev.jsonl"))
    parser.add_argument("--test-pred", type=Path, default=Path("submissions/greedy_dev_ensemble_all.jsonl"))
    parser.add_argument("--test-answer", type=Path, default=Path("data/raw/nikluge-2022-nli-test-answer.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("submissions/greedy_dev_ensemble_calibrated.jsonl"))
    parser.add_argument("--report-json", type=Path, default=Path("reports/greedy_dev_ensemble_calibrated_report.json"))
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def outputs(path: Path) -> np.ndarray:
    return np.asarray([float(row["output"]) for row in load_jsonl(path)], dtype=np.float64)


def mse(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean((np.clip(pred, 1.0, 7.0) - truth) ** 2))


def design(values: np.ndarray, degree: int) -> np.ndarray:
    cols = [np.ones_like(values)]
    for power in range(1, degree + 1):
        cols.append(values**power)
    return np.stack(cols, axis=1)


def fit_poly(x: np.ndarray, y: np.ndarray, degree: int, alpha: float) -> np.ndarray:
    x_design = design(x, degree)
    eye = np.eye(x_design.shape[1])
    eye[0, 0] = 0.0
    return np.linalg.solve(x_design.T @ x_design + alpha * eye, x_design.T @ y)


def apply_poly(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return design(x, len(coef) - 1) @ coef


def write_submission(path: Path, template_rows: list[dict[str, Any]], pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row, value in zip(template_rows, np.clip(pred, 1.0, 7.0)):
            f.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "input": row["input"],
                        "output": f"{float(value):.6f}".rstrip("0").rstrip("."),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    y_dev = outputs(args.dev_answer)
    p_dev = outputs(args.dev_pred)
    p_test = outputs(args.test_pred)
    test_rows = load_jsonl(args.test_pred)
    y_test = outputs(args.test_answer) if args.test_answer.exists() else None

    variants: list[dict[str, Any]] = []
    for degree in [1, 2, 3]:
        for alpha in [0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            coef = fit_poly(p_dev, y_dev, degree, alpha)
            dev_cal = apply_poly(p_dev, coef)
            test_cal = apply_poly(p_test, coef)
            variants.append(
                {
                    "name": f"poly{degree}_alpha{alpha:g}",
                    "degree": degree,
                    "alpha": alpha,
                    "coef": coef.tolist(),
                    "dev_mse": mse(dev_cal, y_dev),
                    "test_answer_mse": mse(test_cal, y_test) if y_test is not None else None,
                    "test_pred": test_cal,
                }
            )

    variants.append(
        {
            "name": "identity",
            "degree": 0,
            "alpha": 0.0,
            "coef": [],
            "dev_mse": mse(p_dev, y_dev),
            "test_answer_mse": mse(p_test, y_test) if y_test is not None else None,
            "test_pred": p_test,
        }
    )
    best = min(variants, key=lambda row: row["dev_mse"])
    write_submission(args.output_jsonl, test_rows, best["test_pred"])
    report = {
        "best": {k: v for k, v in best.items() if k != "test_pred"},
        "variants": [{k: v for k, v in row.items() if k != "test_pred"} for row in sorted(variants, key=lambda row: row["dev_mse"])],
        "output_jsonl": str(args.output_jsonl),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
