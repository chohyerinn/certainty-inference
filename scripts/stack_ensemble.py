#!/usr/bin/env python
"""Build weighted dev ensembles from available run predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--dev-answer", type=Path, default=Path("data/raw/nikluge-2022-nli-dev.jsonl"))
    parser.add_argument("--test-answer", type=Path, default=Path("data/raw/nikluge-2022-nli-test-answer.jsonl"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("submissions/stacked_dev_ensemble.jsonl"))
    parser.add_argument("--report-json", type=Path, default=Path("reports/stacked_dev_ensemble_report.json"))
    parser.add_argument("--max-iter", type=int, default=5000)
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


def keep_run(name: str, include_prefixes: list[str], exclude_prefixes: list[str]) -> bool:
    if include_prefixes and not any(name.startswith(prefix) for prefix in include_prefixes):
        return False
    return not any(name.startswith(prefix) for prefix in exclude_prefixes)


def project_simplex(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if np.all(values >= 0.0) and abs(float(values.sum()) - 1.0) < 1e-12:
        return values
    order = np.sort(values)[::-1]
    cssv = np.cumsum(order) - 1.0
    idx = np.arange(1, len(values) + 1)
    keep = order - cssv / idx > 0
    if not np.any(keep):
        return np.full_like(values, 1.0 / len(values))
    theta = cssv[keep][-1] / idx[keep][-1]
    return np.maximum(values - theta, 0.0)


def fit_simplex(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int,
    prior: np.ndarray | None = None,
) -> np.ndarray:
    n_rows, n_cols = x.shape
    weights = np.full(n_cols, 1.0 / n_cols)
    if prior is None:
        prior = weights.copy()
    spectral = float(np.linalg.norm(x, ord=2) ** 2)
    step = 1.0 / max((2.0 * spectral / n_rows) + (2.0 * alpha), 1e-9)
    for _ in range(max_iter):
        pred = x @ weights
        grad = (2.0 / n_rows) * (x.T @ (pred - y)) + 2.0 * alpha * (weights - prior)
        weights = project_simplex(weights - step * grad)
    return weights


def collect_candidates(
    runs_dir: Path,
    y_dev: np.ndarray,
    y_test: np.ndarray | None,
    include_prefixes: list[str],
    exclude_prefixes: list[str],
) -> list[dict[str, Any]]:
    candidates = []
    for run in sorted(runs_dir.iterdir()):
        if not keep_run(run.name, include_prefixes, exclude_prefixes):
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
        test_mse = mse(test_pred, y_test) if y_test is not None and len(test_pred) == len(y_test) else None
        candidates.append(
            {
                "run": run.name,
                "dev_path": str(dev_path),
                "test_path": str(test_path),
                "dev_pred": dev_pred,
                "test_pred": test_pred,
                "dev_mse": mse(dev_pred, y_dev),
                "test_answer_mse": test_mse,
            }
        )
    return sorted(candidates, key=lambda row: row["dev_mse"])


def summarize_weights(names: list[str], weights: np.ndarray, limit: int = 25) -> list[dict[str, Any]]:
    order = np.argsort(-weights)
    rows = []
    for idx in order[:limit]:
        if weights[idx] <= 1e-8:
            continue
        rows.append({"run": names[idx], "weight": float(weights[idx])})
    return rows


def main() -> None:
    args = parse_args()
    y_dev = outputs(args.dev_answer)
    y_test = outputs(args.test_answer) if args.test_answer.exists() else None
    candidates = collect_candidates(args.runs_dir, y_dev, y_test, args.include_prefix, args.exclude_prefix)
    if not candidates:
        raise SystemExit("no candidates found")

    variants: list[dict[str, Any]] = []
    for top_k in [1, 2, 3, 5, 8, 12, 20, 30, len(candidates)]:
        top_k = min(top_k, len(candidates))
        subset = candidates[:top_k]
        names = [row["run"] for row in subset]
        x_dev = np.stack([row["dev_pred"] for row in subset], axis=1)
        x_test = np.stack([row["test_pred"] for row in subset], axis=1)

        uniform = np.full(top_k, 1.0 / top_k)
        inv = np.asarray([1.0 / max(row["dev_mse"], 1e-12) for row in subset], dtype=np.float64)
        inv /= inv.sum()
        priors = {"uniform": uniform, "inverse_dev": inv}

        for prior_name, prior in priors.items():
            pred_dev = x_dev @ prior
            pred_test = x_test @ prior
            variants.append(
                {
                    "name": f"{prior_name}_top{top_k}",
                    "dev_mse": mse(pred_dev, y_dev),
                    "test_answer_mse": mse(pred_test, y_test) if y_test is not None else None,
                    "weights": summarize_weights(names, prior),
                    "pred_test": pred_test,
                }
            )

        for alpha in [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            for prior_name, prior in priors.items():
                weights = fit_simplex(x_dev, y_dev, alpha, args.max_iter, prior=prior)
                pred_dev = x_dev @ weights
                pred_test = x_test @ weights
                variants.append(
                    {
                        "name": f"simplex_top{top_k}_alpha{alpha:g}_{prior_name}",
                        "dev_mse": mse(pred_dev, y_dev),
                        "test_answer_mse": mse(pred_test, y_test) if y_test is not None else None,
                        "weights": summarize_weights(names, weights),
                        "pred_test": pred_test,
                    }
                )

    best = min(variants, key=lambda row: row["dev_mse"])
    write_submission(args.output_jsonl, Path(candidates[0]["test_path"]), best["pred_test"])

    report_variants = []
    for row in sorted(variants, key=lambda item: item["dev_mse"])[:40]:
        report_variants.append({k: v for k, v in row.items() if k != "pred_test"})
    report = {
        "candidate_count": len(candidates),
        "include_prefix": args.include_prefix,
        "exclude_prefix": args.exclude_prefix,
        "single_best_by_dev": [
            {
                "run": row["run"],
                "dev_mse": row["dev_mse"],
                "test_answer_mse": row["test_answer_mse"],
            }
            for row in candidates[:30]
        ],
        "best": {k: v for k, v in best.items() if k != "pred_test"},
        "top_variants": report_variants,
        "output_jsonl": str(args.output_jsonl),
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
