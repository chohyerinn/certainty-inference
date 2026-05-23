#!/usr/bin/env python
"""Validate and score a submission JSONL against an answer JSONL."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("answer_jsonl", type=Path)
    parser.add_argument("submission_jsonl", type=Path)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument(
        "--no-strict-range",
        action="store_true",
        help="allow predictions outside the official 1..7 range",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def to_float(value: Any, label: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: output is not numeric: {value!r}") from exc
    if not math.isfinite(score):
        raise ValueError(f"{label}: output is not finite: {value!r}")
    return score


def main() -> None:
    args = parse_args()
    answers = load_jsonl(args.answer_jsonl)
    submissions = load_jsonl(args.submission_jsonl)

    if len(answers) != len(submissions):
        raise SystemExit(f"row count mismatch: answers={len(answers)} submissions={len(submissions)}")

    squared_errors: list[float] = []
    absolute_errors: list[float] = []

    for idx, (answer, submission) in enumerate(zip(answers, submissions), start=1):
        answer_id = answer.get("id")
        submission_id = submission.get("id")
        if answer_id != submission_id:
            raise SystemExit(f"line {idx}: id mismatch: answer={answer_id!r} submission={submission_id!r}")
        if "output" not in submission:
            raise SystemExit(f"line {idx} ({answer_id}): missing output")

        y_true = to_float(answer.get("output"), f"answer line {idx}")
        y_pred = to_float(submission.get("output"), f"submission line {idx}")
        if not args.no_strict_range and not 1.0 <= y_pred <= 7.0:
            raise SystemExit(f"line {idx} ({answer_id}): prediction outside 1..7: {y_pred}")

        err = y_pred - y_true
        squared_errors.append(err * err)
        absolute_errors.append(abs(err))

    mse = sum(squared_errors) / len(squared_errors)
    mae = sum(absolute_errors) / len(absolute_errors)
    rmse = math.sqrt(mse)
    leaderboard_score = -100.0 * mse
    max_abs_error = max(absolute_errors)

    result = {
        "rows": len(answers),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max_abs_error,
        "leaderboard_score": leaderboard_score,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"rows: {result['rows']}")
        print(f"MSE: {mse:.10f}")
        print(f"RMSE: {rmse:.10f}")
        print(f"MAE: {mae:.10f}")
        print(f"max_abs_error: {max_abs_error:.10f}")
        print(f"leaderboard_score (-100*MSE): {leaderboard_score:.7f}")


if __name__ == "__main__":
    main()
