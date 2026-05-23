#!/usr/bin/env python
"""Average predictions from multiple JSONL submissions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("submission_jsonl", nargs="+", type=Path)
    parser.add_argument("--clip-min", type=float, default=1.0)
    parser.add_argument("--clip-max", type=float, default=7.0)
    parser.add_argument("--digits", type=int, default=6)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    args = parse_args()
    all_rows = [load_jsonl(path) for path in args.submission_jsonl]
    if not all_rows:
        raise SystemExit("at least one submission is required")

    n = len(all_rows[0])
    for path, rows in zip(args.submission_jsonl, all_rows):
        if len(rows) != n:
            raise SystemExit(f"row count mismatch for {path}: expected {n}, got {len(rows)}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8", newline="\n") as f:
        for idx, base_row in enumerate(all_rows[0]):
            item_id = base_row["id"]
            item_input = base_row["input"]
            preds = []
            for path, rows in zip(args.submission_jsonl, all_rows):
                row = rows[idx]
                if row["id"] != item_id:
                    raise SystemExit(f"id mismatch at row {idx + 1} in {path}")
                preds.append(float(row["output"]))
            pred = float(np.clip(np.mean(preds), args.clip_min, args.clip_max))
            output = f"{pred:.{args.digits}f}".rstrip("0").rstrip(".")
            f.write(json.dumps({"id": item_id, "input": item_input, "output": output}, ensure_ascii=False) + "\n")

    print(f"wrote ensemble of {len(args.submission_jsonl)} submissions to {args.output_jsonl}")


if __name__ == "__main__":
    main()
