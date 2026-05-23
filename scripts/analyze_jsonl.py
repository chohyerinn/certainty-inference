#!/usr/bin/env python
"""Print compact statistics for a certainty-inference JSONL file."""

from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def describe(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "max": max(values),
        "pstdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.jsonl)
    context_lens = [len(row["input"]["context"]) for row in rows]
    prompt_lens = [len(row["input"]["prompt"]) for row in rows]

    result: dict[str, Any] = {
        "rows": len(rows),
        "first_id": rows[0]["id"] if rows else None,
        "last_id": rows[-1]["id"] if rows else None,
        "context_chars": describe([float(x) for x in context_lens]) if rows else None,
        "prompt_chars": describe([float(x) for x in prompt_lens]) if rows else None,
    }

    if rows and "output" in rows[0]:
        outputs = [float(row["output"]) for row in rows]
        result["output"] = describe(outputs)
        result["top_outputs"] = collections.Counter(outputs).most_common(20)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"rows: {result['rows']}")
    print(f"first_id: {result['first_id']}")
    print(f"last_id: {result['last_id']}")
    print(f"context_chars: {result['context_chars']}")
    print(f"prompt_chars: {result['prompt_chars']}")
    if "output" in result:
        print(f"output: {result['output']}")
        print("top_outputs:")
        for value, count in result["top_outputs"]:
            print(f"  {value}: {count}")


if __name__ == "__main__":
    main()
