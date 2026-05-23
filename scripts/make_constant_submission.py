#!/usr/bin/env python
"""Build a JSONL submission by assigning the same score to every item."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--value", type=float, default=6.0)
    parser.add_argument("--digits", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1.0 <= args.value <= 7.0:
        raise SystemExit("--value must be between 1 and 7")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_value = f"{args.value:.{args.digits}f}".rstrip("0").rstrip(".")

    count = 0
    with args.input_jsonl.open("r", encoding="utf-8") as src, args.output_jsonl.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row or "input" not in row:
                raise ValueError(f"line {line_no}: expected id and input keys")
            row["output"] = output_value
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"wrote {count} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
