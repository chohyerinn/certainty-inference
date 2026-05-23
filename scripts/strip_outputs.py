#!/usr/bin/env python
"""Create an input-only JSONL file by removing `output` from each row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with args.input_jsonl.open("r", encoding="utf-8") as src, args.output_jsonl.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line_no, line in enumerate(src, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.pop("output", None)
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    print(f"wrote {count} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
