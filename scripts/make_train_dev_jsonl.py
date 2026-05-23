#!/usr/bin/env python
"""Concatenate labeled train/dev JSONL files for final-fit runs."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/raw/nikluge-2022-nli-train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/raw/nikluge-2022-nli-dev.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/derived/nikluge-2022-nli-traindev.jsonl"))
    return parser.parse_args()


def read_lines(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def main() -> None:
    args = parse_args()
    rows = read_lines(args.train) + read_lines(args.dev)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(row + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
