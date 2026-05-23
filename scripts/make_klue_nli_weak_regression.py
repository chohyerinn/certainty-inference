#!/usr/bin/env python
"""Convert public KLUE-NLI labels into weak 1..7 regression targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


LABEL_TO_SCORE = {
    0: "6.75",  # entailment
    1: "4.0",   # neutral
    2: "1.25",  # contradiction
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--split", default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("klue", "nli", split=args.split)
    with args.output_jsonl.open("w", encoding="utf-8", newline="\n") as f:
        for idx, row in enumerate(dataset):
            item = {
                "id": f"klue-nli-weak-{args.split}-{idx:06d}",
                "input": {
                    "context": row["premise"],
                    "prompt": row["hypothesis"],
                },
                "output": LABEL_TO_SCORE[int(row["label"])],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"wrote {len(dataset)} rows to {args.output_jsonl}")


if __name__ == "__main__":
    main()
