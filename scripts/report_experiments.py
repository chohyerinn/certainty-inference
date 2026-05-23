#!/usr/bin/env python
"""Collect experiment metrics into a compact report."""

from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    rows = []
    for run_dir in sorted(Path("runs").glob("*")):
        if not run_dir.is_dir():
            continue
        best = load_json(run_dir / "best_metrics.json")
        test = load_json(run_dir / "test_answer_score.json")
        if not best and not test:
            continue
        rows.append(
            {
                "run": run_dir.name,
                "dev_mse": best.get("mse"),
                "best_epoch": best.get("epoch"),
                "test_answer_mse": test.get("mse"),
                "test_answer_lb": test.get("leaderboard_score"),
            }
        )
    rows.sort(key=lambda row: (row["dev_mse"] is None, row["dev_mse"] or 999, row["test_answer_mse"] or 999))
    out = Path("reports/experiment_report.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows[:30]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
