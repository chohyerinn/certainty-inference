#!/usr/bin/env python
"""Create TF-IDF nearest-neighbor regression candidates."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/raw/nikluge-2022-nli-train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/raw/nikluge-2022-nli-dev.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/raw/nikluge-2022-nli-test.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--prefix", default="tfidf-knn")
    parser.add_argument("--top-n", type=int, default=40)
    parser.add_argument("--report-json", type=Path, default=Path("reports/tfidf_knn_report.json"))
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def label_array(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(row["output"]) for row in rows], dtype=np.float64)


def normalize(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_text(row: dict[str, Any], mode: str) -> str:
    item = row["input"]
    context = normalize(item["context"])
    prompt = normalize(item["prompt"])
    if mode == "prompt":
        return prompt
    if mode == "context":
        return context
    if mode == "cp":
        return context + " [SEP] " + prompt
    if mode == "pc":
        return prompt + " [SEP] " + context
    if mode == "prompt3_cp":
        return (prompt + " ") * 3 + "[SEP] " + context
    if mode == "prompt5_cp":
        return (prompt + " ") * 5 + "[SEP] " + context
    raise ValueError(mode)


def mse(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean((np.clip(pred, 1.0, 7.0) - y_true) ** 2))


def predict_from_similarity(
    sim: sparse.csr_matrix,
    y_train: np.ndarray,
    k: int,
    power: float,
    shrink: float,
) -> np.ndarray:
    mean = float(y_train.mean())
    preds: list[float] = []
    indptr = sim.indptr
    indices = sim.indices
    data = sim.data
    for row in range(sim.shape[0]):
        start, end = indptr[row], indptr[row + 1]
        row_idx = indices[start:end]
        row_sim = data[start:end]
        if len(row_idx) == 0:
            pred = mean
        else:
            if len(row_idx) > k:
                top = np.argpartition(row_sim, -k)[-k:]
                row_idx = row_idx[top]
                row_sim = row_sim[top]
            order = np.argsort(-row_sim)
            row_idx = row_idx[order]
            row_sim = np.maximum(row_sim[order], 0.0)
            if power == 0.0:
                weights = np.ones_like(row_sim, dtype=np.float64)
            else:
                weights = np.power(row_sim + 1e-12, power)
            denom = float(weights.sum())
            pred = mean if denom <= 0.0 else float(np.dot(weights, y_train[row_idx]) / denom)
        preds.append((1.0 - shrink) * pred + shrink * mean)
    return np.clip(np.asarray(preds, dtype=np.float64), 1.0, 7.0)


def write_jsonl(path: Path, template: list[dict[str, Any]], pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row, value in zip(template, np.clip(pred, 1.0, 7.0)):
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


def vectorizer_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for analyzer, ranges in [
        ("char_wb", [(2, 4), (3, 5), (2, 5), (3, 6), (2, 6)]),
        ("char", [(2, 4), (3, 5), (2, 5), (3, 6), (2, 6)]),
        ("word", [(1, 1), (1, 2), (1, 3)]),
    ]:
        for ngram_range in ranges:
            configs.append(
                {
                    "analyzer": analyzer,
                    "ngram_range": ngram_range,
                    "min_df": 1,
                    "max_df": 1.0,
                    "sublinear_tf": True,
                    "norm": "l2",
                }
            )
    return configs


def main() -> None:
    args = parse_args()
    train_rows = load_jsonl(args.train)
    dev_rows = load_jsonl(args.dev)
    test_rows = load_jsonl(args.test)
    y_train = label_array(train_rows)
    y_dev = label_array(dev_rows)

    modes = ["prompt", "context", "cp", "pc", "prompt3_cp", "prompt5_cp"]
    ks = [1, 2, 3, 5, 8, 13, 21, 34, 55]
    powers = [0.0, 1.0, 2.0, 4.0, 8.0]
    shrinks = [0.0, 0.05, 0.1, 0.2]

    candidates: list[dict[str, Any]] = []
    cache: dict[tuple[str, str, tuple[int, int]], tuple[sparse.csr_matrix, sparse.csr_matrix]] = {}
    for mode in modes:
        train_texts = [make_text(row, mode) for row in train_rows]
        dev_texts = [make_text(row, mode) for row in dev_rows]
        test_texts = [make_text(row, mode) for row in test_rows]
        all_texts = train_texts + dev_texts + test_texts
        for cfg in vectorizer_configs():
            key = (mode, cfg["analyzer"], cfg["ngram_range"])
            vectorizer = TfidfVectorizer(**cfg)
            vectorizer.fit(all_texts)
            x_train = vectorizer.transform(train_texts)
            x_dev = vectorizer.transform(dev_texts)
            x_test = vectorizer.transform(test_texts)
            sim_dev = (x_dev @ x_train.T).tocsr()
            sim_test = (x_test @ x_train.T).tocsr()
            cache[key] = (sim_dev, sim_test)
            for k, power, shrink in itertools.product(ks, powers, shrinks):
                dev_pred = predict_from_similarity(sim_dev, y_train, k=k, power=power, shrink=shrink)
                score = mse(y_dev, dev_pred)
                candidates.append(
                    {
                        "mode": mode,
                        "analyzer": cfg["analyzer"],
                        "ngram_range": cfg["ngram_range"],
                        "k": k,
                        "power": power,
                        "shrink": shrink,
                        "dev_mse": score,
                        "dev_pred": dev_pred,
                        "sim_key": key,
                    }
                )

    candidates.sort(key=lambda row: row["dev_mse"])
    selected = candidates[: args.top_n]
    report = []
    seen_names: set[str] = set()
    for rank, cand in enumerate(selected, start=1):
        mode = cand["mode"]
        analyzer = cand["analyzer"]
        lo, hi = cand["ngram_range"]
        k = cand["k"]
        power = cand["power"]
        shrink = cand["shrink"]
        tag = f"{args.prefix}-{rank:02d}-{mode}-{analyzer}{lo}{hi}-k{k}-p{power:g}-s{shrink:g}"
        tag = tag.replace(".", "p")
        if tag in seen_names:
            tag += f"-r{rank}"
        seen_names.add(tag)
        run_dir = args.out_dir / tag
        sim_dev, sim_test = cache[cand["sim_key"]]
        test_pred = predict_from_similarity(sim_test, y_train, k=k, power=power, shrink=shrink)
        write_jsonl(run_dir / "dev_predictions.jsonl", dev_rows, cand["dev_pred"])
        write_jsonl(run_dir / "test_submission.jsonl", test_rows, test_pred)
        metrics = {
            "run": tag,
            "mode": mode,
            "analyzer": analyzer,
            "ngram_range": [lo, hi],
            "k": k,
            "power": power,
            "shrink": shrink,
            "dev_mse": cand["dev_mse"],
        }
        (run_dir / "best_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        report.append(metrics)

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report[:20], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
