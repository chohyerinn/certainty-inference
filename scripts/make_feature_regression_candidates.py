#!/usr/bin/env python
"""Create small structured text-feature regression candidates."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR


CUES_NEG = ["않", "안 ", "못", "없", "아니", "반대", "불가능", "틀리", "거짓", "아니다"]
CUES_CERT = ["확실", "분명", "반드시", "사실", "맞", "기억", "알", "이다", "였다", "한다"]
CUES_UNCERT = ["듯", "것 같다", "가능", "수 있", "추정", "예상", "전망", "생각", "아마", "모르", "의문", "추측"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/raw/nikluge-2022-nli-train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/raw/nikluge-2022-nli-dev.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/raw/nikluge-2022-nli-test.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--prefix", default="feature-reg")
    parser.add_argument("--report-json", type=Path, default=Path("reports/feature_regression_report.json"))
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    return re.findall(r"[가-힣A-Za-z0-9]+", text)


def char_ngrams(text: str, n: int) -> set[str]:
    text = re.sub(r"\s+", "", text)
    return {text[i : i + n] for i in range(max(0, len(text) - n + 1))}


def jaccard(a: set[Any], b: set[Any]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def count_cues(text: str, cues: list[str]) -> int:
    return sum(text.count(cue) for cue in cues)


def longest_common_ratio(a: str, b: str) -> float:
    match = SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
    return match.size / max(1, len(b))


def row_features(row: dict[str, Any]) -> list[float]:
    item = row["input"]
    context = normalize(item["context"])
    prompt = normalize(item["prompt"])
    c_words = words(context)
    p_words = words(prompt)
    c_set = set(c_words)
    p_set = set(p_words)
    c_counter = Counter(c_words)
    p_counter = Counter(p_words)
    shared_count = sum(min(c_counter[w], p_counter[w]) for w in p_counter)
    nums_c = set(re.findall(r"\d+(?:\.\d+)?", context))
    nums_p = set(re.findall(r"\d+(?:\.\d+)?", prompt))
    last_sentences = re.split(r"(?:[.!?。！？]\\s+|다\\.\\s+|요\\.\\s+)", context)
    tail = " ".join(last_sentences[-2:])

    features = [
        len(context),
        len(prompt),
        len(c_words),
        len(p_words),
        len(prompt) / max(1, len(context)),
        len(p_words) / max(1, len(c_words)),
        float(prompt in context),
        float(prompt.replace(" ", "") in context.replace(" ", "")),
        SequenceMatcher(None, context, prompt).ratio(),
        SequenceMatcher(None, tail, prompt).ratio(),
        longest_common_ratio(context, prompt),
        longest_common_ratio(tail, prompt),
        jaccard(c_set, p_set),
        shared_count / max(1, len(p_words)),
        shared_count / max(1, len(c_words)),
        jaccard(char_ngrams(context, 2), char_ngrams(prompt, 2)),
        jaccard(char_ngrams(context, 3), char_ngrams(prompt, 3)),
        jaccard(char_ngrams(tail, 2), char_ngrams(prompt, 2)),
        jaccard(char_ngrams(tail, 3), char_ngrams(prompt, 3)),
        count_cues(context, CUES_NEG),
        count_cues(prompt, CUES_NEG),
        count_cues(context, CUES_CERT),
        count_cues(prompt, CUES_CERT),
        count_cues(context, CUES_UNCERT),
        count_cues(prompt, CUES_UNCERT),
        float(bool(nums_p)),
        len(nums_c & nums_p) / max(1, len(nums_p)),
        context.count("\"") + context.count("'") + context.count("“") + context.count("”"),
        prompt.count("?") + prompt.count("까"),
        context.count("P1:") + context.count("P2:"),
    ]
    return [float(x) for x in features]


def matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([row_features(row) for row in rows], dtype=np.float64)


def labels(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([float(row["output"]) for row in rows], dtype=np.float64)


def mse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.clip(p, 1.0, 7.0) - y) ** 2))


def write_jsonl(path: Path, template: list[dict[str, Any]], pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row, value in zip(template, np.clip(pred, 1.0, 7.0)):
            f.write(json.dumps({"id": row["id"], "input": row["input"], "output": f"{float(value):.6f}".rstrip("0").rstrip(".")}, ensure_ascii=False) + "\n")


def candidate_models(seed: int) -> list[tuple[str, Any]]:
    return [
        ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        ("ridge-poly2", make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(alpha=100.0))),
        ("elastic", make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=10000, random_state=seed))),
        ("svr-rbf", make_pipeline(StandardScaler(), SVR(C=3.0, epsilon=0.15, gamma="scale"))),
        ("kr-rbf", make_pipeline(StandardScaler(), KernelRidge(alpha=1.0, kernel="rbf", gamma=0.05))),
        ("rf", RandomForestRegressor(n_estimators=600, min_samples_leaf=3, random_state=seed, n_jobs=-1)),
        ("extra", ExtraTreesRegressor(n_estimators=800, min_samples_leaf=2, random_state=seed, n_jobs=-1)),
        ("gbr", GradientBoostingRegressor(random_state=seed, n_estimators=250, learning_rate=0.03, max_depth=2, subsample=0.8)),
        ("hist", HistGradientBoostingRegressor(random_state=seed, learning_rate=0.03, max_iter=300, l2_regularization=0.1, max_leaf_nodes=8)),
    ]


def main() -> None:
    args = parse_args()
    train_rows = load_jsonl(args.train)
    dev_rows = load_jsonl(args.dev)
    test_rows = load_jsonl(args.test)
    x_train = matrix(train_rows)
    x_dev = matrix(dev_rows)
    x_test = matrix(test_rows)
    y_train = labels(train_rows)
    y_dev = labels(dev_rows)

    report = []
    for seed in [42, 7, 123, 2024, 3407]:
        for name, model in candidate_models(seed):
            tag = f"{args.prefix}-{name}-seed{seed}"
            run_dir = args.out_dir / tag
            model.fit(x_train, y_train)
            dev_pred = np.clip(model.predict(x_dev), 1.0, 7.0)
            test_pred = np.clip(model.predict(x_test), 1.0, 7.0)
            write_jsonl(run_dir / "dev_predictions.jsonl", dev_rows, dev_pred)
            write_jsonl(run_dir / "test_submission.jsonl", test_rows, test_pred)
            metrics = {"run": tag, "model": name, "seed": seed, "dev_mse": mse(y_dev, dev_pred)}
            (run_dir / "best_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
            report.append(metrics)
    report.sort(key=lambda row: row["dev_mse"])
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report[:30], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
