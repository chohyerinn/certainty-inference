#!/usr/bin/env python
"""Fine-tune a transformer encoder for certainty-inference regression."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/raw/nikluge-2022-nli-train.jsonl"))
    parser.add_argument("--dev", type=Path, default=Path("data/raw/nikluge-2022-nli-dev.jsonl"))
    parser.add_argument("--test", type=Path, default=Path("data/raw/nikluge-2022-nli-test.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/xlm-roberta-base-seed42"))
    parser.add_argument("--model-name", default="xlm-roberta-base")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--standardize-target", action="store_true")
    parser.add_argument("--no-dev-eval", action="store_true")
    parser.add_argument("--encoder-init", type=Path, default=None)
    parser.add_argument("--pooling", choices=["cls", "mean", "cls_mean"], default="cls")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--loss", choices=["mse", "smooth_l1"], default="mse")
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--head-type", choices=["regression", "class_expectation"], default="regression")
    parser.add_argument("--class-target", choices=["soft", "hard"], default="soft")
    parser.add_argument(
        "--input-mode",
        choices=["pair", "swapped_pair", "context_only", "prompt_only", "marked_concat"],
        default="pair",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class NLIDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        max_length: int,
        has_labels: bool,
        input_mode: str,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_labels = has_labels
        self.input_mode = input_mode

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        inputs = row["input"]
        if self.input_mode == "swapped_pair":
            encoded = self.tokenizer(
                inputs["prompt"],
                inputs["context"],
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )
        elif self.input_mode == "context_only":
            encoded = self.tokenizer(
                inputs["context"],
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )
        elif self.input_mode == "prompt_only":
            encoded = self.tokenizer(
                inputs["prompt"],
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )
        elif self.input_mode == "marked_concat":
            encoded = self.tokenizer(
                f"맥락: {inputs['context']} 제시문: {inputs['prompt']}",
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )
        else:
            encoded = self.tokenizer(
                inputs["context"],
                inputs["prompt"],
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )
        item: dict[str, Any] = {
            "id": row["id"],
            "input": row["input"],
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        if self.has_labels:
            item["labels"] = float(row["output"])
        return item


@dataclass
class Collator:
    tokenizer: Any
    has_labels: bool

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        features = [
            {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]}
            for item in batch
        ]
        padded = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        result: dict[str, Any] = {
            "ids": [item["id"] for item in batch],
            "inputs": [item["input"] for item in batch],
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
        }
        if self.has_labels:
            result["labels"] = torch.tensor([item["labels"] for item in batch], dtype=torch.float)
        return result


class Regressor(nn.Module):
    def __init__(
        self,
        model_name: str,
        dropout: float = 0.1,
        pooling: str = "cls",
        head_type: str = "regression",
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.head_type = head_type
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.float()
        hidden_size = self.encoder.config.hidden_size
        head_size = hidden_size * 2 if pooling == "cls_mean" else hidden_size
        out_size = 7 if head_type == "class_expectation" else 1
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_size),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        cls_pooled = hidden[:, 0]
        if self.pooling == "cls":
            pooled = cls_pooled
        else:
            mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
            mean_pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            pooled = mean_pooled if self.pooling == "mean" else torch.cat([cls_pooled, mean_pooled], dim=-1)
        output = self.head(pooled)
        if self.head_type == "class_expectation":
            return output
        return output.squeeze(-1)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": mae,
        "leaderboard_score": -100.0 * mse,
    }


def soft_ordinal_targets(labels: torch.Tensor) -> torch.Tensor:
    labels = labels.clamp(1.0, 7.0)
    lower = torch.floor(labels).long().clamp(1, 7)
    upper = torch.ceil(labels).long().clamp(1, 7)
    upper_weight = labels - lower.to(dtype=labels.dtype)
    lower_weight = 1.0 - upper_weight
    target = torch.zeros(labels.shape[0], 7, dtype=labels.dtype, device=labels.device)
    target.scatter_add_(1, (lower - 1).unsqueeze(1), lower_weight.unsqueeze(1))
    target.scatter_add_(1, (upper - 1).unsqueeze(1), upper_weight.unsqueeze(1))
    return target


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
    has_labels: bool,
    target_mean: float,
    target_std: float,
    head_type: str,
) -> tuple[list[str], list[dict[str, str]], np.ndarray, np.ndarray | None]:
    model.eval()
    ids: list[str] = []
    inputs: list[dict[str, str]] = []
    preds: list[float] = []
    labels: list[float] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16 and device.type == "cuda"):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
            if head_type == "class_expectation":
                values = torch.arange(1, 8, dtype=logits.dtype, device=logits.device)
                logits = (torch.softmax(logits, dim=-1) * values).sum(dim=-1)
            elif target_std != 1.0 or target_mean != 0.0:
                logits = logits * target_std + target_mean
            pred = torch.clamp(logits, 1.0, 7.0).detach().cpu().numpy()
            ids.extend(batch["ids"])
            inputs.extend(batch["inputs"])
            preds.extend(pred.tolist())
            if has_labels:
                labels.extend(batch["labels"].numpy().tolist())
    y_pred = np.asarray(preds, dtype=np.float64)
    y_true = np.asarray(labels, dtype=np.float64) if has_labels else None
    return ids, inputs, y_pred, y_true


def write_submission(path: Path, ids: list[str], inputs: list[dict[str, str]], preds: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for item_id, item_input, pred in zip(ids, inputs, preds):
            row = {
                "id": item_id,
                "input": item_input,
                "output": f"{float(pred):.6f}".rstrip("0").rstrip("."),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps(vars(args), ensure_ascii=False, default=str), flush=True)
    print(f"device={device} cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_rows = read_jsonl(args.train)
    dev_rows = [] if args.no_dev_eval else read_jsonl(args.dev)
    test_rows = read_jsonl(args.test)
    train_targets = np.asarray([float(row["output"]) for row in train_rows], dtype=np.float64)
    target_mean = float(train_targets.mean()) if args.standardize_target else 0.0
    target_std = float(train_targets.std()) if args.standardize_target else 1.0
    if target_std == 0.0:
        target_std = 1.0
    print(f"target_mean={target_mean:.8f} target_std={target_std:.8f}", flush=True)

    train_loader = DataLoader(
        NLIDataset(train_rows, tokenizer, args.max_length, has_labels=True, input_mode=args.input_mode),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=Collator(tokenizer, has_labels=True),
    )
    dev_loader = None
    if not args.no_dev_eval:
        dev_loader = DataLoader(
            NLIDataset(dev_rows, tokenizer, args.max_length, has_labels=True, input_mode=args.input_mode),
            batch_size=args.batch_size * 2,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=Collator(tokenizer, has_labels=True),
        )
    test_loader = DataLoader(
        NLIDataset(test_rows, tokenizer, args.max_length, has_labels=False, input_mode=args.input_mode),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=Collator(tokenizer, has_labels=False),
    )

    model = Regressor(
        args.model_name,
        dropout=args.dropout,
        pooling=args.pooling,
        head_type=args.head_type,
    ).to(device)
    if args.encoder_init is not None:
        state = torch.load(args.encoder_init, map_location="cpu")
        missing, unexpected = model.encoder.load_state_dict(state, strict=False)
        print(
            f"loaded encoder_init={args.encoder_init} missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_fp16 and device.type == "cuda")
    if args.loss == "smooth_l1":
        loss_fn = nn.SmoothL1Loss(beta=args.smooth_l1_beta)
    else:
        loss_fn = nn.MSELoss()

    best_mse = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            if args.head_type == "class_expectation":
                if args.class_target == "hard":
                    target = (torch.round(labels).long() - 1).clamp(0, 6)
                else:
                    target = soft_ordinal_targets(labels)
            else:
                target = labels
            if args.standardize_target and args.head_type == "regression":
                labels = (labels - target_mean) / target_std
                target = labels
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.use_fp16 and device.type == "cuda"):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                if args.head_type == "class_expectation":
                    if args.class_target == "hard":
                        loss = F.cross_entropy(logits, target)
                    else:
                        loss = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                else:
                    loss = loss_fn(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))

        metrics: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
        }
        if dev_loader is not None:
            dev_ids, dev_inputs, dev_pred, dev_true = predict(
                model, dev_loader, device, args.use_fp16, True, target_mean, target_std, args.head_type
            )
            assert dev_true is not None
            metrics.update(metric_dict(dev_true, dev_pred))
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)

        score = float(metrics.get("mse", -epoch))
        improved = score < best_mse if dev_loader is not None else epoch == args.epochs
        if improved:
            best_mse = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save(model.state_dict(), args.output_dir / "best_model.pt")
            torch.save(model.encoder.state_dict(), args.output_dir / "best_encoder.pt")
            if dev_loader is not None:
                write_submission(args.output_dir / "dev_predictions.jsonl", dev_ids, dev_inputs, dev_pred)
            with (args.output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
        else:
            bad_epochs += 1

        with (args.output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        if dev_loader is not None and bad_epochs >= args.patience:
            print(f"early_stop epoch={epoch} best_epoch={best_epoch} best_mse={best_mse:.8f}", flush=True)
            break

    model.load_state_dict(torch.load(args.output_dir / "best_model.pt", map_location=device))
    test_ids, test_inputs, test_pred, _ = predict(
        model, test_loader, device, args.use_fp16, False, target_mean, target_std, args.head_type
    )
    write_submission(args.output_dir / "test_submission.jsonl", test_ids, test_inputs, test_pred)
    print(f"done best_epoch={best_epoch} best_mse={best_mse:.10f}", flush=True)


if __name__ == "__main__":
    main()
