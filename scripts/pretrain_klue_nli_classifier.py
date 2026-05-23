#!/usr/bin/env python
"""Pre-fine-tune an encoder on public KLUE-NLI classification data."""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("runs/klue-nli-pretrain-roberta-large-seed42"))
    parser.add_argument("--model-name", default="klue/roberta-large")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--use-fp16", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class KlueNLIDataset(Dataset):
    def __init__(self, split: str, tokenizer: Any, max_length: int) -> None:
        self.rows = load_dataset("klue", "nli", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        encoded = self.tokenizer(
            row["premise"],
            row["hypothesis"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "label": int(row["label"]),
        }


@dataclass
class Collator:
    tokenizer: Any

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        features = [
            {"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]}
            for item in batch
        ]
        padded = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        padded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return padded


class Classifier(nn.Module):
    def __init__(self, model_name: str, num_labels: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.float()
        hidden_size = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        return self.head(pooled)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, use_fp16: bool) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_fp16 and device.type == "cuda"):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)
            losses.append(float(loss.detach().cpu()))
            pred = logits.argmax(dim=-1)
            correct += int((pred == labels).sum().detach().cpu())
            total += int(labels.numel())
    return {"loss": float(np.mean(losses)), "accuracy": correct / total}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps(vars(args), ensure_ascii=False, default=str), flush=True)
    print(f"device={device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_loader = DataLoader(
        KlueNLIDataset("train", tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=Collator(tokenizer),
    )
    valid_loader = DataLoader(
        KlueNLIDataset("validation", tokenizer, args.max_length),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=Collator(tokenizer),
    )

    model = Classifier(args.model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        int(total_steps * args.warmup_ratio),
        total_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_fp16 and device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()

    best_acc = -1.0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.use_fp16 and device.type == "cuda"):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))

        metrics = evaluate(model, valid_loader, device, args.use_fp16)
        metrics["epoch"] = epoch
        metrics["train_loss"] = float(np.mean(losses))
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)
        if metrics["accuracy"] > best_acc:
            best_acc = float(metrics["accuracy"])
            torch.save(model.encoder.state_dict(), args.output_dir / "best_encoder.pt")
            with (args.output_dir / "best_metrics.json").open("w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
        with (args.output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"done best_accuracy={best_acc:.6f}", flush=True)


if __name__ == "__main__":
    main()
