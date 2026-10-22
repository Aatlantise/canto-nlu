#!/usr/bin/env python3
"""Leakage-safe supervised sentiment fine-tuning for encoder models."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


EVALUATOR_VERSION = "cantonlu-sa-encoder-supervised-v1"
LABELS = ("smile", "ok", "cry")
LABEL2ID = {label: index for index, label in enumerate(LABELS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--valid-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-valid-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if "sentence" not in item or "label" not in item:
                raise ValueError(f"Missing sentence/label at {path}:{line_number}")
            label = str(item["label"])
            if label not in LABEL2ID:
                raise ValueError(f"Unknown label {label!r} at {path}:{line_number}")
            rows.append(
                {
                    "row_id": item.get("id", len(rows)),
                    "text": str(item["sentence"]),
                    "labels": LABEL2ID[label],
                }
            )
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def balanced_prefix(rows: list[dict], maximum: int | None) -> list[dict]:
    """Take a deterministic near-balanced subset for smoke tests."""
    if maximum is None or maximum >= len(rows):
        return rows
    buckets = {label_id: [] for label_id in range(len(LABELS))}
    for row in rows:
        buckets[row["labels"]].append(row)
    selected: list[dict] = []
    cursor = 0
    while len(selected) < maximum:
        added = False
        for label_id in range(len(LABELS)):
            if cursor < len(buckets[label_id]) and len(selected) < maximum:
                selected.append(buckets[label_id][cursor])
                added = True
        if not added:
            break
        cursor += 1
    return selected


def compute_metrics(eval_prediction) -> dict[str, float]:
    logits, gold_ids = eval_prediction
    if isinstance(logits, tuple):
        logits = logits[0]
    pred_ids = np.argmax(logits, axis=-1)
    report = classification_report(
        gold_ids,
        pred_ids,
        labels=list(range(len(LABELS))),
        target_names=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(gold_ids, pred_ids),
        "macro_f1": report["macro avg"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = {
        "train": balanced_prefix(read_jsonl(args.train_data), args.max_train_samples),
        "valid": balanced_prefix(read_jsonl(args.valid_data), args.max_valid_samples),
        "test": balanced_prefix(read_jsonl(args.test_data), args.max_test_samples),
    }
    datasets = {name: Dataset.from_list(rows) for name, rows in source_rows.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    encoded = {
        name: dataset.map(tokenize, batched=True, remove_columns=["text", "row_id"])
        for name, dataset in datasets.items()
    }
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label={index: label for index, label in enumerate(LABELS)},
        label2id=LABEL2ID,
    )

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["valid"],
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    train_result = trainer.train()
    train_seconds = time.time() - started
    valid_metrics = trainer.evaluate(encoded["valid"], metric_key_prefix="valid")

    test_started = time.time()
    prediction = trainer.predict(encoded["test"], metric_key_prefix="test")
    test_seconds = time.time() - test_started
    logits = prediction.predictions[0] if isinstance(prediction.predictions, tuple) else prediction.predictions
    pred_ids = np.argmax(logits, axis=-1)
    gold_ids = prediction.label_ids
    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()

    predictions_path = args.output_dir / "test_predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as stream:
        for row, gold_id, pred_id, probs in zip(source_rows["test"], gold_ids, pred_ids, probabilities):
            record = {
                "row_id": row["row_id"],
                "gold": LABELS[int(gold_id)],
                "prediction": LABELS[int(pred_id)],
                "probabilities": {label: float(probs[index]) for index, label in enumerate(LABELS)},
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = classification_report(
        gold_ids,
        pred_ids,
        labels=list(range(len(LABELS))),
        target_names=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    summary = {
        "evaluator_version": EVALUATOR_VERSION,
        "model": args.model,
        "protocol": "supervised train; validation macro-F1 checkpoint selection; held-out test evaluated once",
        "labels": list(LABELS),
        "split_sizes": {name: len(rows) for name, rows in source_rows.items()},
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "seed": args.seed,
        },
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_macro_f1": trainer.state.best_metric,
        "validation_metrics": {key: float(value) for key, value in valid_metrics.items()},
        "test_metrics": {key: float(value) for key, value in prediction.metrics.items()},
        "classification_report": report,
        "confusion_matrix_rows_gold_columns_prediction": confusion_matrix(
            gold_ids, pred_ids, labels=list(range(len(LABELS)))
        ).tolist(),
        "train_seconds": train_seconds,
        "test_seconds": test_seconds,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None
        ),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "train_metrics": {key: float(value) for key, value in train_result.metrics.items()},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    trainer.save_model(args.output_dir / "best_model")
    tokenizer.save_pretrained(args.output_dir / "best_model")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
