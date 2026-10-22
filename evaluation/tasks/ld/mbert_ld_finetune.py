#!/usr/bin/env python3
"""Fine-tune mBERT or XLM-R for 3-way Cantonese/Mandarin/mixed LD."""

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
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ld_metrics import LABELS, read_ld_jsonl, sha256


EVALUATOR_VERSION = "ld-encoder-v1"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="google-bert/bert-base-multilingual-cased")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-eval", type=int)
    return parser.parse_args()


def make_dataset(path: Path, limit: int | None = None) -> tuple[Dataset, list[dict]]:
    rows = read_ld_jsonl(path, limit)
    dataset = Dataset.from_dict(
        {
            "id": [row["id"] for row in rows],
            "source_id": [row["source_id"] for row in rows],
            "text": [row["sentence"] for row in rows],
            "label": [row["label"] for row in rows],
        }
    )
    return dataset, rows


def compute_metrics(eval_prediction):
    logits, gold = eval_prediction
    predicted = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, predicted, labels=list(range(len(LABELS))), average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(gold, predicted),
        "macro_f1": f1,
        "macro_precision": precision,
        "macro_recall": recall,
    }


def main() -> None:
    args = arguments()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": args.data_dir / "ld_train.jsonl",
        "validation": args.data_dir / "ld_val.jsonl",
        "test": args.data_dir / "ld_test.jsonl",
    }
    source_hashes = {name: sha256(path) for name, path in paths.items()}
    datasets_and_rows = {
        "train": make_dataset(paths["train"], args.limit_train),
        "validation": make_dataset(paths["validation"], args.limit_eval),
        "test": make_dataset(paths["test"], args.limit_eval),
    }
    raw = {name: value[0] for name, value in datasets_and_rows.items()}
    rows = {name: value[1] for name, value in datasets_and_rows.items()}

    source_sets = {
        name: {row["source_id"] for row in split_rows} for name, split_rows in rows.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = source_sets[left] & source_sets[right]
        if overlap:
            raise ValueError(f"source_id leakage between {left} and {right}: {len(overlap)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    encoded = {
        name: dataset.map(
            tokenize, batched=True, remove_columns=["id", "source_id", "text"]
        )
        for name, dataset in raw.items()
    }
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id={label: i for i, label in enumerate(LABELS)},
    )
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    train_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    train_result = trainer.train()
    train_seconds = time.perf_counter() - started
    validation_metrics = trainer.evaluate(
        encoded["validation"], metric_key_prefix="validation"
    )
    prediction = trainer.predict(encoded["test"], metric_key_prefix="test")
    predicted_ids = np.argmax(prediction.predictions, axis=-1)
    gold_ids = prediction.label_ids

    prediction_path = args.output_dir / "test_predictions.jsonl"
    with prediction_path.open("w", encoding="utf-8") as stream:
        for row, gold, predicted in zip(rows["test"], gold_ids, predicted_ids):
            record = {
                "id": row["id"],
                "source_id": row["source_id"],
                "gold_id": int(gold),
                "gold": LABELS[int(gold)],
                "prediction_id": int(predicted),
                "prediction": LABELS[int(predicted)],
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "model": args.model,
        "task": "Cantonese/Mandarin/mixed language detection",
        "protocol": "supervised fine-tuning; best validation macro-F1 checkpoint; test evaluated once",
        "primary_metric": "test_macro_f1",
        "label_mapping": {str(i): label for i, label in enumerate(LABELS)},
        "split_sizes": {name: len(dataset) for name, dataset in raw.items()},
        "source_id_counts": {name: len(ids) for name, ids in source_sets.items()},
        "source_id_overlap": 0,
        "source_sha256": source_hashes,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "seed": args.seed,
        },
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_macro_f1": trainer.state.best_metric,
        "validation_metrics": {key: float(value) for key, value in validation_metrics.items()},
        "test_metrics": {key: float(value) for key, value in prediction.metrics.items()},
        "classification_report": classification_report(
            gold_ids,
            predicted_ids,
            labels=list(range(len(LABELS))),
            target_names=list(LABELS),
            output_dict=True,
            zero_division=0,
        ),
        "train_seconds": train_seconds,
        "total_seconds": time.perf_counter() - started,
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
        "evaluator_version": EVALUATOR_VERSION,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    trainer.save_model(args.output_dir / "best_model")
    tokenizer.save_pretrained(args.output_dir / "best_model")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Predictions: {prediction_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")


if __name__ == "__main__":
    main()
