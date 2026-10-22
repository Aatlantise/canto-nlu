#!/usr/bin/env python3
"""Fine-tune mBERT on CantoNLU OpenRice sentiment without test leakage."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
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

LABELS = ["smile", "ok", "cry"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("mbert-results/openrice-sa"))
    p.add_argument("--model", default="google-bert/bert-base-multilingual-cased")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--validation-only",
        action="store_true",
        help="Search mode: train/select using train+valid and do not evaluate test.",
    )
    return p.parse_args()


def load_split(path):
    frame = pd.read_csv(path, sep="\t")
    if list(frame.columns) != ["label", "text_a"]:
        raise ValueError(f"Unexpected columns in {path}: {list(frame.columns)}")
    unknown = set(frame["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"Unknown labels in {path}: {sorted(unknown)}")
    return Dataset.from_dict({
        "text": frame["text_a"].fillna("").astype(str).tolist(),
        "label": [LABEL2ID[x] for x in frame["label"]],
    })


def metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1,
        "macro_precision": precision,
        "macro_recall": recall,
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = {name: load_split(args.data_dir / f"{name}.tsv") for name in ("train", "valid", "test")}
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    encoded = {name: ds.map(tokenize, batched=True, remove_columns=["text"]) for name, ds in raw.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label={i: label for i, label in enumerate(LABELS)},
        label2id=LABEL2ID,
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    train_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
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
        eval_dataset=encoded["valid"],
        compute_metrics=metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    train_result = trainer.train()
    train_seconds = time.time() - started

    valid_metrics = trainer.evaluate(encoded["valid"], metric_key_prefix="valid")
    if args.validation_only:
        search_summary = {
            "model": args.model,
            "protocol": "hyperparameter search using train and valid only; test was not evaluated",
            "split_sizes": {"train": len(raw["train"]), "valid": len(raw["valid"])},
            "hyperparameters": {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": 0.01,
                "max_length": args.max_length,
                "seed": args.seed,
            },
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_validation_macro_f1": trainer.state.best_metric,
            "validation_metrics": {k: float(v) for k, v in valid_metrics.items()},
            "train_seconds": train_seconds,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "peak_gpu_memory_gib": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None,
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
            },
            "train_metrics": {k: float(v) for k, v in train_result.metrics.items()},
        }
        with open(args.output_dir / "search_summary.json", "w", encoding="utf-8") as f:
            json.dump(search_summary, f, ensure_ascii=False, indent=2)
        print(json.dumps(search_summary, ensure_ascii=False, indent=2))
        print(f"Search summary: {(args.output_dir / 'search_summary.json').resolve()}")
        return

    test_started = time.time()
    prediction = trainer.predict(encoded["test"], metric_key_prefix="test")
    test_seconds = time.time() - test_started
    pred_ids = np.argmax(prediction.predictions, axis=-1)
    gold_ids = prediction.label_ids

    rows = []
    for i, (gold, pred) in enumerate(zip(gold_ids, pred_ids)):
        rows.append({"index": i, "gold": LABELS[int(gold)], "prediction": LABELS[int(pred)]})
    with open(args.output_dir / "test_predictions.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "model": args.model,
        "protocol": "supervised fine-tuning: train; checkpoint selection: validation macro-F1; final evaluation: test once",
        "split_sizes": {name: len(ds) for name, ds in raw.items()},
        "hyperparameters": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "weight_decay": 0.01,
            "max_length": args.max_length, "seed": args.seed,
        },
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_valid_macro_f1": trainer.state.best_metric,
        "validation_metrics": {k: float(v) for k, v in valid_metrics.items()},
        "test_metrics": {k: float(v) for k, v in prediction.metrics.items()},
        "classification_report": classification_report(
            gold_ids, pred_ids, target_names=LABELS, output_dict=True, zero_division=0
        ),
        "train_seconds": train_seconds,
        "test_seconds": test_seconds,
        "total_seconds": time.time() - started,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_gpu_memory_gib": (torch.cuda.max_memory_allocated() / 1024**3) if torch.cuda.is_available() else None,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__},
        "train_metrics": {k: float(v) for k, v in train_result.metrics.items()},
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    trainer.save_model(args.output_dir / "best_model")
    tokenizer.save_pretrained(args.output_dir / "best_model")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary: {(args.output_dir / 'summary.json').resolve()}")


if __name__ == "__main__":
    main()
