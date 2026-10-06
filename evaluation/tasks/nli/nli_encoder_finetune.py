#!/usr/bin/env python3
"""Fine-tune mBERT or XLM-R-base for binary Cantonese NLI, then score a JSONL sample."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path


LABEL2ID = {"not_entailment": 0, "entailment": 1}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}
EVALUATOR_VERSION = "cantonese-nli-encoder-finetune-v1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default="hon9kon9ize/yue-all-nli")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-triplets", type=int)
    parser.add_argument("--max-validation-triplets", type=int)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def load_test_rows(path: Path):
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"id", "premise", "hypothesis", "label"} - set(row)
            if missing:
                raise ValueError(f"Line {line_number} is missing {sorted(missing)}")
            if row["label"] not in LABEL2ID:
                raise ValueError(f"Line {line_number} has unknown label {row['label']!r}")
            if str(row["id"]) in seen:
                raise ValueError(f"Duplicate id {row['id']!r}")
            seen.add(str(row["id"]))
            rows.append(
                {
                    "id": str(row["id"]),
                    "premise": str(row["premise"]),
                    "hypothesis": str(row["hypothesis"]),
                    "label": LABEL2ID[row["label"]],
                }
            )
    if not rows:
        raise ValueError("No test examples found")
    return rows


def expand_triplets(batch):
    premises = []
    hypotheses = []
    labels = []
    for anchor, positive, negative in zip(
        batch["anchor"], batch["positive"], batch["negative"]
    ):
        premises.extend((anchor, anchor))
        hypotheses.extend((positive, negative))
        labels.extend((LABEL2ID["entailment"], LABEL2ID["not_entailment"]))
    return {"premise": premises, "hypothesis": hypotheses, "label": labels}


def metric_dict(eval_prediction):
    import numpy as np

    predictions = np.argmax(eval_prediction.predictions, axis=-1)
    labels = eval_prediction.label_ids
    return {"accuracy": float((predictions == labels).mean())}


def main():
    args = parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("Choose at most one of --bf16 and --fp16")
    try:
        import datasets
        import numpy as np
        import torch
        import transformers
        from datasets import Dataset, load_dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "Install torch, transformers, datasets, accelerate, and numpy first."
        ) from exc

    set_seed(args.seed)
    test_rows = load_test_rows(args.test_data)
    raw = load_dataset(args.dataset)
    validation_name = "validation" if "validation" in raw else "dev" if "dev" in raw else None
    if validation_name is None:
        raise ValueError(f"Dataset has no validation/dev split; found {list(raw)}")
    required = {"anchor", "positive", "negative"}
    for split_name in ("train", validation_name):
        missing = required - set(raw[split_name].column_names)
        if missing:
            raise ValueError(f"{split_name} is missing {sorted(missing)}")

    train_triplets = raw["train"]
    validation_triplets = raw[validation_name]
    if args.max_train_triplets is not None:
        train_triplets = train_triplets.select(
            range(min(args.max_train_triplets, len(train_triplets)))
        )
    if args.max_validation_triplets is not None:
        validation_triplets = validation_triplets.select(
            range(min(args.max_validation_triplets, len(validation_triplets)))
        )
    cache_dir = args.output_dir / "dataset_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_examples = train_triplets.map(
        expand_triplets,
        batched=True,
        remove_columns=train_triplets.column_names,
        cache_file_name=str(cache_dir / "train-expanded.arrow"),
        desc="Expanding train triplets",
    )
    validation_examples = validation_triplets.map(
        expand_triplets,
        batched=True,
        remove_columns=validation_triplets.column_names,
        cache_file_name=str(cache_dir / "validation-expanded.arrow"),
        desc="Expanding validation triplets",
    )
    test_examples = Dataset.from_list(test_rows)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def tokenize(batch):
        return tokenizer(
            batch["premise"],
            batch["hypothesis"],
            truncation=True,
            max_length=args.max_length,
        )

    train_tokenized = train_examples.map(
        tokenize,
        batched=True,
        cache_file_name=str(cache_dir / "train-tokenized.arrow"),
        desc="Tokenizing train",
    )
    validation_tokenized = validation_examples.map(
        tokenize,
        batched=True,
        cache_file_name=str(cache_dir / "validation-tokenized.arrow"),
        desc="Tokenizing validation",
    )
    test_tokenized = test_examples.map(
        tokenize,
        batched=True,
        cache_file_name=str(cache_dir / "test-tokenized.arrow"),
        desc="Tokenizing test",
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    training_kwargs = {
        "output_dir": str(args.output_dir / "checkpoints"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 500,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "save_total_limit": 2,
        "seed": args.seed,
        "data_seed": args.seed,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "report_to": [],
    }
    parameter_names = inspect.signature(TrainingArguments.__init__).parameters
    if "warmup_ratio" in parameter_names:
        training_kwargs["warmup_ratio"] = args.warmup_ratio
        warmup_config = {"warmup_ratio": args.warmup_ratio}
    elif "warmup_steps" in parameter_names:
        updates_per_epoch = math.ceil(
            len(train_tokenized)
            / (args.batch_size * args.gradient_accumulation_steps)
        )
        warmup_steps = round(updates_per_epoch * args.epochs * args.warmup_ratio)
        training_kwargs["warmup_steps"] = warmup_steps
        warmup_config = {"warmup_steps": warmup_steps}
    else:
        warmup_config = {"warmup": "unsupported by installed Transformers"}
    if "eval_strategy" in parameter_names:
        training_kwargs["eval_strategy"] = "epoch"
    else:
        training_kwargs["evaluation_strategy"] = "epoch"
    training_args = TrainingArguments(**training_kwargs)
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_tokenized,
        "eval_dataset": validation_tokenized,
        "data_collator": DataCollatorWithPadding(tokenizer),
        "compute_metrics": metric_dict,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=2)],
    }
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir / "best_model"))
    tokenizer.save_pretrained(str(args.output_dir / "best_model"))

    prediction_output = trainer.predict(test_tokenized)
    predicted_ids = np.argmax(prediction_output.predictions, axis=-1)
    records = []
    for row, prediction in zip(test_rows, predicted_ids.tolist()):
        gold = ID2LABEL[row["label"]]
        predicted = ID2LABEL[int(prediction)]
        records.append(
            {
                "id": row["id"],
                "premise": row["premise"],
                "hypothesis": row["hypothesis"],
                "gold": gold,
                "prediction": predicted,
                "correct": gold == predicted,
            }
        )
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    correct = sum(record["correct"] for record in records)
    summary = {
        "evaluator_version": EVALUATOR_VERSION,
        "base_model": args.model,
        "dataset": args.dataset,
        "setting": "supervised full-model fine-tuning",
        "primary_metric": "accuracy",
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "epochs": args.epochs,
            "weight_decay": args.weight_decay,
            **warmup_config,
            "max_length": args.max_length,
            "seed": args.seed,
        },
        "source_train_triplets": len(train_triplets),
        "expanded_train_examples": len(train_examples),
        "source_validation_triplets": len(validation_triplets),
        "expanded_validation_examples": len(validation_examples),
        "test_samples": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "trainer_test_metrics": prediction_output.metrics,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "datasets_version": datasets.__version__,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
