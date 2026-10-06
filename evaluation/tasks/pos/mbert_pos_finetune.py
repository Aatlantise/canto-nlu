#!/usr/bin/env python3
"""Fine-tune mBERT for Cantonese UPOS tagging without test leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from datasets import Dataset
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)


LABELS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "VERB",
]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}
PUNCT_ID = LABEL2ID["PUNCT"]
EVALUATOR_VERSION = "mbert-pos-supervised-v3-empty-wordpiece-safe"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="google-bert/bert-base-multilingual-cased")
    parser.add_argument("--evaluator-version", default=EVALUATOR_VERSION)
    parser.add_argument("--epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    return parser.parse_args()


def load_rows(path):
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    seen = set()
    for row in rows:
        if not {"id", "tokens", "upos"}.issubset(row):
            raise ValueError("Each row requires id, tokens, and upos")
        row["id"] = int(row["id"])
        if row["id"] in seen:
            raise ValueError(f"Duplicate id: {row['id']}")
        seen.add(row["id"])
        if len(row["tokens"]) != len(row["upos"]):
            raise ValueError(f"Length mismatch in id {row['id']}")
        unknown = sorted(set(row["upos"]) - set(LABELS))
        if unknown:
            raise ValueError(f"Unknown tags in id {row['id']}: {unknown}")
    return rows


def token_key(row):
    return "\x1f".join(row["tokens"])


def split_train_validation(rows, seed, validation_ratio):
    if not 0.0 < validation_ratio < 1.0:
        raise ValueError("validation_ratio must be between 0 and 1")
    # Exact duplicate sentences stay in the same split, preventing leakage.
    groups = [token_key(row) for row in rows]
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=validation_ratio, random_state=seed
    )
    train_indices, validation_indices = next(splitter.split(rows, groups=groups))
    return {
        "train": [rows[index] for index in train_indices],
        "validation": [rows[index] for index in validation_indices],
    }


def as_dataset(rows):
    return Dataset.from_dict(
        {
            "id": [row["id"] for row in rows],
            "tokens": [row["tokens"] for row in rows],
            "upos": [row["upos"] for row in rows],
        }
    )


def metric_values(logits, label_ids):
    pred_ids = np.argmax(logits, axis=-1)
    flat_gold, flat_pred = [], []
    exact = []
    for gold_row, pred_row in zip(label_ids, pred_ids):
        mask = gold_row != -100
        gold = gold_row[mask]
        pred = pred_row[mask]
        flat_gold.extend(gold.tolist())
        flat_pred.extend(pred.tolist())
        exact.append(bool(len(gold)) and bool(np.array_equal(gold, pred)))
    flat_gold = np.asarray(flat_gold)
    flat_pred = np.asarray(flat_pred)
    non_punct = flat_gold != PUNCT_ID
    return {
        "token_accuracy": float(np.mean(flat_gold == flat_pred)),
        "non_punct_accuracy": float(
            np.mean(flat_gold[non_punct] == flat_pred[non_punct])
        ),
        "macro_f1": float(
            f1_score(
                flat_gold,
                flat_pred,
                labels=list(range(len(LABELS))),
                average="macro",
                zero_division=0,
            )
        ),
        "sentence_exact_match": float(np.mean(exact)),
        "tokens": int(len(flat_gold)),
        "non_punct_tokens": int(np.sum(non_punct)),
    }


def main():
    args = parse_args()
    wall_started = time.time()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_rows(args.train_data)
    test_rows = load_rows(args.test_data)
    train_keys = {token_key(row) for row in train_rows}
    overlap = [row["id"] for row in test_rows if token_key(row) in train_keys]
    if overlap:
        raise ValueError(
            f"Train/test leakage: {len(overlap)} test rows duplicate train tokens"
        )
    split = split_train_validation(
        train_rows, args.seed, args.validation_ratio
    )
    split["test"] = test_rows
    full_split_ids = {name: [row["id"] for row in values] for name, values in split.items()}
    manifest = {
        "evaluator_version": args.evaluator_version,
        "seed": args.seed,
        "protocol": (
            "The supplied train file is split into train/validation with "
            "GroupShuffleSplit, grouping exact token sequences to prevent duplicate "
            "leakage. The supplied test file remains untouched and is evaluated once."
        ),
        "validation_ratio": args.validation_ratio,
        "sizes": {name: len(values) for name, values in split.items()},
        "ids": full_split_ids,
        "source_sha256": {
            "train": hashlib.sha256(args.train_data.read_bytes()).hexdigest(),
            "test": hashlib.sha256(args.test_data.read_bytes()).hexdigest(),
        },
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    run_split = {name: list(values) for name, values in split.items()}
    if args.max_train_samples is not None:
        run_split["train"] = run_split["train"][: args.max_train_samples]
    if args.max_eval_samples is not None:
        run_split["validation"] = run_split["validation"][: args.max_eval_samples]
        run_split["test"] = run_split["test"][: args.max_eval_samples]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def tokenize_and_align(batch):
        # BERT's BasicTokenizer removes some control/private-use characters.
        # Such a pre-tokenized word would otherwise produce zero WordPieces and
        # break the one-label-per-original-token alignment. Represent it as UNK.
        safe_tokens = [
            [
                token if tokenizer.tokenize(token) else tokenizer.unk_token
                for token in sentence
            ]
            for sentence in batch["tokens"]
        ]
        encoded = tokenizer(
            safe_tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=args.max_length,
        )
        aligned = []
        for index, tags in enumerate(batch["upos"]):
            word_ids = encoded.word_ids(batch_index=index)
            covered = {word_id for word_id in word_ids if word_id is not None}
            missing = sorted(set(range(len(tags))) - covered)
            if missing:
                raise ValueError(
                    f"max_length={args.max_length} omitted word indexes {missing} "
                    f"from a sentence with {len(tags)} tokens"
                )
            labels, previous = [], None
            for word_id in word_ids:
                if word_id is None or word_id == previous:
                    labels.append(-100)
                else:
                    labels.append(LABEL2ID[tags[word_id]])
                previous = word_id
            aligned.append(labels)
        encoded["labels"] = aligned
        return encoded

    raw_datasets = {name: as_dataset(values) for name, values in run_split.items()}
    encoded = {
        name: dataset.map(
            tokenize_and_align,
            batched=True,
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {name}",
        )
        for name, dataset in raw_datasets.items()
    }
    model = AutoModelForTokenClassification.from_pretrained(
        args.model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    training_args = TrainingArguments(
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
        logging_steps=10,
        seed=args.seed,
        data_seed=args.seed,
    )

    def compute_metrics(eval_pred):
        return metric_values(eval_pred.predictions, eval_pred.label_ids)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        compute_metrics=compute_metrics,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    run_started = time.time()
    train_result = trainer.train()
    train_seconds = time.time() - run_started
    validation_metrics = trainer.evaluate(
        encoded["validation"], metric_key_prefix="validation"
    )
    test_started = time.time()
    prediction = trainer.predict(encoded["test"], metric_key_prefix="test")
    test_seconds = time.time() - test_started
    test_values = metric_values(prediction.predictions, prediction.label_ids)
    pred_ids = np.argmax(prediction.predictions, axis=-1)

    prediction_rows = []
    flat_gold, flat_pred = [], []
    for row, gold_row, pred_row in zip(
        run_split["test"], prediction.label_ids, pred_ids
    ):
        mask = gold_row != -100
        gold_ids = gold_row[mask].tolist()
        predicted_ids = pred_row[mask].tolist()
        gold_tags = [ID2LABEL[index] for index in gold_ids]
        predicted_tags = [ID2LABEL[index] for index in predicted_ids]
        if len(predicted_tags) != len(row["tokens"]):
            raise RuntimeError(f"Prediction alignment failed for id {row['id']}")
        flat_gold.extend(gold_ids)
        flat_pred.extend(predicted_ids)
        prediction_rows.append(
            {
                "id": row["id"],
                "tokens": row["tokens"],
                "upos": gold_tags,
                "prediction": predicted_tags,
                "correct": gold_tags == predicted_tags,
            }
        )
    with (args.output_dir / "test_predictions.jsonl").open(
        "w", encoding="utf-8"
    ) as stream:
        for row in prediction_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    per_label = classification_report(
        flat_gold,
        flat_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    summary = {
        "model": args.model,
        "evaluator_version": args.evaluator_version,
        "protocol": (
            "supervised fine-tuning on train; best checkpoint selected by validation "
            "macro-F1; test evaluated once"
        ),
        "split_sizes": {name: len(values) for name, values in run_split.items()},
        "full_split_sizes": {name: len(values) for name, values in split.items()},
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_length": args.max_length,
            "seed": args.seed,
            "validation_ratio": args.validation_ratio,
        },
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_macro_f1": trainer.state.best_metric,
        "validation_metrics": {
            key: float(value) for key, value in validation_metrics.items()
        },
        "test_metrics": test_values,
        "invalid_outputs": 0,
        "invalid_output_rate": 0.0,
        "per_label": per_label,
        "train_seconds": train_seconds,
        "test_seconds": test_seconds,
        "train_and_final_eval_seconds": time.time() - run_started,
        "wall_time_seconds": time.time() - wall_started,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "peak_gpu_memory_gib": (
            torch.cuda.max_memory_allocated() / 1024**3
            if torch.cuda.is_available()
            else None
        ),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "train_metrics": {
            key: float(value) for key, value in train_result.metrics.items()
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    trainer.save_model(args.output_dir / "best_model")
    tokenizer.save_pretrained(args.output_dir / "best_model")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
