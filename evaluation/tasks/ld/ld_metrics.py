#!/usr/bin/env python3
"""Shared data validation and dependency-free metrics for LD evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


LABEL_ID_TO_NAME = {0: "cantonese", 1: "mandarin", 2: "mixed"}
LABELS = tuple(LABEL_ID_TO_NAME.values())
LABEL_NAME_TO_ID = {name: idx for idx, name in LABEL_ID_TO_NAME.items()}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ld_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = {"id", "source_id", "sentence", "label"} - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            row_id = int(row["id"])
            if row_id in seen_ids:
                raise ValueError(f"{path}:{line_number} duplicate id: {row_id}")
            seen_ids.add(row_id)
            label_id = int(row["label"])
            if label_id not in LABEL_ID_TO_NAME:
                raise ValueError(f"{path}:{line_number} unknown label: {label_id}")
            if not isinstance(row["sentence"], str) or not row["sentence"].strip():
                raise ValueError(f"{path}:{line_number} has an empty sentence")
            row["id"] = row_id
            row["source_id"] = int(row["source_id"])
            row["label"] = label_id
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No records found in {path}")
    return rows


def classification_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    counts = {label: Counter() for label in LABELS}
    correct = 0
    invalid = 0
    confusion = {gold: {pred: 0 for pred in LABELS} for gold in LABELS}

    for row in rows:
        gold = str(row["gold"])
        prediction = row.get("prediction")
        if gold not in LABELS:
            raise ValueError(f"Unknown gold label: {gold}")
        if prediction not in LABELS:
            prediction = None
            invalid += 1
        else:
            confusion[gold][prediction] += 1
        correct += prediction == gold
        for label in LABELS:
            if gold == label and prediction == label:
                counts[label]["tp"] += 1
            elif gold != label and prediction == label:
                counts[label]["fp"] += 1
            elif gold == label and prediction != label:
                counts[label]["fn"] += 1

    report: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in LABELS:
        tp = counts[label]["tp"]
        fp = counts[label]["fp"]
        fn = counts[label]["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(row["gold"] == label for row in rows),
        }

    return {
        "samples": len(rows),
        "valid_predictions": len(rows) - invalid,
        "invalid_predictions": invalid,
        "invalid_output_rate": invalid / len(rows) if rows else None,
        "accuracy": correct / len(rows) if rows else None,
        "macro_f1": sum(f1_values) / len(f1_values) if rows else None,
        "per_class": report,
        "confusion_matrix": confusion,
    }
