#!/usr/bin/env python3
"""Verify that all open-weight models completed every full-test evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


MODELS = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "gemma-3-12b": "google/gemma-3-12b-it",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}
TASKS = {
    "sentiment": (Path("data/sentiment/test.jsonl"), 999),
    "ld": (Path("data/ld/lid_test.jsonl"), 6000),
    "pos": (Path("data/pos/pos_test.jsonl"), 101),
    "deps": (Path("data/deps/deps_test.jsonl"), 101),
    "wsd": (Path("data/wsd/test.jsonl"), 1646),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=Path("results/open_weight_full")
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("results/open_weight_full/manifest.csv")
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nonempty_line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def prediction_counts(path: Path) -> tuple[int, int]:
    rows = 0
    ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if "id" not in record:
                raise ValueError(f"Missing prediction id at {path}:{line_number}")
            rows += 1
            ids.add(str(record["id"]))
    return rows, len(ids)


def expected_tasks() -> dict[str, tuple[Path, int]]:
    tasks = dict(TASKS)
    nli_data = Path("data/nli/full_test.jsonl")
    nli_metadata = Path("data/nli/full_test.metadata.json")
    if not nli_data.exists() or not nli_metadata.exists():
        raise SystemExit("Export the NLI full test and metadata before verification.")
    metadata = read_json(nli_metadata)
    expected = int(metadata["output_rows"])
    if expected <= 100:
        raise SystemExit(f"NLI metadata contains only {expected} rows, not a full test.")
    if metadata.get("sha256") != sha256_file(nli_data):
        raise SystemExit("NLI full-test SHA-256 does not match its metadata.")
    tasks["nli"] = (nli_data, expected)
    return tasks


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    manifest_rows: list[dict[str, Any]] = []

    for task, (data_path, expected) in expected_tasks().items():
        if not data_path.exists():
            failures.append(f"missing dataset: {data_path}")
            continue
        actual_data_rows = nonempty_line_count(data_path)
        if actual_data_rows != expected:
            failures.append(
                f"{data_path}: expected {expected} dataset rows, found {actual_data_rows}"
            )
        data_sha256 = sha256_file(data_path)

        for model_key, model_id in MODELS.items():
            prediction_path = args.results_root / model_key / f"{task}.jsonl"
            summary_path = prediction_path.with_suffix(".summary.json")
            run_path = prediction_path.with_suffix(".run.json")
            prefix = f"{model_key}/{task}"
            if not prediction_path.exists() or not summary_path.exists() or not run_path.exists():
                failures.append(f"{prefix}: missing predictions, summary, or run metadata")
                continue

            summary = read_json(summary_path)
            run_metadata = read_json(run_path)
            prediction_rows, unique_prediction_ids = prediction_counts(prediction_path)
            checks = {
                "model": summary.get("model") == model_id,
                "task": summary.get("task") == task,
                "summary_samples": summary.get("samples") == expected,
                "prediction_rows": prediction_rows == expected,
                "unique_prediction_ids": unique_prediction_ids == expected,
                "data_sha256": summary.get("data_sha256") == data_sha256,
                "expected_samples": summary.get("expected_samples") == expected,
                "run_data_sha256": run_metadata.get("data_sha256") == data_sha256,
                "run_model": run_metadata.get("model") == model_id,
                "run_task": run_metadata.get("task") == task,
            }
            failed_checks = [name for name, passed in checks.items() if not passed]
            if failed_checks:
                failures.append(f"{prefix}: failed {', '.join(failed_checks)}")

            manifest_rows.append(
                {
                    "model_key": model_key,
                    "model_id": model_id,
                    "task": task,
                    "data": data_path,
                    "data_sha256": data_sha256,
                    "expected_samples": expected,
                    "actual_samples": summary.get("samples"),
                    "invalid_predictions": summary.get("invalid_predictions"),
                    "invalid_output_rate": summary.get("invalid_output_rate"),
                    "accuracy": summary.get("accuracy"),
                    "macro_f1": summary.get("macro_f1"),
                    "token_accuracy": summary.get("token_accuracy"),
                    "uas": summary.get("uas"),
                    "las": summary.get("las"),
                    "load_in_4bit": summary.get("load_in_4bit"),
                    "finished_utc": summary.get("finished_utc"),
                }
            )

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    if manifest_rows:
        with args.manifest.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)

    for row in manifest_rows:
        print(
            f"{row['model_key']:<14} {row['task']:<10} "
            f"{row['actual_samples']!s:>6}/{row['expected_samples']:<6} "
            f"invalid={row['invalid_predictions']}"
        )

    if failures:
        print("\nVerification failed:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print(f"\nVerified {len(manifest_rows)} complete full-test runs.")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
