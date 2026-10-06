#!/usr/bin/env python3
"""Collect NLI summary files into a compact CSV and Markdown table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DISPLAY_NAMES = {
    "google-bert/bert-base-multilingual-cased": "mBERT",
    "FacebookAI/xlm-roberta-base": "XLM-R-base",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct",
    "google/gemma-3-12b-it": "Gemma-3-12B-it",
    "deepseek-v4-flash": "DeepSeek-V4-Flash",
    "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    paths = sorted(
        set(args.results_root.rglob("summary.json"))
        | set(args.results_root.rglob("*.summary.json"))
    )
    rows = []
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not str(summary.get("evaluator_version", "")).startswith("cantonese-nli-"):
            continue
        model_id = summary.get("model") or summary.get("base_model") or summary.get(
            "requested_model"
        )
        if not model_id or summary.get("accuracy") is None:
            continue
        samples = summary.get("samples", summary.get("test_samples"))
        rows.append(
            {
                "model": DISPLAY_NAMES.get(model_id, model_id),
                "model_id": model_id,
                "setting": summary.get("setting"),
                "samples": samples,
                "correct": summary.get("correct"),
                "accuracy": summary["accuracy"],
                "accuracy_percent": 100 * summary["accuracy"],
                "invalid_output_rate": summary.get("invalid_output_rate", 0.0),
                "summary_file": str(path),
            }
        )
    order = {name: index for index, name in enumerate(DISPLAY_NAMES.values())}
    rows.sort(key=lambda row: order.get(row["model"], len(order)))
    if not rows:
        raise SystemExit(f"No completed NLI summaries found under {args.results_root}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "| Model | Setting | N | Correct | NLI Acc. | Invalid |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['setting']} | {row['samples']} | "
            f"{row['correct']} | {row['accuracy_percent']:.2f} | "
            f"{100 * row['invalid_output_rate']:.2f}% |"
        )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
