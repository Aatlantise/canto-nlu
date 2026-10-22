#!/usr/bin/env python3
"""Combine XLM-R and Llama SA summaries into a compact Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlmr", type=Path, required=True)
    parser.add_argument("--llama", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def pct(value) -> str:
    return "n.a." if value is None else f"{100 * float(value):.2f}"


def main() -> None:
    args = parse_args()
    xlmr = json.loads(args.xlmr.read_text(encoding="utf-8"))
    llama = json.loads(args.llama.read_text(encoding="utf-8"))
    xmetrics = xlmr["test_metrics"]
    lines = [
        "# Cantonese sentiment results",
        "",
        "| Model | Protocol | Test n | Macro-F1 (%) | Accuracy (%) | Invalid output (%) |",
        "|---|---|---:|---:|---:|---:|",
        (
            f"| {xlmr['model']} | supervised fine-tuning | {xlmr['split_sizes']['test']} | "
            f"{pct(xmetrics.get('test_macro_f1'))} | {pct(xmetrics.get('test_accuracy'))} | 0.00 |"
        ),
        (
            f"| {llama['model']} | frozen zero-shot | {llama['samples']} | "
            f"{pct(llama.get('macro_f1'))} | {pct(llama.get('accuracy'))} | "
            f"{pct(llama.get('invalid_output_rate'))} |"
        ),
        "",
        "The protocols differ, so this table measures practical task performance under two stated regimes; "
        "it is not an architecture-only comparison.",
    ]
    rendered = "\n".join(lines) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
