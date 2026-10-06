#!/usr/bin/env python3
"""Reparse existing stepwise raw answers without another model invocation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deps_stepwise_common import calculate_stepwise_metrics, parse_stepwise_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluator-version", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with input_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as destination:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            prediction, status, details = parse_stepwise_answer(
                record["raw_answer"], record["tokens"]
            )
            rescored = {
                **record,
                "previous_parse_status": record.get("parse_status"),
                "prediction": prediction,
                "parse_status": status,
                "parse_details": details,
                "evaluator_version": args.evaluator_version,
            }
            destination.write(json.dumps(rescored, ensure_ascii=False) + "\n")
            records.append(rescored)

    if not records:
        raise RuntimeError("No prediction records found")
    summary = {
        "requested_model": records[0].get("requested_model") or records[0].get("model"),
        "setting": records[0].get("setting"),
        "evaluator_version": args.evaluator_version,
        "source_predictions": str(input_path),
        "rescore_note": (
            "No model calls. Raw answers reparsed by ID+exact FORM with tolerant "
            "recovery of over-complete TSV rows; strict-format metrics are retained."
        ),
        "metrics_policy": (
            "Primary UAS/LAS include punctuation. Missing or malformed token rows "
            "score as incorrect only for those tokens."
        ),
        **calculate_stepwise_metrics(records),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Rescored predictions: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
