#!/usr/bin/env python3
"""Export the complete Hugging Face NLI test split as binary JSONL pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="hon9kon9ize/yue-all-nli")
    parser.add_argument("--split", default="test")
    parser.add_argument("--revision", help="Optional Hugging Face dataset revision.")
    parser.add_argument("--output", type=Path, default=here / "full_test.jsonl")
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=here / "full_test.metadata.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_present(row: dict[str, Any], names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in row:
            return name
    return None


def unroll(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for index, row in enumerate(rows):
        premise_col = first_present(row, ("anchor", "premise"))
        positive_col = first_present(row, ("positive", "entailment"))
        negative_col = first_present(row, ("negative", "contradiction"))

        if premise_col and positive_col and negative_col:
            yield {
                "id": f"{index}_ent",
                "premise": row[premise_col],
                "hypothesis": row[positive_col],
                "label": "entailment",
            }
            yield {
                "id": f"{index}_non_ent",
                "premise": row[premise_col],
                "hypothesis": row[negative_col],
                "label": "not_entailment",
            }
            continue

        if {"premise", "hypothesis", "label"} <= row.keys():
            raw_label = row["label"]
            is_entailment = raw_label in (0, "0", "entailment")
            yield {
                "id": str(index),
                "premise": row["premise"],
                "hypothesis": row["hypothesis"],
                "label": "entailment" if is_entailment else "not_entailment",
            }
            continue

        raise ValueError(
            "Unsupported NLI schema. Expected triplets with anchor/premise, "
            "positive/entailment, and negative/contradiction, or standard "
            "premise/hypothesis/label rows. "
            f"Found columns: {sorted(row)}"
        )


def main() -> None:
    args = parse_args()
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install the repository requirements before exporting NLI.") from exc

    load_kwargs: dict[str, Any] = {"split": args.split}
    if args.revision:
        load_kwargs["revision"] = args.revision
    source = load_dataset(args.dataset, **load_kwargs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    labels: Counter[str] = Counter()
    seen_ids: set[str] = set()
    output_rows = 0
    with args.output.open("w", encoding="utf-8") as stream:
        for record in unroll(source):
            record_id = str(record["id"])
            if record_id in seen_ids:
                raise ValueError(f"Duplicate generated NLI id: {record_id}")
            seen_ids.add(record_id)
            labels[record["label"]] += 1
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_rows += 1

    if output_rows <= 100:
        raise ValueError(
            f"The exported split contains only {output_rows} examples; refusing to "
            "treat it as the full test set."
        )

    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "requested_revision": args.revision,
        "dataset_fingerprint": getattr(source, "_fingerprint", None),
        "source_rows": len(source),
        "output_rows": output_rows,
        "labels": dict(sorted(labels.items())),
        "output": str(args.output),
        "sha256": sha256_file(args.output),
    }
    args.metadata_output.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
