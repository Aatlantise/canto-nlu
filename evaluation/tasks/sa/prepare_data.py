#!/usr/bin/env python3
"""Download and prepare the OpenRice Cantonese sentiment dataset."""

from __future__ import annotations

import csv
import json
import urllib.request
from pathlib import Path


UPSTREAM_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "toastynews/openrice-senti/master"
)
SPLITS = ("train", "valid", "test")
LABELS = {"smile", "ok", "cry"}
OUTPUT_DIR = Path(__file__).resolve().parent


def download_tsv(split: str) -> Path:
    output_path = OUTPUT_DIR / f"{split}.tsv"

    if output_path.exists():
        print(f"Using existing file: {output_path}")
        return output_path

    url = f"{UPSTREAM_BASE_URL}/{split}.tsv"
    print(f"Downloading {url}")

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "cantonese-nlu-evaluation"},
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        output_path.write_bytes(response.read())

    print(f"Downloaded: {output_path}")
    return output_path


def convert_tsv_to_jsonl(input_path: Path) -> Path:
    output_path = input_path.with_suffix(".jsonl")
    count = 0

    with input_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as input_stream, output_path.open(
        "w",
        encoding="utf-8",
    ) as output_stream:
        reader = csv.DictReader(input_stream, delimiter="\t")

        if not reader.fieldnames:
            raise ValueError(f"Missing header in {input_path}")

        required_columns = {"label", "text_a"}
        if not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"Expected columns {sorted(required_columns)}; "
                f"found {reader.fieldnames}"
            )

        for row_id, row in enumerate(reader):
            label = row["label"].strip()
            sentence = row["text_a"].strip()

            if label not in LABELS:
                raise ValueError(
                    f"Unknown label {label!r} at row {row_id}"
                )

            record = {
                "id": row_id,
                "sentence": sentence,
                "label": label,
                "source": "openrice-senti",
            }

            output_stream.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            count += 1

    print(f"Wrote {count} examples to {output_path}")
    return output_path


def main() -> None:
    for split in SPLITS:
        tsv_path = download_tsv(split)
        convert_tsv_to_jsonl(tsv_path)


if __name__ == "__main__":
    main()
