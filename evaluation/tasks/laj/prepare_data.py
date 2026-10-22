#!/usr/bin/env python3
"""Download and prepare the Cantonese LAJ evaluation datasets."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from build_comparison_dataset import build_comparison_dataset
from build_laj_dataset import DEFAULT_CONFIG, build_laj_dataset


SOURCE_URL = (
    "https://raw.githubusercontent.com/"
    "hannliu/SiniticMTError/main/cantonese.jsonl"
)
HERE = Path(__file__).resolve().parent
SOURCE_PATH = HERE / "cantonese.jsonl"
SINGLE_SENTENCE_PATH = HERE / "laj_dataset.jsonl"
COMPARISON_PATH = HERE / "comparison_test.jsonl"


def download_source() -> None:
    if SOURCE_PATH.exists():
        print(f"Using existing source file: {SOURCE_PATH}")
        return

    print(f"Downloading {SOURCE_URL}")
    request = urllib.request.Request(
        SOURCE_URL,
        headers={"User-Agent": "cantonese-nlu-evaluation"},
    )

    with urllib.request.urlopen(request, timeout=120) as response:
        SOURCE_PATH.write_bytes(response.read())

    print(f"Downloaded: {SOURCE_PATH}")


def main() -> None:
    download_source()

    build_laj_dataset(
        str(SOURCE_PATH),
        str(SINGLE_SENTENCE_PATH),
        DEFAULT_CONFIG,
    )

    build_comparison_dataset(
        SINGLE_SENTENCE_PATH,
        COMPARISON_PATH,
        seed=42,
    )

    print("LAJ preparation complete.")
    print(f"Source: {SOURCE_PATH}")
    print(f"Single-sentence dataset: {SINGLE_SENTENCE_PATH}")
    print(f"Comparison dataset: {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
