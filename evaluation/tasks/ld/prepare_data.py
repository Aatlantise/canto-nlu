"""Download and verify the canonical CantoNLU language-detection splits."""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path


UPSTREAM_COMMIT = "ad1d74416891b866b1c26221d4b270e52e9dc9e5"
BASE_URL = (
    "https://raw.githubusercontent.com/Aatlantise/canto-nlu/"
    f"{UPSTREAM_COMMIT}/data/ld"
)

SPLITS = {
    "ld_train.jsonl": {
        "rows": 42_753,
        "sha256": "3bb990ebc17927290c516d8feaaeab58bc5576c70a83fe1d9d4da226814034a4",
    },
    "ld_val.jsonl": {
        "rows": 2_425,
        "sha256": "d6fdd177a5801d91920cba141b755744aeed2025f4255bf31d360e57d9adee83",
    },
    "ld_test.jsonl": {
        "rows": 2_341,
        "sha256": "3220bc2ab99c66426c67140dc9685db95dea1ae4cd14e8cd2de227343682f020",
    },
}


def inspect_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    rows = 0
    with path.open("rb") as handle:
        for line in handle:
            digest.update(line)
            rows += 1
    return digest.hexdigest(), rows


def verify_file(path: Path, expected_sha256: str, expected_rows: int) -> None:
    actual_sha256, actual_rows = inspect_file(path)
    if actual_sha256 != expected_sha256 or actual_rows != expected_rows:
        raise RuntimeError(
            f"Verification failed for {path}: expected {expected_rows} rows and "
            f"SHA-256 {expected_sha256}, got {actual_rows} rows and {actual_sha256}."
        )


def download_file(name: str, output_dir: Path, force: bool) -> None:
    metadata = SPLITS[name]
    destination = output_dir / name

    if destination.exists() and not force:
        verify_file(destination, metadata["sha256"], metadata["rows"])
        print(f"Verified existing file: {destination}")
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    url = f"{BASE_URL}/{name}"
    print(f"Downloading {url}")

    try:
        with urllib.request.urlopen(url) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        verify_file(temporary, metadata["sha256"], metadata["rows"])
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    print(f"Wrote and verified: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the fixed CantoNLU LD train/validation/test splits."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Destination directory (default: tasks/ld/data).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing files before verifying them.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLITS:
        download_file(name, output_dir, args.force)

    print("All canonical LD splits are ready.")


if __name__ == "__main__":
    main()
