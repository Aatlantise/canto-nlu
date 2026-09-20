import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def load_laj_dataset(input_path: Path) -> Dict[int, List[Dict[str, Any]]]:
    records_by_source: Dict[int, List[Dict[str, Any]]] = {}

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            source_id = record["source_id"]
            records_by_source.setdefault(source_id, []).append(record)

    return records_by_source


def validate_pair(records: List[Dict[str, Any]], source_id: int) -> None:
    if len(records) != 2:
        raise ValueError(
            f"Expected exactly 2 records for source_id {source_id}, found {len(records)}"
        )

    if records[0]["source_id"] != source_id or records[1]["source_id"] != source_id:
        raise ValueError(
            f"Mismatched source_id for pair {source_id}: "
            f"{records[0]['source_id']} vs {records[1]['source_id']}"
        )


def make_comparison_example(
    example_id: int,
    source_id: int,
    sentence1: str,
    sentence2: str,
    label: int,
) -> Dict[str, Any]:
    return {
        "id": example_id,
        "source_id": source_id,
        "sentence1": sentence1,
        "sentence2": sentence2,
        "label": label,
    }


def build_comparison_dataset(
    input_path: Path,
    output_path: Path,
    seed: int = 42,
) -> None:
    random.seed(seed)
    pairs = load_laj_dataset(input_path)

    total_pairs = len(pairs)
    label1_count = 0
    label2_count = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f_out:
        for example_id, source_id in enumerate(sorted(pairs)):
            ref_record, mt_record = pairs[source_id]
            validate_pair([ref_record, mt_record], source_id)

            if random.random() < 0.5:
                sentence1 = ref_record["sentence"]
                sentence2 = mt_record["sentence"]
                label = 1
            else:
                sentence1 = mt_record["sentence"]
                sentence2 = ref_record["sentence"]
                label = 2

            if label == 1:
                label1_count += 1
            else:
                label2_count += 1

            comparison_example = make_comparison_example(
                example_id,
                source_id,
                sentence1,
                sentence2,
                label,
            )
            f_out.write(json.dumps(comparison_example, ensure_ascii=False) + "\n")

    assert label1_count + label2_count == total_pairs, (
        f"Label counts do not sum to total pairs: {label1_count} + {label2_count} != {total_pairs}"
    )

    print(f"total number of pairs: {total_pairs}")
    print(f"label 1 count: {label1_count}")
    print(f"label 2 count: {label2_count}")


def main():
    script_dir = Path(__file__).resolve().parent
    parent_dir = script_dir.parent
    parser = argparse.ArgumentParser(
        description="Build a comparison dataset from laj_dataset.jsonl."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=parent_dir / "laj_dataset.jsonl",
        help="Path to the input LAJ dataset JSONL file (default: parent directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "comparison_test.jsonl",
        help="Path to the output comparison JSONL file (default: same folder as script).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible pair ordering.",
    )
    args = parser.parse_args()

    build_comparison_dataset(args.input, args.output, seed=args.seed)

if __name__ == "__main__":
    main()
