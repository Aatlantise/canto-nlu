import csv
import json
import random
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def extract_balanced_samples(jsonl_path, csv_path, source_ids_csv=None, total_samples=99, seed=42):
    random.seed(seed)

    # Optionally restrict to the pairs used in the single-sentence LAJ sample,
    # so both annotation sets cover the same source sentences
    allowed_source_ids = None
    if source_ids_csv is not None:
        with open(source_ids_csv, 'r', encoding='utf-8-sig') as f:
            allowed_source_ids = {int(row['source_id']) for row in csv.DictReader(f)}

    # Group examples by their label (1: first sentence is the reference, 2: second is)
    class_data = defaultdict(list)
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if allowed_source_ids is None or item['source_id'] in allowed_source_ids:
                    class_data[item['label']].append(item)

    num_classes = len(class_data)
    if num_classes == 0:
        print("No valid records found in the JSONL file.")
        return

    # Distribute samples evenly across classes, handling any remainder
    base_per_class = total_samples // num_classes
    remainder = total_samples % num_classes

    selected_items = []
    for i, (label, items) in enumerate(sorted(class_data.items())):
        n_to_sample = base_per_class + (1 if i < remainder else 0)
        sampled = random.sample(items, min(n_to_sample, len(items)))
        selected_items.extend(sampled)

    # Shuffle the final dataset to avoid class-sequential ordering in the CSV
    random.shuffle(selected_items)

    if selected_items:
        # Blank columns for human annotators to fill out
        for item in selected_items:
            item['human_label'] = ''
            item['notes'] = ''

        fieldnames = ['id', 'source_id', 'sentence1', 'sentence2', 'label', 'human_label', 'notes']
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected_items)
        print(f"Successfully extracted {len(selected_items)} class-balanced examples to {csv_path}")
    else:
        print("No items were selected.")


extract_balanced_samples(
    SCRIPT_DIR / 'comparison_test.jsonl',
    SCRIPT_DIR / 'sample_comparison_laj.csv',
    source_ids_csv=SCRIPT_DIR.parent / 'laj_finetune' / 'laj_sample.csv',
    total_samples=99,
)
