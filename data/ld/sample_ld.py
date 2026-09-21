import json
import random
from collections import defaultdict
import csv


def extract_balanced_samples(jsonl_path, csv_path, total_samples=101):
    # Group examples by their label
    class_data = defaultdict(list)
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                class_data[item['label']].append(item)

    num_classes = len(class_data)
    if num_classes == 0:
        print("No valid records found in the JSONL file.")
        return

    # Distribute samples evenly across classes, handling any remainder
    base_per_class = total_samples // num_classes
    remainder = total_samples % num_classes

    selected_items = []
    for i, (label, items) in enumerate(class_data.items()):
        # Assign an extra sample to the first few classes if total doesn't divide evenly
        n_to_sample = base_per_class + (1 if i < remainder else 0)

        # Sample randomly (caps at total available items if fewer exist)
        sampled = random.sample(items, min(n_to_sample, len(items)))
        selected_items.extend(sampled)

    # Shuffle the final dataset to avoid class-sequential ordering in the CSV
    random.shuffle(selected_items)

    # Write the selected samples to a CSV file
    if selected_items:
        fieldnames = list(selected_items[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selected_items)
        print(f"Successfully extracted {len(selected_items)} class-balanced examples to {csv_path}")
    else:
        print("No items were selected.")

# Example execution:
extract_balanced_samples('ld_test.jsonl', 'ld.csv', total_samples=101)