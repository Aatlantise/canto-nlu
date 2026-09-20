import argparse
import json
import os
import random
import sys
from collections import Counter

def analyze(path):
    counter = Counter()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            counter[obj["label"]] += 1

    total = counter[0] + counter[1]

    print(f"\n{path}")
    print(f"Total: {total}")
    print(f"acceptable (1): {counter[1]} ({counter[1]/total:.2%})")
    print(f"unacceptable (0): {counter[0]} ({counter[0]/total:.2%})")


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)
    p = argparse.ArgumentParser(description="Split laj_dataset.jsonl by source_id")
    p.add_argument("--input", default=os.path.join(parent, "laj_dataset.jsonl"))
    p.add_argument("--train_out", default=os.path.join(here, "finetune_train.jsonl"))
    p.add_argument("--test_out", default=os.path.join(here, "finetune_test.jsonl"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_frac", type=float, default=0.9)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    groups = {}  
    total_examples = 0

    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(2)

    with open(args.input, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            total_examples += 1
            try:
                obj = json.loads(raw)
            except Exception as e:
                print(f"Failed to parse JSON on line {lineno}: {e}", file=sys.stderr)
                sys.exit(3)
            if "source_id" not in obj:
                print(f"Missing 'source_id' on line {lineno}", file=sys.stderr)
                sys.exit(4)
            sid = obj["source_id"]
            groups.setdefault(sid, []).append(raw)

    source_ids = list(groups.keys())
    random.shuffle(source_ids)
    n_source = len(source_ids)
    n_train = int(n_source * args.train_frac)
    train_ids = set(source_ids[:n_train])
    test_ids = set(source_ids[n_train:])

    train_examples = 0
    test_examples = 0

    with open(args.train_out, "w", encoding="utf-8") as tout:
        for sid in source_ids:
            if sid in train_ids:
                for raw in groups[sid]:
                    tout.write(raw + "\n")
                    train_examples += 1

    with open(args.test_out, "w", encoding="utf-8") as tout:
        for sid in source_ids:
            if sid in test_ids:
                for raw in groups[sid]:
                    tout.write(raw + "\n")
                    test_examples += 1

    print(f"total_examples: {total_examples}")
    print(f"unique_source_ids: {n_source}")
    print(f"train_examples: {train_examples}")
    print(f"test_examples: {test_examples}")
    print(f"train_source_ids: {len(train_ids)}")
    print(f"test_source_ids: {len(test_ids)}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    analyze(os.path.join(script_dir, "finetune_train.jsonl"))
    analyze(os.path.join(script_dir, "finetune_test.jsonl"))


if __name__ == "__main__":
    main()


