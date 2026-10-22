#!/usr/bin/env python3
"""Summarize mBERT validation-only search runs and identify the best config."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search-dir", type=Path, default=Path("/home/USER/mbert-search"))
    args = parser.parse_args()
    rows = []
    for path in args.search_dir.glob("*/search_summary.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        hp = data["hyperparameters"]
        vm = data["validation_metrics"]
        rows.append({
            "run": path.parent.name,
            "learning_rate": hp["learning_rate"],
            "batch_size": hp["batch_size"],
            "epochs": hp["epochs"],
            "valid_macro_f1": vm["valid_macro_f1"],
            "valid_accuracy": vm["valid_accuracy"],
            "valid_loss": vm["valid_loss"],
            "train_seconds": data["train_seconds"],
        })
    if not rows:
        raise SystemExit(f"No search_summary.json files found under {args.search_dir}")
    rows.sort(key=lambda x: (-x["valid_macro_f1"], x["valid_loss"]))
    print("run\tlr\tbatch\tepochs\tvalid_macro_f1\tvalid_accuracy\tvalid_loss\ttrain_seconds")
    for x in rows:
        print(
            f'{x["run"]}\t{x["learning_rate"]}\t{x["batch_size"]}\t{x["epochs"]}\t'
            f'{x["valid_macro_f1"]:.6f}\t{x["valid_accuracy"]:.6f}\t'
            f'{x["valid_loss"]:.6f}\t{x["train_seconds"]:.1f}'
        )
    print("\nBEST CONFIGURATION")
    print(json.dumps(rows[0], indent=2))


if __name__ == "__main__":
    main()
