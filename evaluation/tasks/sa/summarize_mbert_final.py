#!/usr/bin/env python3
"""Summarize fixed-hyperparameter mBERT test runs across random seeds."""

import json
import statistics
from pathlib import Path

root = Path("/home/USER/mbert-final")
rows = []
for path in sorted(root.glob("seed*/summary.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    hp = data["hyperparameters"]
    tm = data["test_metrics"]
    rows.append({
        "seed": hp["seed"],
        "accuracy": tm["test_accuracy"],
        "macro_f1": tm["test_macro_f1"],
        "loss": tm["test_loss"],
        "train_seconds": data["train_seconds"],
    })

if not rows:
    raise SystemExit(f"No summary.json files found under {root}")

print("seed\ttest_accuracy\ttest_macro_f1\ttest_loss\ttrain_seconds")
for row in rows:
    print(
        f'{row["seed"]}\t{row["accuracy"]:.6f}\t{row["macro_f1"]:.6f}\t'
        f'{row["loss"]:.6f}\t{row["train_seconds"]:.1f}'
    )

def stats(key):
    values = [row[key] for row in rows]
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }

result = {
    "fixed_hyperparameters": {
        "learning_rate": 2e-5,
        "batch_size": 32,
        "epochs": 3,
        "weight_decay": 0.01,
        "max_length": 128,
    },
    "test_accuracy": stats("accuracy"),
    "test_macro_f1": stats("macro_f1"),
    "runs": rows,
}
print("\nFINAL SUMMARY")
print(json.dumps(result, indent=2))
(root / "aggregate_summary.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8"
)
