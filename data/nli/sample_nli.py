"""Draw a balanced 100-example NLI sample (for the human baseline and LLM evaluation)
from the test split of Canto-XNLI (aatlantise/canto-xnli on Hugging Face).

As in fine-tuning (main.py), `neutral` pairs are dropped and the task is binary:
`entailment` vs `not_entailment` (= contradiction). The sample is 50 + 50, shuffled,
with a fixed seed. `id` is the XNLI pairID, which does not reveal the label.
"""
import csv
import json
import random
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).resolve().parent
# Same pinned commit as download.py, so the sample is reproducible.
CANTO_XNLI_ID = "aatlantise/canto-xnli"
CANTO_XNLI_REVISION = "a4146b47902ee02cb1abc5d78879173389ce6fdf"
SEED = 42
N_PER_LABEL = 50

rows = []
for ex in load_dataset(CANTO_XNLI_ID, revision=CANTO_XNLI_REVISION, split="test"):
    if ex["label"] == "neutral":
        continue
    rows.append({
        "id": ex["pairID"],
        "premise": ex["yue_premise"],
        "hypothesis": ex["yue_hypothesis"],
        "label": "entailment" if ex["label"] == "entailment" else "not_entailment",
    })

rng = random.Random(SEED)
sample = []
for label in ("entailment", "not_entailment"):
    sample += rng.sample([r for r in rows if r["label"] == label], N_PER_LABEL)
rng.shuffle(sample)

with open(HERE / "sample_nli.jsonl", "w", encoding="utf-8") as f:
    for r in sample:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(HERE / "nli_sample.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["id", "premise", "hypothesis", "label"])
    w.writeheader()
    w.writerows(sample)
print(f"Wrote {len(sample)} examples to sample_nli.jsonl and nli_sample.csv")
