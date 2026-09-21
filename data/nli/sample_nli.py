from datasets import load_dataset
import pandas as pd

# Load the test split directly into a Pandas DataFrame
df = load_dataset("hon9kon9ize/yue-all-nli", split="test").to_pandas()

# Print columns to verify data structure
print("Columns in dataset:", df.columns.tolist())

rows = []

# 2. Extract pairs based on column structure
# Case A: Triplet format (e.g., 'anchor'/'premise', 'positive', 'negative')
if "anchor" in df.columns or "premise" in df.columns:
    premise_col = "anchor" if "anchor" in df.columns else "premise"

    # Determine positive (entailment) and negative (non-entailment) column names
    pos_col = "positive" if "positive" in df.columns else "entailment"
    neg_col = (
        "negative"
        if "negative" in df.columns
        else ("contradiction" if "contradiction" in df.columns else "hypothesis")
    )

    for idx, row in df.iterrows():
        # Entailing pair
        rows.append(
            {
                "id": f"{idx}_ent",
                "premise": row[premise_col],
                "hypothesis": row[pos_col],
                "label": "entailment",
            }
        )
        # Non-entailing pair
        rows.append(
            {
                "id": f"{idx}_non_ent",
                "premise": row[premise_col],
                "hypothesis": row[neg_col],
                "label": "not_entailment",
            }
        )

# Case B: Standard Pair format (e.g., 'premise', 'hypothesis', 'label')
elif "label" in df.columns:
    for idx, row in df.iterrows():
        # Map label integer/string to binary classification
        is_entailment = row["label"] in [0, "entailment", "0"]
        rows.append(
            {
                "id": f"{idx}",
                "premise": row["premise"],
                "hypothesis": row["hypothesis"],
                "label": (
                    "entailment" if is_entailment else "not_entailment"
                ),
            }
        )

# 3. Create unrolled DataFrame
unrolled_df = pd.DataFrame(rows)

# 4. Extract a balanced subset of 100 items (50 entailment + 50 non-entailment)
entailment_samples = unrolled_df[unrolled_df["label"] == "entailment"].sample(
    n=50, random_state=42
)
non_entailment_samples = unrolled_df[
    unrolled_df["label"] == "not_entailment"
].sample(n=50, random_state=42)

# Combine and shuffle
balanced_subset = (
    pd.concat([entailment_samples, non_entailment_samples])
    .sample(frac=1, random_state=42)
    .reset_index(drop=True)
)

# 5. Export for human baseline testing
balanced_subset.to_csv("yue_nli_human_baseline_100.csv", index=True, encoding="utf-8-sig")
balanced_subset.to_json(
    "yue_nli_human_baseline_100.jsonl", orient="records", lines=True,
    force_ascii=False,
)

print(
    f"Successfully exported {len(balanced_subset)} balanced items for evaluation."
)
print(balanced_subset["label"].value_counts())