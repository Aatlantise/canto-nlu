import pandas as pd

# Input and output file paths
input_jsonl = "laj_test.jsonl"  # Replace with your input filename
output_csv = "laj_sample.csv"

# 1. Read the JSONL file
df = pd.read_json(input_jsonl, lines=True)

# 2. (Optional) Reorder columns for a cleaner annotation layout
# You can also add blank columns for human annotators to fill out (e.g., 'human_label', 'notes')
df["human_label"] = ""  # Column for human annotator score
df["notes"] = ""  # Column for annotator notes

# Reorder columns logically
column_order = [
    "id",
    "source_id",
    "sentence",
    "label",
    "human_label",
    "notes",
]
df = df[[col for col in column_order if col in df.columns]]

# 3. Export to CSV using 'utf-8-sig' to ensure CJK characters display properly in Excel
df.to_csv(output_csv, index=False, encoding="utf-8-sig")

print(f"Successfully converted {len(df)} rows to {output_csv}")