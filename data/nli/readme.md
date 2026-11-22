# Natural language inference

This is the natural language inference (NLI) subset of CantoNLU,
built from [`hon9kon9ize/yue-all-nli`](https://huggingface.co/datasets/hon9kon9ize/yue-all-nli).
Due to the large size of the dataset, we only provide a subsample for human annotation.
The full dataset can be directly taken from the Hugging Face repository for 
language model fine-tuning and evaluation.

## Dataset info
Each source row (premise plus a positive/entailing and a negative/non-entailing
hypothesis) is unrolled into two binary-labeled premise-hypothesis pairs:
* `entailment`
* `not_entailment`

```json
{"id":"743_non_ent","premise":"一個電單車司機過緊個坑。","hypothesis":"駱駝騎手騎過個埞。","label":"not_entailment"}
```

## Requirements
Run `python3 sample_nli.py`. The script downloads the `test` split of
`hon9kon9ize/yue-all-nli` via `datasets.load_dataset`, unrolls it into
premise-hypothesis pairs, and samples 50 `entailment` + 50 `not_entailment`
examples (fixed `random_state=42`).

For a complete model evaluation, export every row from the upstream test split:

```bash
python data/nli/export_full_test.py
```

This creates ignored local files `full_test.jsonl` and
`full_test.metadata.json`. The metadata records the Hugging Face dataset
fingerprint, source and output row counts, label counts, and output SHA-256.
The full export, rather than `sample_nli.jsonl`, must be passed to model
evaluation jobs.

## Artifacts
* `sample_nli.jsonl` and `nli_sample.csv`: the same 100 balanced examples, one
  per format.
* `full_test.jsonl` and `full_test.metadata.json`: generated, ignored artifacts
  for complete model evaluation.
