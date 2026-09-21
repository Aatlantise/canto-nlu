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

## Artifacts
`sample_nli.jsonl` and `nli_sample.csv`: the same 100 balanced examples, one
per format.
