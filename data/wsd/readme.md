# Word sense disambiguation

This directory hosts the word sense disambiguation (WSD) subset of CantoNLU:
a test set of Cantonese homonym sentence pairs, built from a hand-curated
sense list.

This readme is substantially Claude-generated.

## Pipeline

```
output/senses.csv ── make_wsd_test_jsonl.py ──> test.jsonl
test.jsonl ── evaluate_wsd.py ──> WSD accuracy for a given model
```

`output/senses.csv` is the starting point, checked into the repo. Each row
is one dictionary sense of a Cantonese word, with two example sentences
per sense (`sentence1`, `sentence2`).

`test.jsonl` is generated from `make_wsd_test_jsonl.py`.

## Files

`make_wsd_test_jsonl.py` builds `test.jsonl` from `senses.csv`:
- each sense's own sentence pair is a `similar` example
- every pair of *different* senses of the same word produces `not_similar`
  examples (all 4 cross-sentence combinations)

`evaluate_wsd.py` evaluates a model against the dataset: normalizes
`senses.csv` to traditional characters (`output/senses-traditional.csv`),
masks the target word in each sentence pair, embeds with an encoder model,
and checks whether same-sense pairs are more similar than cross-sense
pairs.

## Artifacts (`output/`)

| File | Produced by | Description                                                              |
|---|---|--------------------------------------------------------------------------|
| `senses.csv` | checked into repo (manual curation) | Sense list with hand-sourced example sentences; pipeline starting point  |
| `senses-traditional.csv` | `evaluate_wsd.py` | `senses.csv` normalized to traditional characters                        |
| `test.jsonl` (top-level) | `make_wsd_test_jsonl.py` | Final WSD evaluation set: `similar`/`not_similar` sentence-pair examples |
