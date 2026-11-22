# Reproducing the open-weight full-test evaluation

This is the minimal reproduction path for the completed audit:
**3 open-weight models x 7 complete test sets = 21 runs**.

## Fixed scope

Base repository revision: `Aatlantise/canto-nlu@5c42cb1`.

Models:

- `Qwen/Qwen2.5-7B-Instruct`
- `google/gemma-3-12b-it`
- `meta-llama/Llama-3.1-8B-Instruct`

All runs use local Hugging Face inference, greedy decoding, and 4-bit NF4
quantization. Invalid structured outputs are retained and counted as incorrect.

| Task | Input | Expected rows | Output stem |
|---|---|---:|---|
| NLI | `data/nli/full_test.jsonl` | 13,202 | `nli` |
| Sentiment | `data/sentiment/test.jsonl` | 999 | `sentiment` |
| LID | `data/ld/lid_test.jsonl` | 6,000 | `ld` |
| POS | `data/pos/pos_test.jsonl` | 101 | `pos` |
| DEPS | `data/deps/deps_test.jsonl` | 101 | `deps` |
| WSD | `data/wsd/test.jsonl` | 1,646 | `wsd` |
| LAJ | `data/laj/laj_dataset.jsonl` | 2,804 | `laj_prompted_full` |

The LAJ run here is prompted classification over all 2,804 single-sentence
examples. It is intentionally separate from the pairwise surprisal formulation.

## Prompt design

The prompts were derived from each benchmark task's label definitions and output
schema, not tuned against model predictions. One fixed prompt per task was used
unchanged for Qwen, Gemma, and Llama. The evaluation is zero-shot: no test example
or model-specific demonstration appears in a prompt. Greedy decoding removes
sampling variation, and no model-specific post-processing correction is applied.

| Task | Prompt decision |
|---|---|
| NLI | Defines binary `entailment` and `not_entailment`; contradiction, irrelevance, and insufficient information map to the benchmark's single non-entailment class. |
| Sentiment | Defines the dataset labels `smile`, `ok`, and `cry` as positive, neutral/mixed, and negative. |
| LID | Uses `cantonese`, `mandarin`, and `other`; it explicitly says that simplified/traditional characters are not the criterion, following the grammar-derived dataset definition. |
| LAJ | Asks whether one Hong Kong Cantonese sentence is natural and acceptable, returning `acceptable` or `unacceptable`. |
| WSD | Defines `similar` as the same target-word sense across two sentences and `not_similar` as different senses. |
| POS | Supplies pre-segmented tokens and the closed Universal Dependencies UPOS inventory; output must be a same-length JSON array. |
| DEPS | Supplies pre-segmented tokens, defines 1-based heads and head `0` as root, and lists the dependency-relation inventory present in the dataset; output must match the documented JSON schema. |

Classification instructions are written in Cantonese to match the benchmark
inputs. POS and DEPS use English because UPOS and Universal Dependencies relation
names are standardized English identifiers. Classification parsing accepts an
exact allowed label or a response containing one unambiguous allowed label.
POS/DEPS parsing is intentionally strict: malformed JSON, unknown labels, and
sequence-length mismatches are recorded as invalid and counted as incorrect.

## 1. Create the environment

The dependency file records the package versions used for the completed run.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-open-weight.txt
hf auth login
```

Gemma and Llama require accepted Hugging Face licenses. Confirm that the account
can download both gated checkpoints before submitting jobs.

## 2. Export the full NLI test

```bash
python data/nli/export_full_test.py
```

This creates ignored local files `data/nli/full_test.jsonl` and
`data/nli/full_test.metadata.json`. The metadata records the source fingerprint,
row counts, label counts, and SHA-256. Do not substitute the 100-row
`sample_nli.jsonl` file.

## 3. Run the evaluations

`evaluate_llm.py` is the only evaluation entry point. Run it once for each model
and task combination in the scope tables above. For example:

```bash
# Qwen NLI
python evaluate_llm.py \
  --task nli \
  --model Qwen/Qwen2.5-7B-Instruct \
  --data data/nli/full_test.jsonl \
  --expected-samples 13202 \
  --output results/open_weight_full/qwen2.5-7b/nli.jsonl \
  --load-in-4bit \
  --resume

# Gemma full prompted LAJ
python evaluate_llm.py \
  --task laj \
  --model google/gemma-3-12b-it \
  --data data/laj/laj_dataset.jsonl \
  --expected-samples 2804 \
  --output results/open_weight_full/gemma-3-12b/laj_prompted_full.jsonl \
  --load-in-4bit \
  --resume
```

Use the same command template for the other model IDs, task inputs, expected row
counts, and output stems listed above. The commands are scheduler-independent;
users may submit them through their own cluster or process manager.

The generated artifacts are:

```text
results/open_weight_full/<model-key>/<output-stem>.jsonl
results/open_weight_full/<model-key>/<output-stem>.run.json
results/open_weight_full/<model-key>/<output-stem>.summary.json
```

## 4. Resume interrupted runs

Rerun the same command with `--resume`. The evaluator validates the saved model,
task, input hash, expected sample count, and inference settings before skipping
completed IDs.

Do not reuse an output file after changing its model, dataset, prompt, or
quantization settings.

## 5. Check completion

Completeness checks are built into `evaluate_llm.py`. `--expected-samples` rejects
a partial input before model loading. Prediction IDs must be unique, resume data
must match the current gold labels, and the completed summary records `samples`,
`expected_samples`, `data_sha256`, parsing status, metrics, and run configuration.
For a successful full run, `samples` equals the expected row count in the table.

Predictions, summaries, run metadata, logs, caches, model weights, and credentials
are generated artifacts and are not committed to Git.
