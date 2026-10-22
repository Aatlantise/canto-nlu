# Word-Sense Disambiguation

This directory contains the Cantonese word-sense-disambiguation evaluation
code for mBERT, XLM-R, Qwen, Gemma, DeepSeek, and Llama.

The task determines whether two Cantonese sentences use the same target word
with the same meaning.

The labels are:

- `similar`: the target word has the same sense in both sentences
- `not_similar`: the target word has different senses

## Dataset

The test set is generated from the included hand-curated `senses.csv` file.

Each row of `senses.csv` represents one sense of a Cantonese target word and
contains two example sentences for that sense.

Generate the evaluation set with:

```bash
python tasks/wsd/prepare_data.py \
  --input tasks/wsd/senses.csv \
  --output tasks/wsd/test.jsonl
```

Windows PowerShell:

```powershell
python tasks\wsd\prepare_data.py `
  --input tasks\wsd\senses.csv `
  --output tasks\wsd\test.jsonl
```

The generated test set contains:

- 302 `similar` examples
- 1,344 `not_similar` examples
- 1,646 examples in total

The expected SHA-256 checksums are:

```text
senses.csv:
8deb5cee88667dea2c0ee9ef076a9765272dca2cc27209fec55306bb28423a97

test.jsonl:
cf0c074466b20092a2e750adc027cee5096b3d120d8436fd071e11899957e155
```

On Windows PowerShell, verify them with:

```powershell
Get-FileHash tasks\wsd\senses.csv -Algorithm SHA256
Get-FileHash tasks\wsd\test.jsonl -Algorithm SHA256
```

Each generated JSONL record has the following structure:

```json
{
  "id": 0,
  "target": "TARGET_WORD",
  "sentence1": "FIRST_SENTENCE",
  "sentence2": "SECOND_SENTENCE",
  "label": "similar"
}
```

## Dataset Construction

For every sense in `senses.csv`, the preparation script creates one positive
pair using the two sentences associated with that sense.

For every pair of different senses belonging to the same target word, the
script creates all four cross-sense sentence combinations and labels them
`not_similar`.

Before generating the test set, the script verifies that the target word
appears in both example sentences.

## Environment Setup

Create and activate a virtual environment.

On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r tasks/wsd/requirements.txt
```

## Balanced Smoke Test

The following command selects one example from each label and uses a small
model to verify the complete evaluation pipeline:

```bash
python tasks/wsd/wsd_local_llm_eval.py \
  --data tasks/wsd/test.jsonl \
  --output outputs/wsd/qwen_smoke.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --family qwen \
  --samples-per-label 1 \
  --seed 42
```

Windows PowerShell:

```powershell
python tasks\wsd\wsd_local_llm_eval.py `
  --data tasks\wsd\test.jsonl `
  --output outputs\wsd\qwen_smoke.jsonl `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --family qwen `
  --samples-per-label 1 `
  --seed 42
```

The 0.5B model is used only to validate the pipeline. Its metrics are not
comparable to the reported 7B-model evaluation.

## Full Qwen Evaluation

The full Qwen evaluation uses `Qwen/Qwen2.5-7B-Instruct`:

```bash
python tasks/wsd/wsd_local_llm_eval.py \
  --data tasks/wsd/test.jsonl \
  --output outputs/wsd/qwen2.5-7b.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --family qwen \
  --seed 42
```

A CUDA-capable GPU is strongly recommended for the full evaluation.

Do not use `--samples-per-label` for the full evaluation because it changes
the original class distribution.

## Outputs

Each evaluation creates:

- `<name>.jsonl`: per-example predictions
- `<name>.summary.json`: aggregate metrics and runtime information

The summary contains:

- Accuracy
- Macro-F1
- Per-class precision, recall, and F1
- Invalid-output rate
- Class distribution
- Runtime and hardware information

## Other Models

The directory also contains:

- `mbert_wsd_mask_cosine.py` for mBERT and XLM-R
- `wsd_local_llm_eval.py` for Qwen, Gemma, and Llama
- `deepseek_v4_wsd_eval_stdlib.py` for the DeepSeek API

Run an evaluator with `--help` to inspect its available arguments:

```bash
python tasks/wsd/wsd_local_llm_eval.py --help
```

The Slurm launchers preserve the original cluster configuration. Update their
paths, environment names, partitions, and output directories before using them
on another cluster.
