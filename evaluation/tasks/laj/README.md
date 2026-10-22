# Linguistic Acceptability Judgment

This directory contains the Cantonese linguistic-acceptability-judgment
evaluation code for mBERT, XLM-R, Qwen, Gemma, DeepSeek, and Llama.

The main evaluation uses forced-choice sentence pairs. Each pair contains a
human reference sentence and a machine-translated sentence. A model is correct
when it assigns lower surprisal to the human reference.

## Dataset Source

The LAJ dataset is derived from the Cantonese subset of
[SiniticMTError](https://github.com/hannliu/SiniticMTError), a dataset of
machine-translation error annotations.

The original source file is downloaded from:

```text
https://raw.githubusercontent.com/hannliu/SiniticMTError/main/cantonese.jsonl
```

Prepare all LAJ datasets with:

```bash
python tasks/laj/prepare_data.py
```

The command downloads the source data and generates:

```text
tasks/laj/cantonese.jsonl
tasks/laj/laj_dataset.jsonl
tasks/laj/comparison_test.jsonl
```

The expected row counts are:

- `cantonese.jsonl`: 1,402 source records
- `laj_dataset.jsonl`: 2,804 single-sentence examples
- `comparison_test.jsonl`: 1,402 forced-choice pairs

The expected SHA-256 checksums are:

```text
cantonese.jsonl:
13eadab17632016223c806808d236657aae41fe0b395e4ac47a87e9d19550776

laj_dataset.jsonl:
6d5ebbce79496336a1edbb6a1cce7a9d421a19a57ddf8d1120ad3d753584bb28

comparison_test.jsonl:
c743cf796f4f261f049396d62fd8efbaead5a2531e01dbb288c8fcd3c9f68b74
```

On Windows PowerShell:

```powershell
Get-FileHash tasks\laj\cantonese.jsonl -Algorithm SHA256
Get-FileHash tasks\laj\laj_dataset.jsonl -Algorithm SHA256
Get-FileHash tasks\laj\comparison_test.jsonl -Algorithm SHA256
```

The downloaded and generated JSONL files are local artifacts and are not
committed to Git.

## Dataset Construction

`build_laj_dataset.py` converts each source record into two single-sentence
examples:

- The human reference is labeled acceptable.
- The machine translation is labeled according to its error annotations.

Punctuation and CJK spacing are normalized to prevent superficial formatting
differences from leaking the label.

`build_comparison_dataset.py` converts the single-sentence examples into
forced-choice pairs. The position of the reference sentence is randomized
with seed `42`.

Each comparison record has the following structure:

```json
{
  "id": 0,
  "source_id": 0,
  "sentence1": "FIRST_SENTENCE",
  "sentence2": "SECOND_SENTENCE",
  "label": 1
}
```

A label of `1` means that `sentence1` is the acceptable sentence. A label of
`2` means that `sentence2` is acceptable.

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
python -m pip install -r tasks/laj/requirements.txt
```

## Smoke Test

The following command scores one valid pair with a small model:

```bash
python tasks/laj/eval_qwen_laj.py \
  --pairs tasks/laj/comparison_test.jsonl \
  --source tasks/laj/cantonese.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --family qwen \
  --output-dir outputs/laj/qwen_smoke \
  --batch-size 1 \
  --limit 1
```

Windows PowerShell:

```powershell
python tasks\laj\eval_qwen_laj.py `
  --pairs tasks\laj\comparison_test.jsonl `
  --source tasks\laj\cantonese.jsonl `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --family qwen `
  --output-dir outputs\laj\qwen_smoke `
  --batch-size 1 `
  --limit 1
```

The 0.5B model is used only to validate the pipeline. Its result is not
comparable to the reported 7B-model evaluation.

## Full Qwen Evaluation

The full Qwen evaluation uses `Qwen/Qwen2.5-7B-Instruct`:

```bash
python tasks/laj/eval_qwen_laj.py \
  --pairs tasks/laj/comparison_test.jsonl \
  --source tasks/laj/cantonese.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --family qwen \
  --output-dir outputs/laj/qwen2.5-7b \
  --batch-size 8
```

A CUDA-capable GPU is strongly recommended. On a compatible Linux CUDA
environment, add `--load-in-4bit` to reduce GPU memory requirements.

When the original source file is provided through `--source`, pairs whose
machine translations have no annotated errors are excluded. The expected
evaluation contains 1,400 valid pairs and excludes 2 pairs.

## Evaluation Method

The evaluator calculates sentence surprisal directly from model token
probabilities. It does not use a chat template or an acceptability prompt.

The primary metric is:

```text
accuracy_total_surprisal
```

The evaluator also reports `accuracy_mean_token_surprisal` as a robustness
check because total surprisal is sensitive to sentence length.

## Outputs

Each evaluation creates:

- `predictions.csv`: sentence-level surprisal, token counts, predictions, and correctness
- `summary.json`: aggregate metrics and run configuration

## Other Models

The directory also contains:

- `eval_mlm_laj.py` for mBERT and XLM-R pseudo-surprisal
- `eval_qwen_laj.py` for Qwen and Gemma
- `eval_causal_laj.py` for Llama and other causal language models

See `LAJ_EVALUATION.md` and `LAJ_MBERT_XLMR_LLAMA_RUN.md` for the original
evaluation notes and model-specific commands.
