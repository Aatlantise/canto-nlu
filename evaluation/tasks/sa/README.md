# Sentiment Analysis

This directory contains the Cantonese sentiment-analysis evaluation code for
mBERT, XLM-R, Qwen, Gemma, DeepSeek, and Llama.

The dataset contains Cantonese restaurant reviews with three sentiment labels:

- `smile`
- `ok`
- `cry`

## Dataset

The dataset is adapted from the Cantonese
[OpenRice sentiment dataset](https://github.com/toastynews/openrice-senti),
which is distributed under the CC BY 4.0 license.

Download and prepare the dataset with:

```bash
python tasks/sa/prepare_data.py
```

The script downloads the original TSV files and produces JSONL versions:

```text
tasks/sa/train.tsv
tasks/sa/valid.tsv
tasks/sa/test.tsv
tasks/sa/train.jsonl
tasks/sa/valid.jsonl
tasks/sa/test.jsonl
```

The dataset contains:

- 9,999 training examples
- 999 validation examples
- 999 test examples

The expected SHA-256 checksum of the original `test.tsv` is:

```text
0f6fa0b8ee7ba54bc976e6fa0db265cf4fb0368a22d5bdc9e88c95bdebcfcb5a
```

On Windows PowerShell, verify it with:

```powershell
Get-FileHash tasks\sa\test.tsv -Algorithm SHA256
```

Each TSV file contains two columns:

```text
label    text_a
```

The generated JSONL records have the following structure:

```json
{
  "id": 0,
  "sentence": "Example Cantonese restaurant review",
  "label": "smile",
  "source": "openrice-senti"
}
```

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
python -m pip install -r tasks/sa/requirements.txt
```

## Smoke Test

The following command uses one training example and a small model to verify
data loading, inference, metric calculation, and output generation:

```bash
python tasks/sa/qwen_sa_eval.py \
  --data tasks/sa/train.tsv \
  --output outputs/sa/qwen_smoke.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --max-samples 1 \
  --seed 42
```

Windows PowerShell:

```powershell
python tasks\sa\qwen_sa_eval.py `
  --data tasks\sa\train.tsv `
  --output outputs\sa\qwen_smoke.jsonl `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --max-samples 1 `
  --seed 42
```

The 0.5B model is used only to validate the pipeline. Its metrics are not
comparable to the reported 7B-model results.

## Full Qwen Evaluation

The original Qwen evaluation uses `Qwen/Qwen2.5-7B-Instruct` on the held-out
test set:

```bash
python tasks/sa/qwen_sa_eval.py \
  --data tasks/sa/test.tsv \
  --output outputs/sa/qwen2.5-7b.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --seed 42
```

A CUDA-capable GPU is strongly recommended for the full evaluation.

On a compatible Linux CUDA environment, `--load-in-4bit` can be used to reduce
GPU memory requirements.

## Outputs

Each evaluation produces:

- `<name>.jsonl`: per-example predictions
- `<name>.summary.json`: aggregate metrics

The summary includes:

- Accuracy
- Macro-F1
- Invalid-output rate
- Accuracy and macro-F1 on valid predictions

## Other Models

The directory also contains evaluation and fine-tuning scripts for:

- mBERT
- XLM-R
- Gemma 3 12B
- DeepSeek
- Llama 3.1 8B

Run an evaluator with `--help` to inspect its command-line options:

```bash
python tasks/sa/qwen_sa_eval.py --help
```

The Slurm launchers preserve the original cluster configuration. Update their
paths, environment names, partitions, and output directories before using them
on another cluster.
