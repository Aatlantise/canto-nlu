# Part-of-Speech Tagging

This directory contains the Cantonese part-of-speech (POS) evaluation code for
Qwen, Gemma, Llama, DeepSeek, mBERT, and XLM-R.

The generative models are evaluated with zero-shot prompting. mBERT and XLM-R
are evaluated through supervised fine-tuning.

## Dataset

The POS dataset is derived from
[UD Cantonese-HK](https://github.com/UniversalDependencies/UD_Cantonese-HK).

The preparation script downloads the source CoNLL-U file and converts it into
the JSONL format expected by the evaluation scripts.

Prepare the dataset with:

```bash
python tasks/pos/prepare_data.py
```

The command generates:

```text
tasks/pos/pos_train.jsonl
tasks/pos/pos_test.jsonl
```

The split contains:

- 903 training sentences
- 101 test sentences
- 1,779 test tokens

The expected SHA-256 checksum of `pos_test.jsonl` is:

```text
e4a66cdd9ad4330e1afcc13a1b1ea07f6d9bc39ca960db3b68d0cb8364b7271b
```

On Windows PowerShell, verify it with:

```powershell
Get-FileHash tasks\pos\pos_test.jsonl -Algorithm SHA256
```

Each JSONL record has the following structure:

```json
{
  "id": 0,
  "tokens": ["你", "喺度", "搵", "乜嘢", "呀", "？"],
  "upos": ["PRON", "ADV", "VERB", "PRON", "PART", "PUNCT"]
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

Install the POS dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r tasks/pos/requirements.txt
```

## Smoke Test

The following command uses a smaller model and one example to verify that the
complete pipeline can load the dataset, run inference, calculate metrics, and
write the output files:

```bash
python tasks/pos/qwen_pos_eval.py \
  --data tasks/pos/pos_test.jsonl \
  --output outputs/pos/qwen_smoke.jsonl \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --max-samples 1 \
  --max-new-tokens 32 \
  --seed 42
```

Windows PowerShell uses the backtick as the line-continuation character:

```powershell
python tasks\pos\qwen_pos_eval.py `
  --data tasks\pos\pos_test.jsonl `
  --output outputs\pos\qwen_smoke.jsonl `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --max-samples 1 `
  --max-new-tokens 32 `
  --seed 42
```

The 0.5B model is provided only as a lightweight pipeline test. Its evaluation
metrics are not directly comparable to the reported 7B-model results.

## Full Qwen Evaluation

The original Qwen evaluation uses `Qwen/Qwen2.5-7B-Instruct`:

```bash
python tasks/pos/qwen_pos_eval.py \
  --data tasks/pos/pos_test.jsonl \
  --output outputs/pos/qwen2.5-7b.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --seed 42
```

A CUDA-capable GPU is strongly recommended for the full evaluation. Running
the 7B model with CPU or disk offloading can be very slow.

## Outputs

Each evaluation produces two files:

- `<name>.jsonl`: per-example predictions and parsing information
- `<name>.summary.json`: aggregate metrics and runtime information

For example:

```text
outputs/pos/qwen2.5-7b.jsonl
outputs/pos/qwen2.5-7b.summary.json
```

The summary includes token accuracy, non-punctuation accuracy, macro-F1,
sentence exact match, invalid-output rate, runtime, model identifier, and seed.

## Other Models

The directory also contains evaluators and launch scripts for:

- Gemma 3 12B
- Llama 3.1 8B
- DeepSeek
- mBERT
- XLM-R

Run any Python entry point with `--help` to inspect its available arguments:

```bash
python tasks/pos/qwen_pos_eval.py --help
```

The historical Slurm launch scripts preserve the original experiment
configuration. Update their cluster paths, environment names, partitions, and
output directories before using them on another computing cluster.
