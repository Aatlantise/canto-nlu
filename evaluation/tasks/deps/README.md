# Dependency Parsing

This directory contains the Cantonese dependency-parsing evaluation code for
mBERT, XLM-R, Qwen, Gemma, DeepSeek, and Llama.

`encoder_deps_finetune.py` implements supervised encoder parsing.
`deps_stepwise_eval.py` implements the current three-step prompting protocol
for generative models.

## Dataset

The dataset is derived from
[UD Cantonese-HK](https://github.com/UniversalDependencies/UD_Cantonese-HK).

Prepare the dataset with:

```bash
python tasks/deps/prepare_data.py
```

The script downloads the source CoNLL-U file and generates:

```text
tasks/deps/deps_train.jsonl
tasks/deps/deps_test.jsonl
```

The split contains:

- 903 training sentences
- 101 test sentences

The expected SHA-256 checksum of `deps_test.jsonl` is:

```text
6b5b75bc9c1c97647f67b641e7d897edb119ef3f14a7a5723e987813a7a1a968
```

On Windows PowerShell, verify it with:

```powershell
Get-FileHash tasks\deps\deps_test.jsonl -Algorithm SHA256
```

Each JSONL record has the following structure:

```json
{
  "id": 0,
  "tokens": ["你", "喺度", "搵", "乜嘢", "呀", "？"],
  "deprels": ["nsubj", "advmod", "root", "obj", "discourse:sp", "punct"],
  "heads": [3, 3, 0, 3, 3, 3]
}
```

Head indices are one-based. A head index of `0` identifies the root.

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
python -m pip install -r tasks/deps/requirements.txt
```

## Smoke Test

The smoke test uses a small model and one training example. It verifies data
loading, model inference, response parsing, metric calculation, and output
writing without evaluating the held-out test set.

```bash
python tasks/deps/deps_stepwise_eval.py \
  --data tasks/deps/deps_train.jsonl \
  --output outputs/deps/qwen_smoke.jsonl \
  --backend local \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --evaluator-version deps-stepwise-smoke-v1 \
  --max-samples 1 \
  --max-new-tokens 256 \
  --seed 42
```

Windows PowerShell:

```powershell
python tasks\deps\deps_stepwise_eval.py `
  --data tasks\deps\deps_train.jsonl `
  --output outputs\deps\qwen_smoke.jsonl `
  --backend local `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --evaluator-version deps-stepwise-smoke-v1 `
  --max-samples 1 `
  --max-new-tokens 256 `
  --seed 42
```

The 0.5B model is only a pipeline test. Its metrics are not comparable to the
reported 7B-model results.

## Full Qwen Evaluation

The original Qwen evaluation uses `Qwen/Qwen2.5-7B-Instruct` on the held-out
test set:

```bash
python tasks/deps/deps_stepwise_eval.py \
  --data tasks/deps/deps_test.jsonl \
  --output outputs/deps/qwen2.5-7b-stepwise.jsonl \
  --backend local \
  --model Qwen/Qwen2.5-7B-Instruct \
  --evaluator-version qwen2.5-7b-deps-stepwise-tsv-v2.1-recovery \
  --seed 42
```

A CUDA-capable GPU is strongly recommended for the full evaluation. Running
the 7B model with CPU or disk offloading can be very slow.

## Outputs

Each run produces:

- `<name>.jsonl`: per-example predictions and parsed dependency structures
- `<name>.summary.json`: aggregate metrics and runtime information

The primary metrics are:

- Unlabeled Attachment Score (`uas`)
- Labeled Attachment Score (`las`)
- Non-punctuation UAS and LAS
- Invalid-output rate
- Three-step completion and token-recall metrics

## Protocol Notes

The current generative evaluation uses a three-step TSV protocol:

1. Predict `ID FORM UPOS`
2. Predict `ID FORM UPOS HEAD`
3. Predict `ID FORM UPOS HEAD DEPREL`

This is a zero-shot adaptation of a paper-inspired protocol. It does not
reproduce a supervised LoRA fine-tuning setting.

Historical single-turn evaluators are retained under `legacy/` and should not
be mixed with the current stepwise results.

See `DEPS_STEPWISE_RUN.md` and `DEPS_ALL_MODELS_MANIFEST.md` for the complete
experiment matrix and Slurm launchers.
