# Natural Language Inference

This directory contains the Cantonese natural-language-inference evaluation
code for mBERT, XLM-R, Qwen, Gemma, DeepSeek, and Llama.

The task is binary classification with two labels:

- `entailment`
- `not_entailment`

## Dataset

The evaluation data is derived from the
[`hon9kon9ize/yue-all-nli`](https://huggingface.co/datasets/hon9kon9ize/yue-all-nli)
dataset on Hugging Face.

Each source triplet is converted into two premise-hypothesis pairs:

1. A positive pair labeled `entailment`
2. A negative pair labeled `not_entailment`

Prepare the complete test set with:

```bash
python tasks/nli/prepare_data.py
```

This generates:

```text
tasks/nli/full_test.jsonl
tasks/nli/full_test.metadata.json
```

The metadata file records:

- Hugging Face dataset identifier
- Requested dataset revision
- Dataset fingerprint
- Source and output row counts
- Label counts
- SHA-256 checksum of the generated JSONL file

To use a specific immutable Hugging Face dataset revision:

```bash
python tasks/nli/prepare_data.py \
  --revision DATASET_COMMIT_HASH
```

Each generated JSONL record has the following structure:

```json
{
  "id": "743_non_ent",
  "premise": "一個電單車司機過緊個坑。",
  "hypothesis": "駱駝騎手騎過個埞。",
  "label": "not_entailment"
}
```

The generated dataset and metadata are local artifacts and are not committed
to Git.

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
python -m pip install -r tasks/nli/requirements.txt
```

## Smoke Test

The following command evaluates one example with a small model to verify data
loading, inference, output parsing, metric calculation, and output writing:

```bash
python tasks/nli/nli_prompted_eval.py \
  --data tasks/nli/full_test.jsonl \
  --output-dir outputs/nli/qwen_smoke \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --batch-size 1 \
  --max-samples 1
```

Windows PowerShell:

```powershell
python tasks\nli\nli_prompted_eval.py `
  --data tasks\nli\full_test.jsonl `
  --output-dir outputs\nli\qwen_smoke `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --batch-size 1 `
  --max-samples 1
```

The 0.5B model is used only to validate the pipeline. Its result is not
comparable to the reported 7B-model evaluation.

## Full Qwen Evaluation

The full Qwen evaluation uses `Qwen/Qwen2.5-7B-Instruct`:

```bash
python tasks/nli/nli_prompted_eval.py \
  --data tasks/nli/full_test.jsonl \
  --output-dir outputs/nli/qwen2.5-7b \
  --model Qwen/Qwen2.5-7B-Instruct \
  --batch-size 4
```

A CUDA-capable GPU is strongly recommended. On a compatible Linux CUDA
environment, add `--load-in-4bit` to reduce GPU memory requirements.

## Outputs

Each prompted-model evaluation creates:

```text
predictions.jsonl
predictions.csv
summary.json
```

The summary contains:

- Accuracy
- Macro-F1
- Invalid-output rate
- Model and evaluator identifiers
- Quantization and compute dtype
- Runtime and software versions

## Evaluation Protocol

The prompted models use zero-shot deterministic greedy decoding. Only the
premise and hypothesis are included in the prompt. The gold label and record
identifier are not exposed to the model.

`nli_encoder_finetune.py` provides supervised fine-tuning for mBERT and XLM-R.
`nli_deepseek_eval_stdlib.py` provides the DeepSeek API evaluation.

See `NLI_ALL_MODELS_README.md` for the complete model matrix and
`NLI_RUN.md` for the original experiment notes.

The historical 100-example balanced sample must not be reported as the full
test-set result.
