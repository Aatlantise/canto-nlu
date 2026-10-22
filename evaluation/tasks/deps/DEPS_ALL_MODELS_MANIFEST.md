# Cantonese DEPS evaluation - all six models

This archive contains the complete current evaluation code for six models.

## Supervised encoder models

| Model | Training/evaluation protocol | Evaluator version |
|---|---|---|
| google-bert/bert-base-multilingual-cased | Fine-tune on train, select by validation LAS, evaluate held-out test once | mbert-deps-supervised-v2-cycle-safe |
| FacebookAI/xlm-roberta-base | Fine-tune on train, select by validation LAS, evaluate held-out test once | xlmr-deps-supervised-v2-cycle-safe |

The encoder parser uses deterministic greedy single-root decoding with cycle
repair. The full scripts run 20 epochs and select the best validation checkpoint.

## Zero-shot generative models

| Model | Backend | Evaluator version family |
|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | Local Hugging Face, bfloat16 | qwen2.5-7b-deps-stepwise-tsv-v2.1-recovery |
| google/gemma-3-12b-it | Local Hugging Face, NF4 | gemma3-12b-deps-stepwise-tsv-v2.1-recovery |
| meta-llama/Llama-3.1-8B-Instruct | Local Hugging Face, NF4, gated access | llama3.1-8b-deps-stepwise-tsv-v2.1-recovery |
| deepseek-v4-flash | DeepSeek official API | deepseek-v4-deps-stepwise-tsv-v2.1-recovery |

All four generative models use the same paper-inspired single-turn protocol:
UPOS, then HEAD, then DEPREL in minimal CoNLL-U-like TSV. Strict-format metrics
and tolerant ID+FORM row recovery are both reported. Primary UAS/LAS include
punctuation. The generative protocol must be reported separately from the
supervised encoder protocol.

## Required inputs

- `/home/USER/deps_train.jsonl`
- `/home/USER/deps_test.jsonl`

Each JSONL row requires `id`, `tokens`, `heads`, and `deprels`.

## Main source files

- `encoder_deps_finetune.py`: mBERT and XLM-R supervised parser
- `deps_eval_common.py`: shared UD labels and validation utilities
- `deps_stepwise_common.py`: three-step prompt, TSV recovery, tree checks, metrics
- `deps_stepwise_eval.py`: all four generative-model backends
- `deps_stepwise_rescore.py`: reparse saved raw answers without model calls
- `DEPS_STEPWISE_RUN.md`: detailed generative-model instructions

## Slurm scripts

There is one smoke and one full script for each model. Smoke scripts are for
runtime/format validation. Full scripts produce the reported held-out results.
Do not submit the same full encoder script twice because duplicate jobs write to
the same checkpoint directory.

## Exclusions

The archive contains no datasets, model weights, predictions, summaries, API
keys, or Hugging Face tokens. The original JSON v1 generative baseline is not
included; this package contains the current six-model evaluation methodology.
