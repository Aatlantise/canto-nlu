# Four-model Cantonese DEPS evaluation code

Protocol: `zero-shot-greedy-three-step-tsv`

Evaluator family: `deps-stepwise-tsv-v2.1-recovery`

## Models

| Model | Backend | Precision/access |
|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | Local Hugging Face | bfloat16 |
| google/gemma-3-12b-it | Local Hugging Face | bitsandbytes NF4, bfloat16 compute |
| meta-llama/Llama-3.1-8B-Instruct | Local Hugging Face | bitsandbytes NF4; gated HF access |
| deepseek-v4-flash | DeepSeek official API | Provider API; served model recorded in results |

## Required input files

- `/home/USER/deps_train.jsonl` for smoke/development checks
- `/home/USER/deps_test.jsonl` for the held-out full evaluation

Each JSONL record must contain `id`, `tokens`, `heads`, and `deprels`.

## Core code

- `deps_eval_common.py`: shared UD labels, data validation, and v1-compatible utilities
- `deps_stepwise_common.py`: three-step prompt, TSV recovery, tree checks, and metrics
- `deps_stepwise_eval.py`: Qwen, Gemma, Llama, and DeepSeek inference backends
- `deps_stepwise_rescore.py`: parser-only re-scoring without additional model calls
- `DEPS_STEPWISE_RUN.md`: complete upload, submission, monitoring, and result instructions

## Reproducibility and reporting

The v2.1 protocol predicts UPOS, then HEAD, then DEPREL in one response. Strict
format metrics and tolerant ID+FORM recovery metrics are both retained. Primary
UAS/LAS include punctuation. Report this protocol separately from the original
JSON v1 experiment because the prompting and recovery policies differ.

No API keys, Hugging Face tokens, datasets, model weights, predictions, or result
summaries are included in this archive.
