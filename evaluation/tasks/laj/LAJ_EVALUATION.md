# Qwen LAJ evaluation

This evaluation follows CantoNLU's forced-choice setup: the human reference is
acceptable, while an MT sentence is included only when it has at least one
annotated error. The model is correct when the acceptable sentence has lower
surprisal.

## Install

```bash
pip install "torch>=2.1" "transformers>=4.37" accelerate
```

## Run

```bash
python eval_qwen_laj.py \
  --pairs comparison_test.jsonl \
  --source cantonese.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --batch-size 8 \
  --output-dir laj_qwen_results
```

On Windows PowerShell, use one line or replace each `\\` with a backtick.

Outputs:

- `summary.json`: pair count, exclusions, ties, and both accuracy metrics.
- `predictions.csv`: sentence-level surprisal, token counts, predictions, and
  correctness for every pair.

`accuracy_total_surprisal` is the requested primary result. Because total
surprisal is length-sensitive and LAJ errors include omissions, the script also
reports `accuracy_mean_token_surprisal` as a robustness check. Sentences are
tokenized directly, without a chat template or grammaticality prompt.

For the supplied files, 1,400 of 1,402 pairs are valid. Two source items have
no MT error spans and are excluded when `--source cantonese.jsonl` is given.
