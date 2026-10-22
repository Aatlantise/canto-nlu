# LAJ: mBERT, XLM-R, and Llama evaluation

## Fixed models

- `bert-base-multilingual-cased`
- `xlm-roberta-base`
- `meta-llama/Meta-Llama-3.1-8B-Instruct`

All jobs use the same `comparison_test.jsonl` pairs and `cantonese.jsonl`
filter. The filter removes source items whose MT sentence has no annotated
error span; with the supplied files this should leave 1,400 forced-choice
pairs.

## Scoring

- mBERT and XLM-R: masked pseudo-log-likelihood. The primary metric is
  accuracy from **mean token pseudo-surprisal**.
- Llama: autoregressive likelihood. The primary metric is accuracy from
  **mean token surprisal**.
- Lower score wins. Total-score accuracy is retained only as a diagnostic
  because sentence-length differences can bias it.

## Upload

Create `/home/USER/laj-models/` and upload:

- `eval_mlm_laj.py`
- `eval_causal_laj.py`
- `comparison_test.jsonl`
- `cantonese.jsonl`
- the three `run_laj_*.sbatch` files

## Dependencies

```bash
conda activate cantonlu-qwen
python -m pip install -U "torch>=2.3" "transformers>=4.45" accelerate sentencepiece
```

Llama is gated on Hugging Face. Accept its license on the model page, then run:

```bash
huggingface-cli login
```

## Smoke tests

Run these from a GPU node before submitting full jobs:

```bash
python eval_mlm_laj.py --pairs comparison_test.jsonl --source cantonese.jsonl --model bert-base-multilingual-cased --limit 10 --batch-size 32 --output-dir results/mbert-smoke
python eval_mlm_laj.py --pairs comparison_test.jsonl --source cantonese.jsonl --model xlm-roberta-base --limit 10 --batch-size 32 --output-dir results/xlmr-smoke
python eval_causal_laj.py --pairs comparison_test.jsonl --source cantonese.jsonl --model meta-llama/Meta-Llama-3.1-8B-Instruct --limit 10 --batch-size 4 --output-dir results/llama-smoke
```

## Full jobs

```bash
sbatch run_laj_mbert.sbatch
sbatch run_laj_xlmr.sbatch
sbatch run_laj_llama.sbatch
```

Each output directory contains `predictions.csv` and `summary.json`.
