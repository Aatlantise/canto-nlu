# Cantonese NLI evaluation: all six models

This bundle reproduces the 100-example Cantonese NLI evaluation for:

- Fine-tuned encoders: mBERT and XLM-R-base
- Zero-shot prompted models: Qwen2.5-7B-Instruct, Gemma-3-12B-it, Llama-3.1-8B-Instruct, and DeepSeek Flash

## Files

- `nli_encoder_finetune.py`: supervised full-model fine-tuning for mBERT and XLM-R-base.
- `nli_prompted_eval.py`: shared local evaluator for Qwen, Gemma, and Llama.
- `nli_deepseek_eval_stdlib.py`: official DeepSeek API evaluator with resumable JSONL output.
- `run_nli_*.sbatch`: Slurm launchers for the five GPU jobs.
- `summarize_nli_results.py`: combines completed summaries into CSV and Markdown tables.
- `sample_nli.jsonl`: balanced 100-example test sample.
- `requirements_nli.txt`: Python dependencies.
- `NLI_FINAL_RESULTS.md`: recorded results and reporting caveats.

## Environment

```bash
python -m pip install -r requirements_nli.txt
```

Gemma and Llama require accepted Hugging Face licenses and an authenticated account.

## GPU jobs

```bash
mkdir -p /home/USER/nli-eval/results/logs
sbatch run_nli_qwen.sbatch
sbatch run_nli_gemma.sbatch
sbatch run_nli_llama.sbatch
sbatch run_nli_mbert.sbatch
sbatch run_nli_xlmr.sbatch
```

The encoder evaluator uses per-output-directory Arrow caches and supports both Transformers 4.x (`warmup_ratio`) and Transformers 5.x (`warmup_steps`).

## DeepSeek API

Run each line separately. Paste only the API key at the hidden prompt.

```bash
read -rsp "DeepSeek API key: " DEEPSEEK_API_KEY
```

```bash
echo
export DEEPSEEK_API_KEY
python -u nli_deepseek_eval_stdlib.py --data sample_nli.jsonl --output results/deepseek-flash/predictions.jsonl --model deepseek-v4-flash --resume
unset DEEPSEEK_API_KEY
```

The legacy request name `deepseek-v4-flash` was served as `deepseek-flash` during the recorded run. Report it as DeepSeek-V4.1-Flash, or retain the requested name only with a routing footnote.

## Summarize

```bash
python summarize_nli_results.py --results-root results --csv nli_results.csv --markdown NLI_RESULTS.md
```

The 100-example scores are sample results, not full 6.6k test-split results. Fine-tuned and prompted models also use different evaluation settings and should remain visibly separated in reporting.
