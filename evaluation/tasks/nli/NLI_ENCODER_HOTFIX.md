# NLI encoder hotfix v2

This update fixes two failures observed on the cluster:

1. Transformers 5.x removed `warmup_ratio` from `TrainingArguments`; the evaluator now converts the requested ratio to an equivalent `warmup_steps` value when necessary.
2. mBERT and XLM-R previously mapped the million-example dataset into the same shared Hugging Face Arrow cache. Each run now writes expanded and tokenized datasets under its own output directory.

Upload `nli_encoder_hotfix_v2.zip`, then run:

```bash
cd /home/USER/nli-eval
unzip -o ~/nli_encoder_hotfix_v2.zip -d /home/USER/nli-eval
python -m py_compile nli_encoder_finetune.py
sbatch run_nli_mbert.sbatch
```

Wait until mBERT has entered training or completed before submitting XLM-R:

```bash
sbatch run_nli_xlmr.sbatch
```

The previous failed output directories may remain. The new per-model cache filenames are safe to reuse; incomplete temporary Arrow files are ignored.
