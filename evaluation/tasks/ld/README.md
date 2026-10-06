# Language detection (LD)

This task classifies text as Cantonese, Mandarin, or mixed. `ld_metrics.py` contains shared loading and metrics; `mbert_ld_finetune.py` is a parameterized encoder trainer used for both mBERT and XLM-R-base; `ld_prompted_eval.py` evaluates Qwen, Gemma, and Llama; and `deepseek_v4_ld_eval_stdlib.py` evaluates DeepSeek.

`LD_MANIFEST.json` records split counts and checksums without redistributing the split files.
