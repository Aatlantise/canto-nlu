# Sentiment analysis (SA)

This folder combines the OpenRice/CantoNLU sentiment evaluators used across supervised encoders, local instruction models, and API models.

Primary entry points include `mbert_sa_finetune.py`, `sa_encoder_finetune.py`, `sa_llama_eval.py`, `qwen_sa_eval.py`, `gemma3_sa_eval.py`, and `deepseek_v4_sa_eval_stdlib.py`. The mBERT search/final summarizers preserve the earlier encoder experiment series.

Use `SA_XLMR_LLAMA_RUN.md` for the later leakage-safe comparison protocol and `README_GPU_SETUP.md` for the initial Qwen GPU workflow. The Slurm scripts retain original cluster paths.

`requirements.txt` covers the combined SA folder. `requirements-original.txt` preserves the narrower dependency set used by the later XLM-R/Llama bundle.

`legacy/deepseek_v3_sa_eval.py` preserves the older DeepSeek-V3/OpenRouter implementation. The canonical DeepSeek entry point for the six-model matrix is `deepseek_v4_sa_eval_stdlib.py`.
