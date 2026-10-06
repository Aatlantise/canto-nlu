# Word-sense disambiguation (WSD)

`wsd_local_llm_eval.py` evaluates Qwen, Gemma, or Llama; `deepseek_v4_wsd_eval_stdlib.py` uses the DeepSeek API; and `mbert_wsd_mask_cosine.py` is a parameterized masked-encoder evaluator used for both mBERT and XLM-R-base. Dataset files are intentionally not included.

Smoke and full Slurm launchers preserve the original cluster configuration.
