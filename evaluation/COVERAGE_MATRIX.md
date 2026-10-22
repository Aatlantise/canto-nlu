# Evaluation coverage: 7 tasks × 6 models

This file defines the canonical scope of the repository. A check mark means that a concrete evaluator can run the task/model combination; several model families intentionally share one parameterized evaluator.

| Task | mBERT | XLM-R-base | Qwen2.5-7B-Instruct | Gemma-3-12B-it | DeepSeek | Llama-3.1-8B-Instruct |
|---|---|---|---|---|---|---|
| WSD | `mbert_wsd_mask_cosine.py` | `mbert_wsd_mask_cosine.py` | `wsd_local_llm_eval.py` | `wsd_local_llm_eval.py` | `deepseek_v4_wsd_eval_stdlib.py` | `wsd_local_llm_eval.py` |
| LAJ | `eval_mlm_laj.py` | `eval_mlm_laj.py` | `eval_qwen_laj.py` | `eval_qwen_laj.py` | `eval_qwen_laj.py`¹ | `eval_causal_laj.py` |
| LD | `mbert_ld_finetune.py` | `mbert_ld_finetune.py` | `ld_prompted_eval.py` | `ld_prompted_eval.py` | `deepseek_v4_ld_eval_stdlib.py` | `ld_prompted_eval.py` |
| NLI | `nli_encoder_finetune.py` | `nli_encoder_finetune.py` | `nli_prompted_eval.py` | `nli_prompted_eval.py` | `nli_deepseek_eval_stdlib.py` | `nli_prompted_eval.py` |
| SA | `mbert_sa_finetune.py` | `sa_encoder_finetune.py` | `qwen_sa_eval.py` | `gemma3_sa_eval.py` | `deepseek_v4_sa_eval_stdlib.py` | `sa_llama_eval.py` |
| POS | `mbert_pos_finetune.py` | `xlmr_pos_finetune.py` | `qwen_pos_eval.py` | `gemma3_pos_eval.py` | `deepseek_v4_pos_eval.py` | `llama_pos_eval.py` |
| DEPS | `encoder_deps_finetune.py` | `encoder_deps_finetune.py` | `deps_stepwise_eval.py` | `deps_stepwise_eval.py` | `deps_stepwise_eval.py` | `deps_stepwise_eval.py` |

¹ LAJ requires sentence-level token likelihoods rather than a generated class label. The existing DeepSeek LAJ run therefore uses the locally hosted `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` causal model. The other DeepSeek task implementations use the official API model name `deepseek-v4-flash`. These must be reported as different model variants.

## Canonical model identifiers

| Display name | Hugging Face/API identifier |
|---|---|
| mBERT | `google-bert/bert-base-multilingual-cased` (some historical LAJ files use the equivalent `bert-base-multilingual-cased`) |
| XLM-R-base | `FacebookAI/xlm-roberta-base` (some historical LAJ files use the equivalent `xlm-roberta-base`) |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` |
| Gemma-3-12B-it | `google/gemma-3-12b-it` |
| DeepSeek | `deepseek-v4-flash`; LAJ exception described above |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` |

## Methodology boundary

The six systems are not trained under one identical regime:

- mBERT and XLM-R are supervised for LD, NLI, SA, POS, and DEPS; WSD and LAJ use frozen encoder representations/scores.
- Qwen, Gemma, DeepSeek, and Llama are zero-shot or frozen-model evaluations.
- Encoder and generative results should be displayed in separate method groups when the protocols differ.

Each task README and run guide contains the task-specific input schema and metric definition.
