# CantoNLU

This repository accompanies our preprint: ["CantoNLU: A benchmark for Cantonese natural language understanding"](https://arxiv.org/html/2510.20670v1), where we introduce a general language understanding benchmark in Cantonese, in a collaboration with York Hay Ng, Sophia Chan, Helena Zhao, and Annie En-Shiun Lee.

## Setup

```bash
pip install -r requirements.txt
```

## Tasks

Each task has its own directory under `data/` with a readme that explains how the data is built.

| Task | Directory | Type | How to evaluate |
|---|---|---|---|
| Natural language inference | `data/nli` | sentence-pair classification | fine-tune (`--task=nli`) |
| Sentiment analysis | `data/sentiment` | classification | fine-tune (`--task=sentiment`) |
| Language detection | `data/ld` | classification | fine-tune (`--task=ld`) |
| Linguistic acceptability | `data/laj` | classification / pairwise comparison | fine-tune (`--task=laj`) or zero-shot comparison |
| Part-of-speech tagging | `data/pos` | token classification | fine-tune (`--task=pos`) |
| Dependency parsing | `data/deps` | parsing | fine-tune (`--task=deps`) |
| Word sense disambiguation | `data/wsd` | sentence-pair similarity | zero-shot |

## Usage

All commands are run from the repository root.

**Fine-tuning** on a task:

```bash
python run.py --task=sentiment --model_dir=./models/yue-monolingual
```

`--task` is one of `nli`, `sentiment`, `ld`, `laj`, `pos`, `deps`. Fine-tuned weights are not saved; neither are TensorBoard logs. Per-epoch validation metrics and final test-set metrics are printed to stdout.
Add `--eval_only` to evaluate a model on the test set without training (classification tasks only).
`--model_dir` accepts either a local checkpoint or a Hugging Face model ID, so the same
entry point can evaluate mBERT (`google-bert/bert-base-multilingual-cased`) and XLM-R-base
(`FacebookAI/xlm-roberta-base`) without task-specific scripts.

**Zero-shot evaluation**:

```bash
python data/wsd/evaluate_wsd.py ./models/yue-monolingual
python data/laj/laj_comparison/evaluate_comparison_laj.py ./models/yue-monolingual
```

For mBERT and XLM-R-base WSD, use the same encoder evaluator with either model ID:

```bash
python data/wsd/evaluate_wsd.py FacebookAI/xlm-roberta-base
```

**Instruction-tuned LMs** use one shared zero-shot entry point for all seven tasks:

```bash
# Qwen2.5, Llama 3.1, or another text-only Hugging Face chat model
python evaluate_llm.py --task sentiment \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output results/qwen-sentiment.jsonl

# Gemma 3 is detected automatically and loaded with its processor
python evaluate_llm.py --task wsd \
  --model google/gemma-3-12b-it \
  --output results/gemma-wsd.jsonl

# DeepSeek API (reads the key from DEEPSEEK_API_KEY)
python evaluate_llm.py --task deps --backend deepseek \
  --model deepseek-chat \
  --output results/deepseek-deps.jsonl
```

`--task` accepts `nli`, `sentiment`, `ld`, `laj`, `pos`, `deps`, or `wsd`.
The runner reuses the checked-in files under `data/`, writes resumable JSONL
predictions, and writes aggregate metrics beside them as `*.summary.json`.
Use `--max-samples` for a smoke test and `--load-in-4bit` for local CUDA inference
(the latter additionally requires `bitsandbytes`).

| Model family | Evaluation route |
|---|---|
| mBERT, XLM-R-base | `run.py` for supervised tasks; `data/wsd/evaluate_wsd.py` for WSD |
| Qwen2.5-7B-Instruct, Gemma-3-12B-it, Llama-3.1-8B-Instruct | `evaluate_llm.py --backend local` |
| DeepSeek | `evaluate_llm.py --backend deepseek` |

Encoder scores use supervised fine-tuning while instruction-tuned LM scores use
zero-shot prompting, so compare the two protocols with that distinction in mind.
No model weights, credentials, predictions, or duplicate dataset copies are stored
in the repository.

## AI assistance disclosure

The multi-model evaluation integration was developed with assistance from OpenAI
Codex. The generated changes were checked with syntax and data/parser smoke tests;
the contributor remains responsible for reviewing the code and reported results.

## Pre-trained model weights

We offer the following pre-trained encoder-only models:
* CantoBERT-mono: bert-base trained from scratch on monolingual Cantonese data
* CantoBERT-transfer: bert-base-chinese additionally pre-trained on Cantonese data
* CantoModernBERT-mono: modernbert-base trained from scratch on monolingual Cantonese data
* CantoModernBERT-transfer: chinese-modernbert-large-wwm additionally pre-trained on Cantonese data

https://drive.google.com/drive/folders/1RBZIQD5ectfb9m5sHDN5TIALEREm-tfH

## Acknowledgment
This readme.md has been drafted by Claude Code.
Please contact Junghyun Min for any questions, comments, or feedback.

## License
MIT
