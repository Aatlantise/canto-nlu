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

**Zero-shot evaluation**:

```bash
python data/wsd/evaluate_wsd.py ./models/yue-monolingual
python data/laj/laj_comparison/evaluate_comparison_laj.py ./models/yue-monolingual
```

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