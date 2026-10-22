# Language Detection (LD)

This task classifies a sentence as Cantonese, Mandarin, or mixed Cantonese-Mandarin text.

| Label | Class |
| --- | --- |
| `0` | Cantonese |
| `1` | Mandarin |
| `2` | Mixed |

The primary metric is three-class macro F1. Invalid generated answers are counted as incorrect predictions.

## Dataset

The benchmark is derived from the first 10,000 sentence pairs in the [Cantonese-Chinese Parallel Corpus](https://huggingface.co/datasets/HKAllen/cantonese-chinese-parallel-corpus). Cantonese and Mandarin sentences form labels 0 and 1. Word-aligned cross-language substitutions produce the mixed examples for label 2.

All variants originating from one parallel sentence pair share a `source_id`. The 90/5/5 train/validation/test split is performed by `source_id`, so source pairs cannot leak between splits. Each JSONL record contains `id`, `source_id`, `sentence`, and `label`.

The fixed canonical files are preserved in the upstream CantoNLU history. `prepare_data.py` downloads them from an immutable commit and verifies both row counts and SHA-256 checksums.

| File | Rows | SHA-256 |
| --- | ---: | --- |
| `ld_train.jsonl` | 42,753 | `3bb990ebc17927290c516d8feaaeab58bc5576c70a83fe1d9d4da226814034a4` |
| `ld_val.jsonl` | 2,425 | `d6fdd177a5801d91920cba141b755744aeed2025f4255bf31d360e57d9adee83` |
| `ld_test.jsonl` | 2,341 | `3220bc2ab99c66426c67140dc9685db95dea1ae4cd14e8cd2de227343682f020` |

The original construction script is available in [upstream commit `861e6e6`](https://github.com/Aatlantise/canto-nlu/commit/861e6e66bcf25e5e2b226317e7b4c4a8574d6b5c). Downloading the fixed splits is recommended because changes in model and alignment-library versions can otherwise alter generated mixed examples.

## Setup

Run all commands from the repository root. On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r tasks\ld\requirements.txt
python tasks\ld\prepare_data.py
```

The final command creates `tasks/ld/data/` and verifies every split. Re-running it is safe: existing files are checked instead of downloaded again.

## Smoke Test

The following command evaluates one example with a small open-weight Qwen checkpoint:

```powershell
python tasks\ld\ld_prompted_eval.py `
  --data tasks\ld\data\ld_test.jsonl `
  --output outputs\ld\qwen_smoke.jsonl `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --family causal `
  --batch-size 1 `
  --limit 1 `
  --seed 42
```

A successful run writes:

- `outputs/ld/qwen_smoke.jsonl`: one prediction record;
- `outputs/ld/qwen_smoke.summary.json`: aggregate metrics and run metadata.

The first model download can take several minutes. A stationary progress display during inference does not necessarily indicate a failure, especially on CPU.

## Full Evaluation

Remove `--limit 1`, choose the intended checkpoint, and use a new output path:

```powershell
python tasks\ld\ld_prompted_eval.py `
  --data tasks\ld\data\ld_test.jsonl `
  --output outputs\ld\qwen_7b_predictions.jsonl `
  --model Qwen/Qwen2.5-7B-Instruct `
  --family causal `
  --batch-size 1 `
  --seed 42 `
  --resume
```

`ld_prompted_eval.py` also supports Gemma (`--family gemma3`) and Llama (`--family causal`). Use `--load-in-4bit` only on a compatible CUDA setup with `bitsandbytes` installed.

For mBERT or XLM-R fine-tuning, run:

```powershell
python tasks\ld\mbert_ld_finetune.py `
  --data-dir tasks\ld\data `
  --output-dir outputs\ld\mbert `
  --model google-bert/bert-base-multilingual-cased `
  --seed 42
```

For the DeepSeek API evaluator, set `DEEPSEEK_API_KEY` in the environment and see `LD_RUN.md`. Do not commit API keys, generated predictions, model caches, or checkpoints.

## Files

- `prepare_data.py`: downloads and verifies the canonical splits.
- `LD_MANIFEST.json`: records split sizes, checksums, labels, and evaluated models.
- `ld_metrics.py`: validates input records and computes shared metrics.
- `mbert_ld_finetune.py`: trains and evaluates mBERT or XLM-R.
- `ld_prompted_eval.py`: evaluates local Qwen, Gemma, or Llama checkpoints.
- `deepseek_v4_ld_eval_stdlib.py`: evaluates a DeepSeek API model.
- `run_ld_*.sbatch`: example Slurm jobs for full experiments.
- `LD_RUN.md`: additional cluster and model-specific commands.
