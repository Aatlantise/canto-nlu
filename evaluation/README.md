# Cantonese NLU Evaluation Suite

整理后的多任务粤语 NLU evaluation 代码。这个目录是从原实验工作区生成的 GitHub 发布副本；原始文件未被移动或删除。

The canonical scope is exactly **7 tasks × 6 model families**. See [`COVERAGE_MATRIX.md`](COVERAGE_MATRIX.md) for the evaluator used by every combination and the DeepSeek LAJ model-variant caveat.

## Tasks

| Directory | Task | Main metric(s) | Evaluation families |
|---|---|---|---|
| `tasks/sa` | Sentiment analysis (SA) | Accuracy, macro-F1 | Fine-tuned encoders, local LLMs, DeepSeek API |
| `tasks/wsd` | Word-sense disambiguation (WSD) | Accuracy, macro-F1 | mBERT and zero-shot LLMs |
| `tasks/pos` | Part-of-speech tagging (POS) | Token accuracy, macro-F1 | Fine-tuned encoders and zero-shot LLMs |
| `tasks/laj` | Linguistic acceptability judgment (LAJ) | Forced-choice accuracy | MLM pseudo-surprisal and causal-LM surprisal |
| `tasks/ld` | Language detection (LD) | Accuracy, macro-F1 | Fine-tuned encoders and zero-shot LLMs |
| `tasks/nli` | Natural language inference (NLI) | Accuracy, macro-F1 | Fine-tuned encoders and prompted LLMs |
| `tasks/deps` | Dependency parsing (DEPS) | UAS, LAS | Fine-tuned encoders and stepwise prompted LLMs |

Compact result summaries and legacy evaluators are retained locally for provenance but ignored by default; the canonical GitHub upload contains only the final 7 × 6 code scope.

## Repository scope

Included:

- Python evaluators, training scripts, scoring utilities, and result summarizers.
- Slurm launch scripts used in the experiments.
- Existing run guides, manifests, and compact result reports.
- Task-specific dependency files.

Excluded:

- Datasets and benchmark source snapshots.
- Model weights, checkpoints, caches, raw predictions, logs, virtual environments, and `node_modules`.
- Duplicate ZIP bundles that contain copies of files already present here.
- API keys and Hugging Face tokens.

See `REPOSITORY_INVENTORY.md` for the detailed export policy.

## Data

The evaluators expect task-specific input files supplied through command-line arguments. Dataset files are not included because provenance and redistribution terms need to be reviewed separately. The original local benchmark snapshot identifies the upstream project as [CantoNLU](https://arxiv.org/html/2510.20670v1). POS/DEPS also refer to [UD Cantonese-HK](https://github.com/UniversalDependencies/UD_Cantonese-HK).

Expected schemas are documented by each task's existing run guide and argument parser. Do not commit private or license-restricted datasets to this repository.

## Environments

Create one environment per task; GPU packages and model families have different version and hardware requirements.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r tasks/<task>/requirements.txt
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1`.

Gemma and Llama checkpoints may require accepted Hugging Face licenses. API-based evaluators read credentials from environment variables shown in `.env.example`; no credential is stored in code.

## Running

Start with the README or run guide inside the relevant task directory. Most Python entry points expose their complete interface through:

```bash
python tasks/<task>/<script>.py --help
```

The committed `.sbatch` files are provenance-preserving experiment launchers. They still contain the original cluster paths such as `/home/USER` and the original Conda activation path. Before submitting them on another cluster, update the data, output, repository, partition, and environment paths described in `CLUSTER_PORTABILITY.md`.

## Before making the GitHub repository public

1. Choose and add a license. No license was inferred automatically.
2. Confirm dataset redistribution and benchmark citation requirements.
3. Replace cluster-specific paths in the Slurm files if portable launchers are required.
4. Review the staged diff, then initialize or copy this directory into the target GitHub repository.
