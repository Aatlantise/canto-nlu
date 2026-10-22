# Cantonese NLU Evaluation Suite


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

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1`.

Gemma and Llama checkpoints may require accepted Hugging Face licenses. API-based evaluators read credentials from environment variables shown in `.env.example`; no credential is stored in code.

## Running

Start with the README or run guide inside the relevant task directory. Most Python entry points expose their complete interface through:

```bash
python tasks/<task>/<script>.py --help
```

The committed `.sbatch` files are provenance-preserving experiment launchers. They still contain the original cluster paths such as `/home/USER` and the original Conda activation path. Before submitting them on another cluster, update the data, output, repository, partition, and environment paths described in `CLUSTER_PORTABILITY.md`.

## Publication and safety

The evaluation code is distributed under the parent repository's license. Dataset
and model licenses remain separate and must be followed when downloading those
artifacts. Never commit API keys, Hugging Face tokens, model weights, caches,
checkpoints, or generated predictions. Replace cluster-specific paths in the Slurm
files before running on another system.

## Reproducibility Quick Start

Each task has its own dependency file, data-preparation command, smoke test,
and full evaluation instructions.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies for only the task that you want to reproduce:

```bash
python -m pip install -r tasks/<task>/requirements.txt
```

### Reproduction Status

| Task | Data preparation | Evaluation guide | Status |
|---|---|---|---|
| POS | `python tasks/pos/prepare_data.py` | [POS README](tasks/pos/README.md) | Reproducible |
| DEPS | `python tasks/deps/prepare_data.py` | [DEPS README](tasks/deps/README.md) | Reproducible |
| SA | `python tasks/sa/prepare_data.py` | [SA README](tasks/sa/README.md) | Reproducible |
| NLI | `python tasks/nli/prepare_data.py` | [NLI README](tasks/nli/README.md) | Reproducible |
| WSD | See command below | [WSD README](tasks/wsd/README.md) | Reproducible |
| LAJ | `python tasks/laj/prepare_data.py` | [LAJ README](tasks/laj/README.md) | Reproducible |
| LD | `python tasks/ld/prepare_data.py` | [LD README](tasks/ld/README.md) | Reproducible |

Prepare the WSD dataset with:

```bash
python tasks/wsd/prepare_data.py \
  --input tasks/wsd/senses.csv \
  --output tasks/wsd/test.jsonl
```

### Smoke Tests

The documented smoke tests use small models and one or two examples. They
verify the complete pipeline:

```text
data preparation
→ data validation
→ model loading
→ inference
→ metric calculation
→ output generation
```

Smoke-test metrics must not be reported as benchmark results. The full
evaluation commands use the original model identifiers and complete test sets.

### Generated Files

Datasets downloaded or generated by the preparation scripts are ignored by
Git. Model predictions, summaries, checkpoints, and caches are also ignored.

Each evaluation guide documents:

- Dataset source and preparation
- Input schema
- Dataset size and checksum
- Environment setup
- Smoke-test command
- Full-evaluation command
- Expected outputs and metrics
- Hardware requirements

### Reproducibility Boundary

The LD preparation script downloads the fixed canonical splits from an immutable
upstream CantoNLU commit and verifies their row counts and SHA-256 checksums. The
LD task README also links the historical construction script. This preserves the
exact evaluation inputs even if alignment-library or model versions later change.

The historical Slurm scripts preserve the original cluster configuration.
Users must update cluster paths, environment names, partitions, and output
directories before submitting them on another system.
