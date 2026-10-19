# Open-weight LLM full-test evaluation

This runbook evaluates three local instruction-tuned model families on every
non-LAJ full test set. LAJ remains separate because it uses pairwise sentence
surprisal rather than generated class labels.

## Fixed scope

Dataset baseline: `Aatlantise/canto-nlu@5c42cb1`.

Models:

* `Qwen/Qwen2.5-7B-Instruct`
* `google/gemma-3-12b-it`
* `meta-llama/Llama-3.1-8B-Instruct`

All three models are loaded locally in 4-bit NF4 mode. DeepSeek's hosted API is
not part of this open-weight run.

| Task argument | Test input | Expected examples |
|---|---|---:|
| `nli` | `data/nli/full_test.jsonl` | Recorded by the exporter; must exceed 100 |
| `sentiment` | `data/sentiment/test.jsonl` | 999 |
| `ld` | `data/ld/lid_test.jsonl` | 6,000 |
| `pos` | `data/pos/pos_test.jsonl` | 101 |
| `deps` | `data/deps/deps_test.jsonl` | 101 |
| `wsd` | `data/wsd/test.jsonl` | 1,646 |

The upstream directory is still named `data/ld`, and the CLI retains `--task
ld` for compatibility. The current dataset is the new grammar-derived LID task.
Its third class is `other`: sentences with both Cantonese and Mandarin features,
or with neither language's exclusive features.

## 1. Prepare the environment

From the repository root on the cluster:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install bitsandbytes
huggingface-cli login
```

Gemma and Llama require accepted Hugging Face model licenses. Put model and
dataset caches on cluster scratch storage when the home-directory quota is
small, for example by exporting `HF_HOME` before submission.

## 2. Export the complete NLI test

```bash
python data/nli/export_full_test.py
```

Do not substitute `data/nli/sample_nli.jsonl`; that file contains only 100
human-annotation examples. The exporter writes:

* `data/nli/full_test.jsonl`
* `data/nli/full_test.metadata.json`

Both generated files are ignored by Git. Preserve them with the run artifacts
because the metadata contains the upstream dataset fingerprint and SHA-256.

## 3. Submit the 18 full runs

The Slurm file is an array of 3 models times 6 tasks. Create the log directory
before calling `sbatch`, because Slurm opens the output file before the job
script runs.

```bash
mkdir -p logs
sbatch slurm/run_open_weight_full.sbatch
```

The default maximum concurrency is three jobs. Override cluster-specific
resources at submission time when necessary, for example:

```bash
sbatch --partition=spot --time=36:00:00 --array=0-17%2 \
  slurm/run_open_weight_full.sbatch
```

If the environment is outside the repository, provide its interpreter or
activation script:

```bash
PYTHON_BIN=/path/to/env/bin/python sbatch slurm/run_open_weight_full.sbatch
```

or:

```bash
VENV_ACTIVATE=/path/to/env/bin/activate sbatch slurm/run_open_weight_full.sbatch
```

Submit from the repository root. Otherwise export `REPO_ROOT` to the absolute
checkout path. Each job checks the dataset row count before model loading, uses
greedy decoding, resumes an interrupted prediction file, and writes to:

```text
results/open_weight_full/<model-key>/<task>.jsonl
results/open_weight_full/<model-key>/<task>.summary.json
```

## 4. Monitor and resume

```bash
squeue -u "$USER"
```

Failed or interrupted array elements can be resubmitted by ID. Existing valid
prediction rows are skipped because the evaluator receives `--resume`.

```bash
sbatch --array=FAILED_IDS slurm/run_open_weight_full.sbatch
```

Do not change the model, task data, prompt, or quantization settings while
resuming an existing output file. Use a new results directory after any such
change.

## 5. Verify completion

After the array finishes:

```bash
python verify_open_weight_full.py
```

Verification fails unless all 18 prediction and summary pairs exist, their row
counts match the full datasets, the model and task names match, and every
summary records the current dataset SHA-256. It writes the compact audit table:

```text
results/open_weight_full/manifest.csv
```

Invalid model outputs remain visible in the manifest but do not imply that the
test set was partial. Investigate high invalid-output rates before reporting
model quality.
