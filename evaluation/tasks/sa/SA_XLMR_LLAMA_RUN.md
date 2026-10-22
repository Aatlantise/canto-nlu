# XLM-R-base and Llama-3.1-8B-Instruct Cantonese sentiment evaluation

## Evaluation protocol

- Task: CantoNLU OpenRice three-way sentiment classification.
- Labels: `smile` (positive), `ok` (neutral or mixed), `cry` (negative).
- Primary metric: macro-F1. Also report accuracy, per-class precision/recall/F1,
  confusion matrix, and invalid-output rate for the generative model.
- XLM-R-base: supervised training on `train.jsonl`, best-checkpoint selection by
  validation macro-F1 on `valid.jsonl`, then one evaluation on `test.jsonl`.
- Llama-3.1-8B-Instruct: frozen zero-shot, greedy decoding, evaluated only on
  `test.jsonl`. The 90-row CSV is a balanced subset of the test set and is used
  only as a runtime smoke test.
- Seed: 42.

The two rows answer different practical questions. XLM-R measures supervised task
adaptation; Llama measures instruction-following without task training. Do not present
their difference as an architecture-only comparison.

## Data audit

| File | Rows | smile | ok | cry |
|---|---:|---:|---:|---:|
| `train.jsonl` | 9,999 | 3,333 | 3,333 | 3,333 |
| `valid.jsonl` | 999 | 333 | 333 | 333 |
| `test.jsonl` | 999 | 333 | 333 | 333 |
| `sentiment_sample.csv` | 90 | 30 | 30 | 30 |

There are no duplicate texts within a split and no text overlap across train,
validation, and test. All 90 CSV rows occur in `test.jsonl` with matching labels.

## Upload

Create a data directory on the GPU server and upload the bundle plus the four data
files. From Windows PowerShell:

```powershell
scp "C:\path\to\LLM-evaluation\sa_xlmr_llama_v1_bundle.zip" USER@HOST:/home/USER/
scp "C:\path\to\downloads\train.jsonl" USER@HOST:/home/USER/sa-data/
scp "C:\path\to\downloads\valid.jsonl" USER@HOST:/home/USER/sa-data/
scp "C:\path\to\downloads\test.jsonl" USER@HOST:/home/USER/sa-data/
scp "C:\path\to\downloads\sentiment_sample.csv" USER@HOST:/home/USER/sa-data/
```

On the server:

```bash
cd /home/USER
mkdir -p sa-data sa-results
unzip -o sa_xlmr_llama_v1_bundle.zip
python -m pip install -r requirements_sa.txt
```

## XLM-R smoke and full run

```bash
XLMR_SMOKE_JOB=$(sbatch --parsable run_sa_xlmr_smoke.sbatch)
echo "XLM-R smoke job: $XLMR_SMOKE_JOB"
```

After it succeeds:

```bash
cat /home/USER/sa-results/xlmr-base-smoke-v1/summary.json
XLMR_JOB=$(sbatch --parsable run_sa_xlmr_full.sbatch)
echo "XLM-R full job: $XLMR_JOB"
```

## Llama access, smoke, and full run

Accept access to `meta-llama/Llama-3.1-8B-Instruct` on Hugging Face. In the shell
that calls `sbatch`, enter the read token without echoing it:

```bash
read -s -p "Hugging Face read token: " HF_TOKEN
echo
export HF_TOKEN
test -n "$HF_TOKEN" && echo "HF token is set"
LLAMA_SMOKE_JOB=$(sbatch --parsable run_sa_llama_smoke.sbatch)
echo "Llama smoke job: $LLAMA_SMOKE_JOB"
```

After the smoke job succeeds:

```bash
cat /home/USER/sa-results/llama-3.1-8b-smoke-v1.summary.json
LLAMA_JOB=$(sbatch --parsable run_sa_llama_full.sbatch)
echo "Llama full job: $LLAMA_JOB"
```

Never put the Hugging Face token in a script, command argument, screenshot, or chat.

## Monitor and compare

```bash
squeue -u "$USER" -o "%.18i %.30j %.2t %.10M %.30R"
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
```

After both full jobs finish:

```bash
python /home/USER/summarize_sa_results.py \
  --xlmr /home/USER/sa-results/xlmr-base-full-v1/summary.json \
  --llama /home/USER/sa-results/llama-3.1-8b-full-v1.summary.json \
  --output /home/USER/sa-results/SA_COMPARISON.md
cat /home/USER/sa-results/SA_COMPARISON.md
```
