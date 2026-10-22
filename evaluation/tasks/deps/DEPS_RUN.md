# Cantonese dependency parsing evaluation

This bundle evaluates the same six models on a held-out dependency parsing test.

## Data and metrics

- Train: 903 sentences, 12,139 tokens
- Test: 101 sentences, 1,779 tokens
- Dependency relations: 49 in train; no unseen test relation
- Train/test exact token overlap: 0
- Train SHA-256: `4c78b7eeec458ec3a5c77e4bdee835dcb7881924cabd374099b2ca4c1323be82`
- Test SHA-256: `6b5b75bc9c1c97647f67b641e7d897edb119ef3f14a7a5723e987813a7a1a968`

Primary UAS and LAS include punctuation, matching the existing CantoNLU metric
implementation. Every summary also reports non-punctuation UAS/LAS.

- mBERT and XLM-R are supervised encoder parsers trained on the train file. Exact
  duplicate sentences stay in one side of a group-aware 90/10 train/validation split.
- Qwen, Gemma, DeepSeek, and Llama are zero-shot and receive only the test tokens.
- Encoder parsers use a learned arc/relation head and deterministic single-root,
  acyclic greedy decoding. The best checkpoint is chosen by validation LAS.
- The LLM prompt and parser are frozen before smoke tests. Smoke runs are execution
  checks only, not prompt development.

## Install

```bash
cd /home/USER
unzip -o deps_v1_bundle.zip
sha256sum deps_train.jsonl deps_test.jsonl
```

## Recommended first smoke tests

Use the two available GPU nodes for mBERT and Qwen. DeepSeek uses CPU/API and can
run at the same time.

```bash
MBERT_SMOKE=$(sbatch --parsable run_deps_mbert_smoke.sbatch)
QWEN_SMOKE=$(sbatch --parsable run_deps_qwen_smoke.sbatch)
echo "mBERT smoke: $MBERT_SMOKE"
echo "Qwen smoke: $QWEN_SMOKE"
```

For DeepSeek, enter the key without displaying it:

```bash
read -s -p "DeepSeek API key: " DEEPSEEK_API_KEY
echo
export DEEPSEEK_API_KEY
DEEPSEEK_SMOKE=$(sbatch --parsable --export=ALL run_deps_deepseek_smoke.sbatch)
echo "DeepSeek smoke: $DEEPSEEK_SMOKE"
unset DEEPSEEK_API_KEY
```

Monitor:

```bash
squeue -u "$USER" -o "%.18i %.30j %.2t %.10M %.30R"
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
```

Smoke logs use `/home/USER/deps-*-smoke-JOB_ID.out`. Do not submit a full job
until its corresponding smoke job finishes with `COMPLETED` and `0:0`.

## Remaining smoke tests

```bash
sbatch run_deps_xlmr_smoke.sbatch
sbatch run_deps_gemma_smoke.sbatch
```

Llama requires accepted model access and an exported Hugging Face read token:

```bash
read -s -p "Hugging Face read token: " HF_TOKEN
echo
export HF_TOKEN
LLAMA_SMOKE=$(sbatch --parsable --export=ALL run_deps_llama_smoke.sbatch)
echo "Llama smoke: $LLAMA_SMOKE"
unset HF_TOKEN
```

## Full runs

After all relevant smoke jobs succeed:

```bash
sbatch run_deps_mbert_full.sbatch
sbatch run_deps_xlmr_full.sbatch
sbatch run_deps_qwen_full.sbatch
sbatch run_deps_gemma_full.sbatch
```

Submit Llama and DeepSeek full runs from shells in which their respective tokens
have been entered and exported, just as in the smoke commands:

```bash
sbatch --export=ALL run_deps_llama_full.sbatch
sbatch --export=ALL run_deps_deepseek_full.sbatch
```

## Final summaries

```text
/home/USER/deps-results/mbert-full-v1/summary.json
/home/USER/deps-results/xlmr-full-v1/summary.json
/home/USER/deps-results/heldout-test/qwen2.5-7b-deps-test.summary.json
/home/USER/deps-results/heldout-test/gemma3-12b-deps-test.summary.json
/home/USER/deps-results/heldout-test/deepseek-v4-flash-deps-test.summary.json
/home/USER/deps-results/heldout-test/llama-3.1-8b-deps-test.summary.json
```

For encoder summaries, use `test_metrics.uas` and `test_metrics.las`. For LLM
summaries, use top-level `uas` and `las`.
