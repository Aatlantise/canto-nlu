# XLM-R and Llama Cantonese POS evaluation

Both jobs run on the Slurm GPU server with one NVIDIA L4 each.

## Protocol

- XLM-R-base is supervised: train on `pos_train.jsonl`, choose the best checkpoint
  using a group-aware 10% validation split, then evaluate `pos_test.jsonl` once.
- Llama 3.1 8B Instruct is zero-shot: use the frozen LLM POS prompt and evaluate
  `pos_test.jsonl` only.
- Seed: 42.
- Test set: 101 sentences, 1,779 tokens.

## Upload

```bash
cd /home/USER
unzip -o xlmr_llama_pos_v1_bundle.zip
grep EVALUATOR_VERSION mbert_pos_finetune.py
```

## XLM-R smoke and full run

```bash
XLMR_SMOKE_JOB=$(sbatch --parsable run_pos_xlmr_smoke.sbatch)
echo "XLM-R smoke: $XLMR_SMOKE_JOB"
```

After it finishes successfully:

```bash
cat /home/USER/pos-results/xlmr-smoke-v1/summary.json
XLMR_JOB=$(sbatch --parsable run_pos_xlmr_full.sbatch)
echo "XLM-R full: $XLMR_JOB"
```

## Llama access and run

First sign in to Hugging Face in a browser, accept the model license at
`https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct`, and create a read token.
In the same server shell used to call `sbatch`, enter the token without displaying it:

```bash
read -s -p "Hugging Face read token: " HF_TOKEN
echo
export HF_TOKEN
test -n "$HF_TOKEN" && echo "HF token is set"
LLAMA_JOB=$(sbatch --parsable run_pos_llama_test.sbatch)
echo "Llama: $LLAMA_JOB"
```

Never paste the token into a command, script, screenshot, or chat.

## Monitor

```bash
squeue -u "$USER" -o "%.18i %.30j %.2t %.10M %.30R"
```

After completion, verify the exit state:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
```

## Results

```text
/home/USER/pos-results/xlmr-base-full-v1/summary.json
/home/USER/pos-results/xlmr-base-full-v1/test_predictions.jsonl
/home/USER/pos-results/heldout-test/llama-3.1-8b-pos-test.summary.json
/home/USER/pos-results/heldout-test/llama-3.1-8b-pos-test.jsonl
```
