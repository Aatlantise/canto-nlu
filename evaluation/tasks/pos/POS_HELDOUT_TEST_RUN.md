# Cantonese POS held-out test evaluation

This bundle evaluates the three frozen zero-shot LLM pipelines on the supplied
held-out test set only.

- Test sentences: 101
- Test tokens: 1,779
- Test SHA-256: `e4a66cdd9ad4330e1afcc13a1b1ea07f6d9bc39ca960db3b68d0cb8364b7271b`
- Seed: 42
- Prompt and parsing policy: unchanged from the pilot runs

The previous 903-sentence results are development/pilot results and must not be
reported as held-out test performance.

## Install

```bash
cd /home/USER
unzip -o pos_heldout_test_v1_bundle.zip
sha256sum pos_test.jsonl
```

## Submit Qwen and Gemma

Both request one L4 GPU. They may be submitted together and Slurm will schedule
them according to available nodes.

```bash
QWEN_JOB=$(sbatch --parsable run_pos_qwen_test.sbatch)
GEMMA_JOB=$(sbatch --parsable run_pos_gemma_test.sbatch)
echo "Qwen: $QWEN_JOB"
echo "Gemma: $GEMMA_JOB"
```

## Submit DeepSeek

Set the API key in the login shell without displaying it, then submit from that
same shell so `sbatch --export=ALL` receives it.

```bash
read -s -p "DeepSeek API key: " DEEPSEEK_API_KEY
echo
export DEEPSEEK_API_KEY
test -n "$DEEPSEEK_API_KEY" && echo "API key is set"
DEEPSEEK_JOB=$(sbatch --parsable run_pos_deepseek_test.sbatch)
echo "DeepSeek: $DEEPSEEK_JOB"
```

Do not paste the API key directly into a command, screenshot, script, or chat.

## Monitor

```bash
squeue -u "$USER" -o "%.18i %.30j %.2t %.10M %.30R"
```

Logs use these paths, with the actual Slurm job ID substituted for `%j`:

```text
/home/USER/pos-qwen-test-v1-%j.out
/home/USER/pos-gemma-test-v1-%j.out
/home/USER/pos-deepseek-test-v1-%j.out
```

After a job disappears from `squeue`, verify its exit state:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,ExitCode
```

## Results

```text
/home/USER/pos-results/heldout-test/qwen2.5-7b-pos-test.summary.json
/home/USER/pos-results/heldout-test/gemma3-12b-pos-test.summary.json
/home/USER/pos-results/heldout-test/deepseek-v4-flash-pos-test.summary.json
```

The matching prediction files have the same names without `.summary`.
