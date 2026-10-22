# Cantonese DEPS: paper-inspired stepwise TSV protocol (v2)

This is a separate, post-hoc experiment inspired by Matsuda, Ma, and Asahara
(IWPT 2025), "Step-by-step Instructions and a Simple Tabular Output Format
Improve the Dependency Parsing Accuracy of LLMs."

It does not overwrite or replace the original zero-shot JSON results.

## What changed

The original v1 prompt requested two JSON arrays (`heads` and `deprels`) at
once. The v2 prompt follows the paper's three-step, single-turn structure:

1. `ID FORM UPOS`
2. `ID FORM UPOS HEAD`
3. `ID FORM UPOS HEAD DEPREL`

Every stage is TSV. The evaluator recovers rows by token ID and exact FORM. It
also tolerates over-complete rows by taking a valid UPOS, the rightmost integer
HEAD before the relation, and the final DEPREL. Strict-format recall is retained
separately, so recovery never hides instruction-following failures. Missing or
malformed rows count as wrong only for those tokens. The final tree is still
checked for one root, legal heads, no self-heads, and no cycles.

UPOS is an auxiliary reasoning scaffold. The current DEPS JSONL files do not
contain gold UPOS, so UPOS accuracy is not reported. UAS and LAS remain the
primary metrics and include punctuation, matching v1.

## Methodological note

The paper's strong results used supervised fine-tuning (LoRA-SFT), not pure
zero-shot inference. This bundle adapts the prompt and table format to the
existing zero-shot comparison; it does not reproduce the paper's SFT setting.
Because v2 was designed after inspecting v1 test behavior, report it as a
separate paper-inspired/post-hoc protocol. Do not silently replace v1 scores.

The prompt is frozen in this bundle. Smoke runs use the first 10 training rows
for execution and format verification; they do not touch the held-out test set.

## Upload and unpack

Upload `deps_stepwise_v2_bundle.zip` to `/home/USER`, then run:

```bash
cd /home/USER
unzip -o deps_stepwise_v2_bundle.zip
python3 -m py_compile deps_stepwise_common.py deps_stepwise_eval.py deps_stepwise_rescore.py
```

## Qwen smoke and full

```bash
QWEN_SW_SMOKE=$(sbatch --parsable run_deps_stepwise_qwen_smoke.sbatch)
echo "Qwen stepwise smoke: $QWEN_SW_SMOKE"
```

After `COMPLETED 0:0`, inspect the smoke summary and submit the full run once:

```bash
QWEN_SW_FULL=$(sbatch --parsable run_deps_stepwise_qwen_full.sbatch)
echo "Qwen stepwise full: $QWEN_SW_FULL"
```

## Gemma smoke and full

```bash
GEMMA_SW_SMOKE=$(sbatch --parsable run_deps_stepwise_gemma_smoke.sbatch)
echo "Gemma stepwise smoke: $GEMMA_SW_SMOKE"
```

After `COMPLETED 0:0`:

```bash
GEMMA_SW_FULL=$(sbatch --parsable run_deps_stepwise_gemma_full.sbatch)
echo "Gemma stepwise full: $GEMMA_SW_FULL"
```

## Llama smoke and full

```bash
read -s -p "Hugging Face read token: " HF_TOKEN
echo
export HF_TOKEN
LLAMA_SW_SMOKE=$(sbatch --parsable --export=ALL run_deps_stepwise_llama_smoke.sbatch)
echo "Llama stepwise smoke: $LLAMA_SW_SMOKE"
unset HF_TOKEN
```

After successful smoke, repeat the hidden-token setup and submit:

```bash
LLAMA_SW_FULL=$(sbatch --parsable --export=ALL run_deps_stepwise_llama_full.sbatch)
echo "Llama stepwise full: $LLAMA_SW_FULL"
```

## DeepSeek smoke and full

```bash
read -s -p "DeepSeek API key: " DEEPSEEK_API_KEY
echo
export DEEPSEEK_API_KEY
DEEPSEEK_SW_SMOKE=$(sbatch --parsable --export=ALL run_deps_stepwise_deepseek_smoke.sbatch)
echo "DeepSeek stepwise smoke: $DEEPSEEK_SW_SMOKE"
unset DEEPSEEK_API_KEY
```

After successful smoke, repeat the hidden-key setup and submit:

```bash
DEEPSEEK_SW_FULL=$(sbatch --parsable --export=ALL run_deps_stepwise_deepseek_full.sbatch)
echo "DeepSeek stepwise full: $DEEPSEEK_SW_FULL"
```

## Monitor

```bash
squeue -u "$USER" -o "%.18i %.30j %.2t %.10M %.30R"
```

When a job leaves the queue:

```bash
sacct -j JOB_ID --format=JobID,JobName%30,State,Elapsed,ExitCode
```

Follow a running log, replacing `JOB_ID`:

```bash
tail -f /home/USER/deps-sw-qwen-smoke-JOB_ID.out
```

## Result locations

Smoke summaries:

```text
/home/USER/deps-results/stepwise-v2/smoke/*.summary.json
```

Full summaries:

```text
/home/USER/deps-results/stepwise-v2/heldout-test/*.summary.json
```

Key fields are `uas`, `las`, `invalid_output_rate`, `task1_token_recall`,
`task2_token_recall`, `task3_token_recall`, `strict_task3_token_recall`,
`complete_three_step_rate`, and `complete_recovered_three_step_rate`.

## Re-score an existing v2 file without rerunning the model

This is useful after a parser-only update. It makes no model or API calls:

```bash
python3 /home/USER/deps_stepwise_rescore.py \
  --input /home/USER/deps-results/stepwise-v2/smoke/qwen-deps-stepwise-smoke.jsonl \
  --output /home/USER/deps-results/stepwise-v2/smoke/qwen-deps-stepwise-smoke-rescored.jsonl \
  --evaluator-version qwen2.5-7b-deps-stepwise-tsv-v2.1-recovery
```

Example extraction:

```bash
python3 -c 'import json; d=json.load(open("/home/USER/deps-results/stepwise-v2/heldout-test/qwen-deps-stepwise-test.summary.json")); ks=["uas","las","non_punct_uas","non_punct_las","invalid_output_rate","task1_token_recall","task2_token_recall","task3_token_recall","complete_three_step_rate"]; print("\n".join(f"{k}: {d.get(k)}" for k in ks))'
```
