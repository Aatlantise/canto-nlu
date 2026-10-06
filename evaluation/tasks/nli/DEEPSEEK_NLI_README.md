# DeepSeek Cantonese NLI evaluation

Files:

- `nli_deepseek_eval_stdlib.py`: dependency-free evaluator using the official DeepSeek Chat Completions API.
- `sample_nli.jsonl`: balanced 100-example evaluation sample.

The recorded run requested the legacy API name `deepseek-v4-flash`. On 2026-09-25 the API returned `deepseek-flash`, which corresponds to DeepSeek-V4.1-Flash. The evaluator records both `requested_model` and `served_model` for auditability.

## Run

Run each command separately. Never paste a command block into the hidden key prompt.

```bash
cd /path/to/extracted/folder
```

```bash
read -rsp "DeepSeek API key: " DEEPSEEK_API_KEY
```

At the prompt, paste only the API key and press Enter. Then run:

```bash
echo
export DEEPSEEK_API_KEY
python -u nli_deepseek_eval_stdlib.py --data sample_nli.jsonl --output results/deepseek-flash/predictions.jsonl --model deepseek-v4-flash --resume
unset DEEPSEEK_API_KEY
```

The output files are:

- `results/deepseek-flash/predictions.jsonl`
- `results/deepseek-flash/predictions.summary.json`

The completed run reported 78/100 correct, 78.00% accuracy, and zero invalid predictions. Because the provider served `deepseek-flash`, report the model as DeepSeek-V4.1-Flash or include a routing footnote if retaining the legacy requested name.
