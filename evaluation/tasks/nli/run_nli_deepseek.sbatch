#!/bin/bash
#SBATCH --job-name=nli-deepseek
#SBATCH --partition=spot
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=/home/USER/nli-eval/results/logs/deepseek-%j.out

set -euo pipefail
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is not set" >&2
  exit 1
fi
mkdir -p /home/USER/nli-eval/results/deepseek-flash /home/USER/nli-eval/results/logs

python3 -u /home/USER/nli-eval/nli_deepseek_eval_stdlib.py \
  --data /home/USER/nli-eval/sample_nli.jsonl \
  --output /home/USER/nli-eval/results/deepseek-flash/predictions.jsonl \
  --model deepseek-v4-flash \
  --resume
