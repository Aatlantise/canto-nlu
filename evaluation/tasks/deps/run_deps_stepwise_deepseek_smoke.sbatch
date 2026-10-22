#!/bin/bash
#SBATCH --job-name=deps-sw-ds-smoke
#SBATCH --partition=spot
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/home/USER/deps-sw-ds-smoke-%j.out
#SBATCH --export=ALL

set -euo pipefail
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then echo "DEEPSEEK_API_KEY is not set" >&2; exit 2; fi
mkdir -p /home/USER/deps-results/stepwise-v2/smoke
python3 -u /home/USER/deps_stepwise_eval.py \
  --data /home/USER/deps_train.jsonl \
  --output /home/USER/deps-results/stepwise-v2/smoke/deepseek-deps-stepwise-dev-smoke.jsonl \
  --backend deepseek \
  --model deepseek-v4-flash \
  --evaluator-version deepseek-v4-deps-stepwise-tsv-v2.1-recovery \
  --max-samples 10 --resume
