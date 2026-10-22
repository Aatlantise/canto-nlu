#!/bin/bash
#SBATCH --job-name=pos-ds-smoke
#SBATCH --partition=spot
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/home/USER/pos-deepseek-smoke-%j.out
#SBATCH --export=ALL

set -euo pipefail
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is not set" >&2
  exit 2
fi
mkdir -p /home/USER/pos-results
python3 -u /home/USER/deepseek_v4_pos_eval.py \
  --data /home/USER/pos_train.jsonl \
  --output /home/USER/pos-results/deepseek-v4-flash-pos-smoke.jsonl \
  --model deepseek-v4-flash \
  --max-samples 10
