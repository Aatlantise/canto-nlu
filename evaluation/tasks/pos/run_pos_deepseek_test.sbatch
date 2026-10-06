#!/bin/bash
#SBATCH --job-name=pos-ds-test-v1
#SBATCH --partition=spot
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=/home/USER/pos-deepseek-test-v1-%j.out
#SBATCH --export=ALL

set -euo pipefail
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is not set" >&2
  exit 2
fi
mkdir -p /home/USER/pos-results/heldout-test

python3 -u /home/USER/deepseek_v4_pos_eval.py \
  --data /home/USER/pos_test.jsonl \
  --output /home/USER/pos-results/heldout-test/deepseek-v4-flash-pos-test.jsonl \
  --model deepseek-v4-flash \
  --resume
