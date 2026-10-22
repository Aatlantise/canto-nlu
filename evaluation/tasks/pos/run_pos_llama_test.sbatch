#!/bin/bash
#SBATCH --job-name=pos-llama-test-v1
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/home/USER/pos-llama-test-v1-%j.out
#SBATCH --export=ALL

set -euo pipefail
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set; accept the Llama license and export a read token" >&2
  exit 2
fi
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/pos-results/heldout-test

python -u /home/USER/llama_pos_eval.py \
  --data /home/USER/pos_test.jsonl \
  --output /home/USER/pos-results/heldout-test/llama-3.1-8b-pos-test.jsonl \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --load-in-4bit \
  --resume
