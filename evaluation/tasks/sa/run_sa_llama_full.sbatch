#!/bin/bash
#SBATCH --job-name=sa-llama-full-v1
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/USER/sa-llama-full-v1-%j.out
#SBATCH --export=ALL

set -euo pipefail
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set; accept the Llama license and export a read token" >&2
  exit 2
fi
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen

python -u /home/USER/sa_llama_eval.py \
  --data /home/USER/sa-data/test.jsonl \
  --output /home/USER/sa-results/llama-3.1-8b-full-v1.jsonl \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --load-in-4bit \
  --resume
