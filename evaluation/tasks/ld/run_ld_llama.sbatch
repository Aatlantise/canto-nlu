#!/bin/bash
#SBATCH --job-name=ld-llama
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/USER/ld-eval/results/logs/llama-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/ld-eval/results/logs /home/USER/ld-eval/results/llama

python -u /home/USER/ld-eval/ld_prompted_eval.py \
  --data /home/USER/ld-eval/ld_test.jsonl \
  --output /home/USER/ld-eval/results/llama/predictions.jsonl \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --family causal \
  --batch-size 16 \
  --resume
