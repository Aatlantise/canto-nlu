#!/bin/bash
#SBATCH --job-name=laj-llama
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/home/USER/laj-models/results/llama-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/laj-models/results/llama

python -u /home/USER/laj-models/eval_causal_laj.py \
  --pairs /home/USER/laj-models/comparison_test.jsonl \
  --source /home/USER/laj-models/cantonese.jsonl \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --batch-size 4 \
  --output-dir /home/USER/laj-models/results/llama
