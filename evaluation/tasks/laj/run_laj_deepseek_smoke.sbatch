#!/bin/bash
#SBATCH --job-name=laj-ds-smoke
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --output=/home/USER/laj-qwen/results/deepseek-smoke-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen

mkdir -p /home/USER/laj-qwen/results/deepseek-smoke

python -u /home/USER/laj-qwen/eval_qwen_laj.py \
  --pairs /home/USER/laj-qwen/comparison_test.jsonl \
  --source /home/USER/laj-qwen/cantonese.jsonl \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --family qwen \
  --load-in-4bit \
  --batch-size 2 \
  --limit 10 \
  --output-dir /home/USER/laj-qwen/results/deepseek-smoke
