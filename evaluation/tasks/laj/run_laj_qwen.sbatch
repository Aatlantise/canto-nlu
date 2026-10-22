#!/bin/bash
#SBATCH --job-name=laj-qwen
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/home/USER/laj-qwen/results/laj-qwen-%j.out

set -euo pipefail

source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen

mkdir -p /home/USER/laj-qwen/results

python -u /home/USER/laj-qwen/eval_qwen_laj.py \
  --pairs /home/USER/laj-qwen/comparison_test.jsonl \
  --source /home/USER/laj-qwen/cantonese.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --batch-size 4 \
  --output-dir /home/USER/laj-qwen/results
