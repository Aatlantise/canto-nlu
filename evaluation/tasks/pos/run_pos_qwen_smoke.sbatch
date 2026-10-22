#!/bin/bash
#SBATCH --job-name=pos-qwen-smoke
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/home/USER/pos-qwen-smoke-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/pos-results

python -u /home/USER/qwen_pos_eval.py \
  --data /home/USER/pos_train.jsonl \
  --output /home/USER/pos-results/qwen2.5-7b-pos-smoke.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --max-samples 10
