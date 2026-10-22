#!/bin/bash
#SBATCH --job-name=pos-gemma-smoke
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/home/USER/pos-gemma-smoke-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/pos-results

python -u /home/USER/gemma3_pos_eval.py \
  --data /home/USER/pos_train.jsonl \
  --output /home/USER/pos-results/gemma3-12b-pos-smoke.jsonl \
  --model google/gemma-3-12b-it \
  --max-samples 10 \
  --load-in-4bit
