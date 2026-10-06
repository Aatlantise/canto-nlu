#!/bin/bash
#SBATCH --job-name=mbert-pos-smoke-v3
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/home/USER/mbert-pos-smoke-v3-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/pos-results/mbert-smoke-v3

python -u /home/USER/mbert_pos_finetune.py \
  --train-data /home/USER/pos_train.jsonl \
  --test-data /home/USER/pos_test.jsonl \
  --output-dir /home/USER/pos-results/mbert-smoke-v3 \
  --model google-bert/bert-base-multilingual-cased \
  --epochs 1 \
  --batch-size 8 \
  --max-train-samples 32 \
  --max-eval-samples 16
