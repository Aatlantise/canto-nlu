#!/bin/bash
#SBATCH --job-name=mbert-pos-full-v3
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/USER/mbert-pos-full-v3-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/pos-results/mbert-full-v3

python -u /home/USER/mbert_pos_finetune.py \
  --train-data /home/USER/pos_train.jsonl \
  --test-data /home/USER/pos_test.jsonl \
  --output-dir /home/USER/pos-results/mbert-full-v3 \
  --model google-bert/bert-base-multilingual-cased \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --max-length 256
