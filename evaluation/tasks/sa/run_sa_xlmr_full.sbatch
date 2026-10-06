#!/bin/bash
#SBATCH --job-name=sa-xlmr-full-v1
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/USER/sa-xlmr-full-v1-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen

python -u /home/USER/sa_encoder_finetune.py \
  --train-data /home/USER/sa-data/train.jsonl \
  --valid-data /home/USER/sa-data/valid.jsonl \
  --test-data /home/USER/sa-data/test.jsonl \
  --output-dir /home/USER/sa-results/xlmr-base-full-v1 \
  --model FacebookAI/xlm-roberta-base \
  --epochs 3 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --max-length 256
