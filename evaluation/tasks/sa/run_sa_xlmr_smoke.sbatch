#!/bin/bash
#SBATCH --job-name=sa-xlmr-smoke-v1
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=/home/USER/sa-xlmr-smoke-v1-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen

python -u /home/USER/sa_encoder_finetune.py \
  --train-data /home/USER/sa-data/train.jsonl \
  --valid-data /home/USER/sa-data/valid.jsonl \
  --test-data /home/USER/sa-data/test.jsonl \
  --output-dir /home/USER/sa-results/xlmr-base-smoke-v1 \
  --model FacebookAI/xlm-roberta-base \
  --epochs 1 \
  --batch-size 16 \
  --max-length 256 \
  --max-train-samples 300 \
  --max-valid-samples 90 \
  --max-test-samples 90
