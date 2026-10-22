#!/bin/bash
#SBATCH --job-name=deps-mbert-full-v1
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/USER/deps-mbert-full-v1-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/deps-results/mbert-full-v1
python -u /home/USER/encoder_deps_finetune.py \
  --train-data /home/USER/deps_train.jsonl \
  --test-data /home/USER/deps_test.jsonl \
  --output-dir /home/USER/deps-results/mbert-full-v1 \
  --model google-bert/bert-base-multilingual-cased \
  --evaluator-version mbert-deps-supervised-v2-cycle-safe \
  --epochs 20 --batch-size 16 --learning-rate 2e-5 --max-length 256
