#!/bin/bash
#SBATCH --job-name=deps-mbert-smoke
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/home/USER/deps-mbert-smoke-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/deps-results/mbert-smoke
python -u /home/USER/encoder_deps_finetune.py \
  --train-data /home/USER/deps_train.jsonl \
  --test-data /home/USER/deps_test.jsonl \
  --output-dir /home/USER/deps-results/mbert-smoke \
  --model google-bert/bert-base-multilingual-cased \
  --evaluator-version mbert-deps-supervised-v2-cycle-safe \
  --epochs 1 --batch-size 8 --max-train-samples 32 --max-eval-samples 16
