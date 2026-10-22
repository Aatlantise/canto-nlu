#!/bin/bash
#SBATCH --job-name=nli-mbert
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/home/USER/nli-eval/results/logs/mbert-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/nli-eval/results/logs

python -u /home/USER/nli-eval/nli_encoder_finetune.py \
  --test-data /home/USER/nli-eval/sample_nli.jsonl \
  --output-dir /home/USER/nli-eval/results/mbert \
  --model google-bert/bert-base-multilingual-cased \
  --batch-size 16 \
  --eval-batch-size 64 \
  --learning-rate 2e-5 \
  --epochs 3 \
  --bf16
