#!/bin/bash
#SBATCH --job-name=ld-xlmr
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/home/USER/ld-eval/results/logs/xlmr-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/ld-eval/results/logs

python -u /home/USER/ld-eval/mbert_ld_finetune.py \
  --data-dir /home/USER/ld-eval \
  --output-dir /home/USER/ld-eval/results/xlmr \
  --model FacebookAI/xlm-roberta-base \
  --epochs 3 \
  --batch-size 32 \
  --learning-rate 2e-5 \
  --seed 42
