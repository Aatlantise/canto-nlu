#!/bin/bash
#SBATCH --job-name=mbert-epoch5
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/home/USER/mbert-epoch5-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen

python /home/USER/mbert_sa_finetune.py \
  --data-dir /home/USER/cantonlu/canto-nlu-main/openrice-senti \
  --output-dir /home/USER/mbert-search/lr2e-5-bs32-ep5-seed42 \
  --learning-rate 2e-5 \
  --batch-size 32 \
  --epochs 5 \
  --seed 42 \
  --validation-only
