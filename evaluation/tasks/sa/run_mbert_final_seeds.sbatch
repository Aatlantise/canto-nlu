#!/bin/bash
#SBATCH --job-name=mbert-final
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/home/USER/mbert-final-%A_%a.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen

seeds=(42 52 62)
seed="${seeds[$SLURM_ARRAY_TASK_ID]}"

python /home/USER/mbert_sa_finetune.py \
  --data-dir /home/USER/cantonlu/canto-nlu-main/openrice-senti \
  --output-dir "/home/USER/mbert-final/seed${seed}" \
  --learning-rate 2e-5 \
  --batch-size 32 \
  --epochs 3 \
  --seed "$seed"
