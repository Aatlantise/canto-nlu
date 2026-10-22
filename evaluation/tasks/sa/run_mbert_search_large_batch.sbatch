#!/bin/bash
#SBATCH --job-name=mbert-search-large-bs
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=0-5%1
#SBATCH --output=/home/USER/mbert-search-large-bs-%A_%a.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen

learning_rates=(1e-5 1e-5 2e-5 2e-5 3e-5 3e-5)
batch_sizes=(32 64 32 64 32 64)
lr="${learning_rates[$SLURM_ARRAY_TASK_ID]}"
bs="${batch_sizes[$SLURM_ARRAY_TASK_ID]}"
run_name="lr${lr}-bs${bs}-ep3-seed42"

python /home/USER/mbert_sa_finetune.py \
  --data-dir /home/USER/cantonlu/canto-nlu-main/openrice-senti \
  --output-dir "/home/USER/mbert-search/${run_name}" \
  --learning-rate "$lr" \
  --batch-size "$bs" \
  --epochs 3 \
  --seed 42 \
  --validation-only
