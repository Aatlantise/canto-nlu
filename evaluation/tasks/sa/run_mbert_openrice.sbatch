#!/bin/bash
#SBATCH --job-name=mbert-openrice
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/home/USER/mbert-results/slurm-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/mbert-results

python /home/USER/mbert_sa_finetune.py \
  --data-dir /home/USER/cantonlu/canto-nlu-main/openrice-senti \
  --output-dir /home/USER/mbert-results/openrice-sa
