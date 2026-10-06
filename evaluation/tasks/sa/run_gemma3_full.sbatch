#!/bin/bash
#SBATCH --job-name=gemma3-sa
#SBATCH --partition=base
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/USER/logs/gemma3-sa-%j.out
#SBATCH --error=/home/USER/logs/gemma3-sa-%j.err

set -euo pipefail

source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen

echo "Started: $(date --iso-8601=seconds)"
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python /home/USER/gemma3_sa_eval.py \
  --data /home/USER/cantonlu/canto-nlu-main/openrice-senti/test.tsv \
  --output /home/USER/gemma-results/gemma-3-12b-it-4bit-sa-full.jsonl

echo "Finished: $(date --iso-8601=seconds)"
