#!/bin/bash
#SBATCH --job-name=sa-qwen-full
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/USER/sa-results/qwen-full-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/sa-results

python -u /home/USER/qwen_sa_eval.py \
  --data /home/USER/cantonlu/canto-nlu-main/openrice-senti/test.tsv \
  --output /home/USER/sa-results/qwen2.5-7b-instruct-sa-full.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct
