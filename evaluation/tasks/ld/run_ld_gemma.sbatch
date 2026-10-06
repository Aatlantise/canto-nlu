#!/bin/bash
#SBATCH --job-name=ld-gemma
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/home/USER/ld-eval/results/logs/gemma-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/ld-eval/results/logs /home/USER/ld-eval/results/gemma

python -u /home/USER/ld-eval/ld_prompted_eval.py \
  --data /home/USER/ld-eval/ld_test.jsonl \
  --output /home/USER/ld-eval/results/gemma/predictions.jsonl \
  --model google/gemma-3-12b-it \
  --family gemma3 \
  --batch-size 4 \
  --load-in-4bit \
  --resume
