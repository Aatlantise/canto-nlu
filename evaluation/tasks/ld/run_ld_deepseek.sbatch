#!/bin/bash
#SBATCH --job-name=ld-deepseek
#SBATCH --partition=spot
#SBATCH --cpus-per-task=2
#SBATCH --time=03:00:00
#SBATCH --output=/home/USER/ld-eval/results/logs/deepseek-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/ld-eval/results/logs /home/USER/ld-eval/results/deepseek

python -u /home/USER/ld-eval/deepseek_v4_ld_eval_stdlib.py \
  --data /home/USER/ld-eval/ld_test.jsonl \
  --output /home/USER/ld-eval/results/deepseek/predictions.jsonl \
  --model deepseek-v4-flash \
  --resume
