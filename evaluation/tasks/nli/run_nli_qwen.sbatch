#!/bin/bash
#SBATCH --job-name=nli-qwen
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/home/USER/nli-eval/results/logs/qwen-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/nli-eval/results/logs

python -u /home/USER/nli-eval/nli_prompted_eval.py \
  --data /home/USER/nli-eval/sample_nli.jsonl \
  --output-dir /home/USER/nli-eval/results/qwen2.5-7b-instruct \
  --model Qwen/Qwen2.5-7B-Instruct \
  --batch-size 4 \
  --load-in-4bit
