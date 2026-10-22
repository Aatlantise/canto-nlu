#!/bin/bash
#SBATCH --job-name=nli-gemma
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/USER/nli-eval/results/logs/gemma-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/nli-eval/results/logs

python -u /home/USER/nli-eval/nli_prompted_eval.py \
  --data /home/USER/nli-eval/sample_nli.jsonl \
  --output-dir /home/USER/nli-eval/results/gemma-3-12b-it \
  --model google/gemma-3-12b-it \
  --batch-size 2 \
  --load-in-4bit
