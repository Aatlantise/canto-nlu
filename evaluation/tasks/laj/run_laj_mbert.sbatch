#!/bin/bash
#SBATCH --job-name=laj-mbert
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/USER/laj-models/results/mbert-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/laj-models/results/mbert

python -u /home/USER/laj-models/eval_mlm_laj.py \
  --pairs /home/USER/laj-models/comparison_test.jsonl \
  --source /home/USER/laj-models/cantonese.jsonl \
  --model bert-base-multilingual-cased \
  --batch-size 32 \
  --output-dir /home/USER/laj-models/results/mbert
