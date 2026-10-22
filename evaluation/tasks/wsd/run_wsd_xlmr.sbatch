#!/bin/bash
#SBATCH --job-name=wsd-xlmr
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/home/USER/wsd-results/xlmr-wsd-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/wsd-results

python /home/USER/mbert_wsd_mask_cosine.py \
  --data /home/USER/cantonlu/canto-nlu-main/wsd/test.jsonl \
  --output /home/USER/wsd-results/xlmr-base-wsd-mask-cosine.jsonl \
  --model FacebookAI/xlm-roberta-base \
  --batch-size 32
