#!/bin/bash
#SBATCH --job-name=deps-sw-gemma-full
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/home/USER/deps-sw-gemma-full-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/deps-results/stepwise-v2/heldout-test
python -u /home/USER/deps_stepwise_eval.py \
  --data /home/USER/deps_test.jsonl \
  --output /home/USER/deps-results/stepwise-v2/heldout-test/gemma-deps-stepwise-test.jsonl \
  --backend gemma \
  --model google/gemma-3-12b-it \
  --evaluator-version gemma3-12b-deps-stepwise-tsv-v2.1-recovery \
  --load-in-4bit --resume
