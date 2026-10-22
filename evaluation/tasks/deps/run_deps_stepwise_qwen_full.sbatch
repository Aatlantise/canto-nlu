#!/bin/bash
#SBATCH --job-name=deps-sw-qwen-full
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/home/USER/deps-sw-qwen-full-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/deps-results/stepwise-v2/heldout-test
python -u /home/USER/deps_stepwise_eval.py \
  --data /home/USER/deps_test.jsonl \
  --output /home/USER/deps-results/stepwise-v2/heldout-test/qwen-deps-stepwise-test.jsonl \
  --backend local \
  --model Qwen/Qwen2.5-7B-Instruct \
  --evaluator-version qwen2.5-7b-deps-stepwise-tsv-v2.1-recovery \
  --resume
