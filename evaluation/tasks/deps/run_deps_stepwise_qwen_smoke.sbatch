#!/bin/bash
#SBATCH --job-name=deps-sw-qwen-smoke
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/home/USER/deps-sw-qwen-smoke-%j.out

set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/deps-results/stepwise-v2/smoke
python -u /home/USER/deps_stepwise_eval.py \
  --data /home/USER/deps_train.jsonl \
  --output /home/USER/deps-results/stepwise-v2/smoke/qwen-deps-stepwise-dev-smoke.jsonl \
  --backend local \
  --model Qwen/Qwen2.5-7B-Instruct \
  --evaluator-version qwen2.5-7b-deps-stepwise-tsv-v2.1-recovery \
  --max-samples 10 --resume
