#!/bin/bash
#SBATCH --job-name=deps-sw-llama-smoke
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/home/USER/deps-sw-llama-smoke-%j.out
#SBATCH --export=ALL

set -euo pipefail
if [[ -z "${HF_TOKEN:-}" ]]; then echo "HF_TOKEN is not set" >&2; exit 2; fi
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/deps-results/stepwise-v2/smoke
python -u /home/USER/deps_stepwise_eval.py \
  --data /home/USER/deps_train.jsonl \
  --output /home/USER/deps-results/stepwise-v2/smoke/llama-deps-stepwise-dev-smoke.jsonl \
  --backend local \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --evaluator-version llama3.1-8b-deps-stepwise-tsv-v2.1-recovery \
  --load-in-4bit --max-samples 10 --resume
