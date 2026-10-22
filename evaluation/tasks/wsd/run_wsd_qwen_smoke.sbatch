#!/bin/bash
#SBATCH --job-name=wsd-qwen-smoke
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/home/USER/wsd-results/qwen-smoke-%j.out
set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/wsd-results
python /home/USER/wsd_local_llm_eval.py \
 --data /home/USER/cantonlu/canto-nlu-main/wsd/test.jsonl \
 --output /home/USER/wsd-results/qwen2.5-7b-wsd-smoke.jsonl \
 --model Qwen/Qwen2.5-7B-Instruct --family qwen --samples-per-label 10
