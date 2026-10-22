#!/bin/bash
#SBATCH --job-name=wsd-gemma-full
#SBATCH --partition=spot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=/home/USER/wsd-results/gemma-full-%j.out
set -euo pipefail
source /home/share/apps/python/anaconda3-3.11/etc/profile.d/conda.sh
conda activate cantonlu-qwen
mkdir -p /home/USER/wsd-results
python /home/USER/wsd_local_llm_eval.py \
 --data /home/USER/cantonlu/canto-nlu-main/wsd/test.jsonl \
 --output /home/USER/wsd-results/gemma-3-12b-it-4bit-wsd-full.jsonl \
 --model google/gemma-3-12b-it --family gemma
