#!/bin/bash
#SBATCH --job-name=wsd-deepseek
#SBATCH --partition=spot
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=/home/USER/wsd-results/deepseek-wsd-%j.out

set -euo pipefail
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is not set" >&2
  exit 1
fi
mkdir -p /home/USER/wsd-results

python3 -u /home/USER/deepseek_v4_wsd_eval_stdlib.py \
  --data /home/USER/cantonlu/canto-nlu-main/wsd/test.jsonl \
  --output /home/USER/wsd-results/deepseek-v4-flash-wsd-full.jsonl \
  --model deepseek-v4-flash \
  --resume
