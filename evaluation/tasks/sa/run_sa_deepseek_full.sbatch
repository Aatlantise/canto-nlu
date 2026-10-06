#!/bin/bash
#SBATCH --job-name=sa-deepseek
#SBATCH --partition=spot
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=/home/USER/sa-results/deepseek-full-%j.out

set -euo pipefail
if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
  echo "DEEPSEEK_API_KEY is not set" >&2
  exit 1
fi
mkdir -p /home/USER/sa-results

python3 -u /home/USER/deepseek_v4_sa_eval_stdlib.py \
  --data /home/USER/cantonlu/canto-nlu-main/openrice-senti/test.tsv \
  --output /home/USER/sa-results/deepseek-v4-flash-sa-full.jsonl \
  --model deepseek-v4-flash \
  --resume
