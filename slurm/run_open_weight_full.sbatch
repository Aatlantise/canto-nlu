#!/bin/bash
#SBATCH --job-name=cantonlu-open-weight
#SBATCH --array=0-17%3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/open-weight-full-%A_%a.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "$REPO_ROOT" || ! -f "$REPO_ROOT/evaluate_llm.py" ]]; then
  echo "Set REPO_ROOT to the canto-nlu checkout, or submit from its root." >&2
  exit 2
fi

cd "$REPO_ROOT"

if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  # VENV_ACTIVATE must name the activation script on the cluster.
  source "$VENV_ACTIVATE"
fi

models=(
  "Qwen/Qwen2.5-7B-Instruct"
  "google/gemma-3-12b-it"
  "meta-llama/Llama-3.1-8B-Instruct"
)
model_keys=("qwen2.5-7b" "gemma-3-12b" "llama-3.1-8b")
tasks=("nli" "sentiment" "ld" "pos" "deps" "wsd")
data_paths=(
  "data/nli/full_test.jsonl"
  "data/sentiment/test.jsonl"
  "data/ld/lid_test.jsonl"
  "data/pos/pos_test.jsonl"
  "data/deps/deps_test.jsonl"
  "data/wsd/test.jsonl"
)
expected_samples=(0 999 6000 101 101 1646)

array_id="${SLURM_ARRAY_TASK_ID:?This file must be submitted as a Slurm array job.}"
model_index=$((array_id / ${#tasks[@]}))
task_index=$((array_id % ${#tasks[@]}))

model="${models[$model_index]}"
model_key="${model_keys[$model_index]}"
task="${tasks[$task_index]}"
data_path="${data_paths[$task_index]}"
expected="${expected_samples[$task_index]}"

if [[ ! -f "$data_path" ]]; then
  echo "Missing $data_path. For NLI, run data/nli/export_full_test.py first." >&2
  exit 2
fi

if [[ "$task" == "nli" ]]; then
  metadata="data/nli/full_test.metadata.json"
  if [[ ! -f "$metadata" ]]; then
    echo "Missing $metadata; regenerate NLI with export_full_test.py." >&2
    exit 2
  fi
  expected="$($PYTHON_BIN -c 'import json; print(json.load(open("data/nli/full_test.metadata.json", encoding="utf-8"))["output_rows"])')"
  if (( expected <= 100 )); then
    echo "NLI metadata reports only $expected rows; refusing a sample-only run." >&2
    exit 2
  fi
fi

actual="$($PYTHON_BIN -c 'import sys; print(sum(1 for line in open(sys.argv[1], encoding="utf-8") if line.strip()))' "$data_path")"
if [[ "$actual" != "$expected" ]]; then
  echo "Expected $expected rows in $data_path, found $actual." >&2
  exit 2
fi

output="results/open_weight_full/$model_key/$task.jsonl"
mkdir -p "$(dirname "$output")"

echo "model=$model"
echo "task=$task"
echo "data=$data_path"
echo "expected_samples=$expected"
echo "output=$output"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

command=(
  "$PYTHON_BIN" -u evaluate_llm.py
  --task "$task"
  --model "$model"
  --data "$data_path"
  --output "$output"
  --expected-samples "$expected"
  --load-in-4bit
  --resume
)

srun "${command[@]}"
