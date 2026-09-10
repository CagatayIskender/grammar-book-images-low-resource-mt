#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/slurm_outputs/%j.out

set -eo pipefail

export XDG_DATA_DIRS="${XDG_DATA_DIRS:-}"
source /etc/profile
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

set -u

if type module >/dev/null 2>&1; then
  module purge || true
  module load python || true
fi

cd /dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT
mkdir -p metrics/baseline/llama/gitksan/grammar/original results/baseline/llama/gitksan/grammar/original
source venv/bin/activate
source "scripts/shared/cache_env.sh"

python runners/baseline/run_grammamt.py \
  --language Gitksan \
  --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --support_n 21 \
  --test_n 1 \
  --use_float32 \
  --out_metrics metrics/baseline/llama/gitksan/grammar/original/metrics_gitksan_Llama_3.1.json \
  --out_jsonl results/baseline/llama/gitksan/grammar/original/results_gitksan_Llama_3.1.jsonl
