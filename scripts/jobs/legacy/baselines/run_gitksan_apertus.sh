#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=70G
#SBATCH --time=00:30:00
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
mkdir -p metrics/baseline/apertus/gitksan/grammar/original results/baseline/apertus/gitksan/grammar/original
source venv/bin/activate
source "scripts/shared/cache_env.sh"

python runners/baseline/run_grammamt_Apertus.py \
  --language Gitksan \
  --model_id swiss-ai/Apertus-8B-Instruct-2509 \
  --support_n 21 \
  --test_n 37 \
  --use_float32 \
  --out_metrics metrics/baseline/apertus/gitksan/grammar/original/metrics_gitksan_Apertus.json \
  --out_jsonl results/baseline/apertus/gitksan/grammar/original/results_gitksan_Apertus.jsonl
