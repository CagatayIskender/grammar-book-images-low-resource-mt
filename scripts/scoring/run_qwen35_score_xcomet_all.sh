#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/slurm_outputs/%j.out

set -eo pipefail

cd /dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT
source "scripts/shared/cache_env.sh"

BASE_PYTHON="/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/venv/bin/python"
SCORER="runners/qwen35/score_qwen35_xcomet_from_jsonl.py"

for prefix in \
  results_gitksan_Qwen35 \
  results_lezgi_Qwen35 \
  results_natugu_Qwen35 \
  results_tsez_Qwen35
do
  echo "[INFO] Scoring ${prefix}"
  "${BASE_PYTHON}" "${SCORER}" \
    --results_dir results \
    --metrics_dir metrics \
    --pattern_prefix "${prefix}"
done
