#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_outputs/%j.out

set -eo pipefail
source "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/Run Scripts/shared/qwen35_job_env.sh"

RESULT_JSONL="results/results_natugu_Qwen35_newmaterials_cheatsheet_jpg.jsonl"
METRICS_JSON="metrics/qwen35/natugu/metrics_natugu_Qwen35_newmaterials_cheatsheet_jpg.json"

python "Qwen3.5 Experiments/run_grammamt_Qwen35_context.py" \
  --language Natugu \
  --model_id Qwen/Qwen3.5-9B \
  --support_n 21 \
  --test_n 99 \
  --grammar_image_dir "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/Natugu Grammar Materials/Boerger Natqgu grammar sketch final/Cheat Sheet from PDF" \
  --use_float32 \
  --out_metrics "${METRICS_JSON}" \
  --out_jsonl "${RESULT_JSONL}"
