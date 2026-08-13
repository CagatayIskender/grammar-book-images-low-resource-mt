#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_outputs/%j.out

set -eo pipefail
source "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/Run Scripts/shared/qwen35_job_env.sh"

RESULT_JSONL="results/results_natugu_Qwen35_newmaterials_summary_text_txt.jsonl"
METRICS_JSON="metrics/qwen35/natugu/metrics_natugu_Qwen35_newmaterials_summary_text_txt.json"

python "Qwen3.5 Experiments/run_grammamt_Qwen35_context.py" \
  --language Natugu \
  --model_id Qwen/Qwen3.5-9B \
  --support_n 21 \
  --test_n 99 \
  --grammar_text_file "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/Natugu Grammar Materials/Boerger Natqgu grammar sketch final/Summary Text/summary_model_readable.txt" \
  --use_float32 \
  --out_metrics "${METRICS_JSON}" \
  --out_jsonl "${RESULT_JSONL}"
