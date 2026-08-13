#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_outputs/%j.out

set -eo pipefail
source "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/Run Scripts/shared/qwen35_job_env.sh"

RESULT_JSONL="results/results_lezgi_Qwen35_newmaterials_cyrillic_summary_tables_jpg.jsonl"
METRICS_JSON="metrics/qwen35/lezgi/metrics_lezgi_Qwen35_newmaterials_cyrillic_summary_tables_jpg.json"

python "Qwen3.5 Experiments/run_grammamt_Qwen35_context.py" \
  --language Lezgi \
  --model_id Qwen/Qwen3.5-9B \
  --support_n 21 \
  --test_n 87 \
  --grammar_image_dir "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/Lezgi Grammar Materials/Lezgi Grammar PDF Cyrillic/Summary Image with Tables" \
  --use_float32 \
  --out_metrics "${METRICS_JSON}" \
  --out_jsonl "${RESULT_JSONL}"
