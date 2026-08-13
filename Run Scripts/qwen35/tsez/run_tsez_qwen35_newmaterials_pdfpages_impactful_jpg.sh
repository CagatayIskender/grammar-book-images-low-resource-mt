#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_outputs/%j.out

set -eo pipefail
source "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/Run Scripts/shared/qwen35_job_env.sh"

mkdir -p metrics/qwen35/tsez results slurm_outputs

RESULT_JSONL="results/results_tsez_Qwen35_newmaterials_pdfpages_impactful_jpg.jsonl"
METRICS_JSON="metrics/qwen35/tsez/metrics_tsez_Qwen35_newmaterials_pdfpages_impactful_jpg.json"

python "Qwen3.5 Experiments/run_grammamt_Qwen35_context.py" \
  --language Tsez \
  --model_id Qwen/Qwen3.5-9B \
  --support_n 21 \
  --test_n 445 \
  --grammar_image_dir "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/Tsez Grammar Materials/Tsez_Grammar_PDF/Pages from the PDF/Selected impactful pages JPG" \
  --use_float32 \
  --out_metrics "${METRICS_JSON}" \
  --out_jsonl "${RESULT_JSONL}"
