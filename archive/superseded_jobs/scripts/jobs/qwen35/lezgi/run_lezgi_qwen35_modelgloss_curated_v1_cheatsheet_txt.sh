#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_outputs/%j.out

set -eo pipefail
source "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/scripts/shared/qwen35_job_env.sh"

RESULT_JSONL="results/curated_v1/qwen35/lezgi/grammar/original/results_lezgi_Qwen35_ModelGloss_curated_v1_cheatsheet_txt.jsonl"
METRICS_JSON="metrics/curated_v1/qwen35/lezgi/grammar/original/metrics_lezgi_Qwen35_ModelGloss_curated_v1_cheatsheet_txt.json"

python "runners/qwen35/run_grammamt_Qwen35_context.py" \
  --language Lezgi \
  --model_id Qwen/Qwen3.5-9B \
  --support_n 21 \
  --test_n 87 \
  --grammar_text_file "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/materials/curated_v1/lezgi/grammar/original/cheatsheet/compact_cheatsheet.txt" \
  --model_gloss \
  --use_float32 \
  --out_metrics "${METRICS_JSON}" \
  --out_jsonl "${RESULT_JSONL}"
