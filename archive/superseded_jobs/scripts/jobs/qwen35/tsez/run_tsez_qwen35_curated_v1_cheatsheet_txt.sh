#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_outputs/%j.out

set -eo pipefail
source "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/scripts/shared/qwen35_job_env.sh"

mkdir -p metrics/qwen35/tsez results slurm_outputs

RESULT_JSONL="results/curated_v1/qwen35/tsez/grammar/original/results_tsez_Qwen35_curated_v1_cheatsheet_txt.jsonl"
METRICS_JSON="metrics/curated_v1/qwen35/tsez/grammar/original/metrics_tsez_Qwen35_curated_v1_cheatsheet_txt.json"

python "runners/qwen35/run_grammamt_Qwen35_context.py" \
  --language Tsez \
  --model_id Qwen/Qwen3.5-9B \
  --support_n 21 \
  --test_n 445 \
  --grammar_text_file "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/materials/curated_v1/tsez/grammar/original/cheatsheet/compact_cheatsheet.txt" \
  --use_float32 \
  --out_metrics "${METRICS_JSON}" \
  --out_jsonl "${RESULT_JSONL}"
