#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_outputs/%j.out

set -eo pipefail
source "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/Run Scripts/shared/qwen35_job_env.sh"

python "Qwen3.5 Experiments/run_grammamt_Qwen35_context.py" \
  --language Gitksan \
  --model_id Qwen/Qwen3.5-9B \
  --support_n 21 \
  --test_n 37 \
  --grammar_image_dir "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/Gitskan Grammar Screenshots/PDF 2 - Rigsby Intro/Summary Image with Tables/Model Input Readable" \
  --use_float32 \
  --out_metrics metrics/qwen35/gitksan/pdf2_rigsby/metrics_gitksan_Qwen35_pdf2_rigsby_summary_tables.json \
  --out_jsonl results/results_gitksan_Qwen35_pdf2_rigsby_summary_tables.jsonl
