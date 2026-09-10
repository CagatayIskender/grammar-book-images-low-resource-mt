#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/slurm_outputs/%j.out

set -eo pipefail
source "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/scripts/shared/qwen35_job_env.sh"
mkdir -p metrics/legacy/qwen35/gitksan/pdf1_brown/original results/legacy/qwen35/gitksan/pdf1_brown/original

python "runners/qwen35/run_grammamt_Qwen35_context.py" \
  --language Gitksan \
  --model_id Qwen/Qwen3.5-9B \
  --support_n 21 \
  --test_n 37 \
  --grammar_image_dir "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/materials/legacy/gitksan/pdf1_brown/Pages from the PDF/PDF Pages All JPG" \
  --model_gloss \
  --use_float32 \
  --out_metrics metrics/legacy/qwen35/gitksan/pdf1_brown/original/metrics_gitksan_Qwen35_ModelGloss_pdf1_brown_pdfpages_all_jpg.json \
  --out_jsonl results/legacy/qwen35/gitksan/pdf1_brown/original/results_gitksan_Qwen35_ModelGloss_pdf1_brown_pdfpages_all_jpg.jsonl
