#!/bin/bash
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/slurm_outputs/%j.out

set -eo pipefail
source "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/scripts/shared/openrouter_job_env.sh"

python "runners/openrouter/run_openrouter_suite.py" \
  --model openai/gpt-5.6-luna \
  --model_label gpt56luna \
  --omit_temperature \
  --source gitksan_pdf1_brown \
  --test_n 2 \
  --conditions cheatsheet_txt,cheatsheet_jpg \
  --smoke

# Exercise the Natugu multimodal payload: 9 selected impactful pages.
python "runners/openrouter/run_openrouter_suite.py" \
  --model openai/gpt-5.6-luna \
  --model_label gpt56luna \
  --omit_temperature \
  --source natugu \
  --test_n 1 \
  --conditions pdfpages_impactful_jpg \
  --normal_only \
  --smoke
