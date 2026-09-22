#!/bin/bash
#SBATCH --job-name=goldv2_gemini25flashlite_lezgi_chain_gloss
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/slurm_outputs/%j.out
set -eo pipefail
cd "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT"
source scripts/shared/openrouter_job_env.sh
export PYTHONUNBUFFERED=1
python runners/matched_gold_v2/run.py --group "configs/matched_gold_v2/groups/gemini25flashlite_lezgi_chain_gloss.json" --deadline-seconds 34200
