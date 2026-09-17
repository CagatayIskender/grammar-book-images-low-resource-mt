#!/bin/bash
#SBATCH --job-name=matched_significance
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/slurm_outputs/%j.out
set -eo pipefail
cd /dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT
source venv/bin/activate
python -u runners/matched/analyze.py --samples 100000 --models qwen3 qwen35 gemini25flashlite gpt56luna --cohort native
