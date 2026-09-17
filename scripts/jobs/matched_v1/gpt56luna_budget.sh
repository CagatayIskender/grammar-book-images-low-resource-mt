#!/bin/bash
#SBATCH --job-name=matched_gpt56luna_budget
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/slurm_outputs/%j.out
set -eo pipefail
cd "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT"
source scripts/shared/openrouter_job_env.sh
export MATCHED_API_BUDGET_USD=4
export PYTHONUNBUFFERED=1
python runners/matched/run.py --group configs/matched_v1/groups/gpt56luna_budget.json --deadline-seconds 34200
