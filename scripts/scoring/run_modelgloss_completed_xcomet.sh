#!/bin/bash
#SBATCH --job-name=modelgloss_completed_xcomet
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/slurm_outputs/%j.out
set -eo pipefail
cd "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT"
source venv/bin/activate
source scripts/shared/cache_env.sh
python -u runners/score_verified.py --targets configs/modelgloss_completed_scoring_targets.json
