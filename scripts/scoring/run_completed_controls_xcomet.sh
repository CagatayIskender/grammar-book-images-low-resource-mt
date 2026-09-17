#!/bin/bash
#SBATCH --job-name=completed_controls_xcomet
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lrz-hgx-h100-026
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/slurm_outputs/%j.out
set -eo pipefail
cd "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT"
source venv/bin/activate
source scripts/shared/cache_env.sh
python -u scripts/score_completed_controls.py
