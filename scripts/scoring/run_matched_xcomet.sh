#!/bin/bash
#SBATCH --job-name=matched_xcomet
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lrz-hgx-h100-015,lrz-hgx-h100-026
#SBATCH --mem=80G
#SBATCH --time=14:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/slurm_outputs/%j.out
set -eo pipefail
cd /dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT
source scripts/shared/cache_env.sh
source venv/bin/activate
python -u runners/matched/score.py --xcomet
