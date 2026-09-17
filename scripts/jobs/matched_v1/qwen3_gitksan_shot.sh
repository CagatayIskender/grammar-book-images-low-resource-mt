#!/bin/bash
#SBATCH --job-name=matched_qwen3_gitksan_shot
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lrz-hgx-h100-015,lrz-hgx-h100-026
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/slurm_outputs/%j.out
set -eo pipefail
cd "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT"
source scripts/shared/cache_env.sh
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python runners/matched/run.py --group "configs/matched_v1/groups/qwen3_gitksan_shot.json" --deadline-seconds 12600
