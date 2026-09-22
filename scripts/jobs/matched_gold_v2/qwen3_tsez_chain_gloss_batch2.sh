#!/bin/bash
#SBATCH --job-name=goldv2_qwen3_tsez_chain_gloss_batch2
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lrz-hgx-h100-015,lrz-hgx-h100-026
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/slurm_outputs/%j.out
set -eo pipefail
cd "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT"
source scripts/shared/cache_env.sh
source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
python runners/matched_gold_v2/run.py --group configs/matched_gold_v2/groups/qwen3_tsez_chain_gloss_batch2.json --deadline-seconds 34200
