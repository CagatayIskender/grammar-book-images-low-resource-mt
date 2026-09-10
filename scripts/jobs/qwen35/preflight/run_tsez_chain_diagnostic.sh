#!/bin/bash
#SBATCH --job-name=diag_qwen35_tsez_support_decoding
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/slurm_outputs/%j.out
set -eo pipefail
cd /dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT
source scripts/shared/qwen35_job_env.sh
export HF_HUB_OFFLINE=1
python -u runners/diagnose_tsez_chain.py --model qwen35
