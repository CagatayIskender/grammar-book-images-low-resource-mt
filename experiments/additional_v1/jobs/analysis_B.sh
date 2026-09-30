#!/bin/bash
#SBATCH --job-name=addv1_analysis_B
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --mem=24G
#SBATCH --cpus-per-task=1
#SBATCH --time=05:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/experiments/additional_v1/logs/%j.out
set -eo pipefail
cd "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT"
source scripts/shared/cache_env.sh
source venv/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/experiments/additional_v1/analyze.py" --package B
