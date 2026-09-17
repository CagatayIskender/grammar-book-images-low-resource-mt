#!/bin/bash
#SBATCH --job-name=mtob_eval_v1_analyze
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/experiments/mtob/slurm_outputs/%j.out
set -euo pipefail
cd /dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/experiments/mtob
export PYTHONPATH="$PWD/vendor:$PWD"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
../../venv/bin/python -u finalize_results.py --stage analyze
