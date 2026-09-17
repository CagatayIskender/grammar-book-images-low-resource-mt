#!/bin/bash
#SBATCH --job-name=mtob_vs_grammamt
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
set -euo pipefail
cd "${GRAMMAMT_COMPARISON_ROOT:?Submit through submit.sh}"
export PYTHONPATH="$PWD/../vendor:$PWD/.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
../../../venv/bin/python -u compare.py analyze
