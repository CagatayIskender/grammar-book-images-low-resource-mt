#!/bin/bash
#SBATCH --job-name=baseline_paired_significance
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/slurm_outputs/%j.out
set -euo pipefail
cd "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
venv/bin/python -u scripts/analyze_baseline_significance.py --models qwen3 qwen35 --samples 10000 --output "${SIGNIFICANCE_OUTPUT:-docs/statistical_significance/with_qwen35}"
