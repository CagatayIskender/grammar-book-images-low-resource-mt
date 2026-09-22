#!/bin/bash
#SBATCH --job-name=goldv2_analysis
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=05:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/slurm_outputs/%j.out
set -eo pipefail
cd /dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT
source venv/bin/activate
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u runners/matched_gold_v2/analyze.py --samples 100000 --cohort native
python -u runners/matched_gold_v2/analyze.py --samples 100000 --cohort common99
