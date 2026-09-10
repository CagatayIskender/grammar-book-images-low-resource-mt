#!/bin/bash
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/slurm_outputs/%j.out

set -eo pipefail
source "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/scripts/shared/openrouter_job_env.sh"
python "runners/openrouter/run_openrouter_suite.py" --source lezgi
