#!/bin/bash
#SBATCH --job-name=mtob_q3_nat_ge
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_outputs/%j.out

set -euo pipefail
export XDG_DATA_DIRS="${XDG_DATA_DIRS:-}"
source /etc/profile
if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
if type module >/dev/null 2>&1; then
  module purge || true
  module load python || true
fi
source "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/venv/bin/activate"
source "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/Run Scripts/shared/cache_env.sh"
export PYTHONPATH="/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/MTOB Grammar Experiments/vendor:/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/MTOB Grammar Experiments:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/MTOB Grammar Experiments"

python run_experiment.py --model qwen3 --source natugu --condition ge
