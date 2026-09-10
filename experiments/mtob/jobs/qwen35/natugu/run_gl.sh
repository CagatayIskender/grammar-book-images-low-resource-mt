#!/bin/bash
#SBATCH --job-name=mtob_q35_nat_gl
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/experiments/mtob/slurm_outputs/%j.out

set -eo pipefail
export XDG_DATA_DIRS="${XDG_DATA_DIRS:-}"
source /etc/profile
if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
if type module >/dev/null 2>&1; then
  module purge || true
  module load python || true
fi
set -u
source "/dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/ge92kun2/envs/grammamt-qwen35/bin/activate"
source "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/scripts/shared/cache_env.sh"
export PYTHONPATH="/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/experiments/mtob/vendor:/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/experiments/mtob:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/experiments/mtob"

python run_experiment.py --model qwen35 --source natugu --condition gl
