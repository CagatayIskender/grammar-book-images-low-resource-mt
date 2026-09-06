#!/bin/bash
#SBATCH --job-name=mtob_gem_lez_gl
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=10:00:00
#SBATCH --output=slurm_outputs/%j.out

set -eo pipefail
export XDG_DATA_DIRS="${XDG_DATA_DIRS:-}"
source /etc/profile
if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
if type module >/dev/null 2>&1; then
  module purge || true
  module load python || true
fi
set -u
source "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/venv/bin/activate"
export PYTHONPATH="/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/MTOB Grammar Experiments/vendor:/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/MTOB Grammar Experiments:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV_FILE="${OPENROUTER_ENV_FILE:-/dss/dsshome1/07/ge92kun2/.config/grammamt/openrouter.env}"
if [ -f "${ENV_FILE}" ]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "[ERROR] OPENROUTER_API_KEY is not set" >&2
  exit 2
fi
cd "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/MTOB Grammar Experiments"

python run_experiment.py --model gemini25flashlite --source lezgi --condition gl
