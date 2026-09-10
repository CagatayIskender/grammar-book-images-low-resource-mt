#!/bin/bash

export XDG_DATA_DIRS="${XDG_DATA_DIRS:-}"
source /etc/profile
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

set -u

if type module >/dev/null 2>&1; then
  module purge || true
  module load python || true
  module list || true
fi

cd /dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT
source "scripts/shared/cache_env.sh"

QWEN35_VENV="${QWEN35_VENV:-/dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/ge92kun2/envs/grammamt-qwen35}"
if [ ! -x "${QWEN35_VENV}/bin/python" ]; then
  echo "[ERROR] Qwen3.5 environment not found at ${QWEN35_VENV}" >&2
  echo "[ERROR] Create it with: bash \"runners/qwen35/setup_qwen35_env.sh\"" >&2
  exit 2
fi

source "${QWEN35_VENV}/bin/activate"
export LD_LIBRARY_PATH="/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/venv/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[INFO] Python: $(which python)"
python --version
python - <<'PY'
import transformers
print(f"[INFO] transformers: {transformers.__version__}")
PY
