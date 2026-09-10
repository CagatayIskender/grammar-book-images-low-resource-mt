#!/bin/bash
set -euo pipefail
ROOT="/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/experiments/mtob"
cd "${ROOT}"

sbatch "${ROOT}/jobs/gpt56luna/lezgi/run_ge.sh"
sbatch "${ROOT}/jobs/gpt56luna/lezgi/run_gs.sh"
sbatch "${ROOT}/jobs/gpt56luna/lezgi/run_gl.sh"
