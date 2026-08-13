#!/bin/bash
set -euo pipefail
ROOT="/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/MTOB Grammar Experiments"
cd "${ROOT}"

sbatch "${ROOT}/jobs/qwen3/lezgi/run_ge.sh"
sbatch "${ROOT}/jobs/qwen3/lezgi/run_gs.sh"
sbatch "${ROOT}/jobs/qwen3/lezgi/run_gl.sh"
