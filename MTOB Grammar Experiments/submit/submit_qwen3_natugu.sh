#!/bin/bash
set -euo pipefail
ROOT="/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/MTOB Grammar Experiments"
cd "${ROOT}"

sbatch "${ROOT}/jobs/qwen3/natugu/run_ge.sh"
sbatch "${ROOT}/jobs/qwen3/natugu/run_gs.sh"
sbatch "${ROOT}/jobs/qwen3/natugu/run_gl.sh"
