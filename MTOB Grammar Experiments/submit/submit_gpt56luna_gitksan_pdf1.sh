#!/bin/bash
set -euo pipefail
ROOT="/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/MTOB Grammar Experiments"
cd "${ROOT}"

sbatch "${ROOT}/jobs/gpt56luna/gitksan_pdf1/run_ge.sh"
sbatch "${ROOT}/jobs/gpt56luna/gitksan_pdf1/run_gs.sh"
sbatch "${ROOT}/jobs/gpt56luna/gitksan_pdf1/run_gl.sh"
