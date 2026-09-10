#!/bin/bash
set -euo pipefail
ROOT="/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/experiments/mtob"
cd "${ROOT}"

sbatch "${ROOT}/jobs/qwen35/gitksan_pdf2/run_ge.sh"
sbatch "${ROOT}/jobs/qwen35/gitksan_pdf2/run_gs.sh"
sbatch "${ROOT}/jobs/qwen35/gitksan_pdf2/run_gl.sh"
