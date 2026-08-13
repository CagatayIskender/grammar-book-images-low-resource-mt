#!/bin/bash
set -euo pipefail
ROOT="/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/MTOB Grammar Experiments"
cd "${ROOT}"

sbatch "${ROOT}/jobs/gemini25flashlite/gitksan_pdf2/run_ge.sh"
sbatch "${ROOT}/jobs/gemini25flashlite/gitksan_pdf2/run_gs.sh"
sbatch "${ROOT}/jobs/gemini25flashlite/gitksan_pdf2/run_gl.sh"
