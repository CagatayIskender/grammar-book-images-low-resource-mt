#!/bin/bash
set -eo pipefail

cd /dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT
source "scripts/shared/openrouter_job_env.sh"
sbatch "scripts/jobs/openrouter/run_gemini25flashlite_smoke.sh"
