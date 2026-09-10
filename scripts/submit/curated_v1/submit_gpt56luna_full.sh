#!/bin/bash
set -eo pipefail

cd /dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT
source "scripts/shared/openrouter_job_env.sh"

JOBS=(
  "scripts/jobs/openrouter/run_gpt56luna_gitksan_pdf1.sh"
  "scripts/jobs/openrouter/run_gpt56luna_gitksan_pdf2.sh"
  "scripts/jobs/openrouter/run_gpt56luna_lezgi.sh"
  "scripts/jobs/openrouter/run_gpt56luna_natugu.sh"
  "scripts/jobs/openrouter/run_gpt56luna_tsez.sh"
)

job_ids=()
for job in "${JOBS[@]}"; do
  echo "[INFO] Submitting ${job}"
  job_id="$(sbatch --parsable "${job}")"
  echo "[INFO] Submitted ${job_id}"
  job_ids+=("${job_id}")
done

dependency="$(IFS=:; echo "${job_ids[*]}")"
score_id="$(sbatch --parsable --dependency="afterok:${dependency}" "scripts/scoring/run_gpt56luna_score_xcomet_all.sh")"
echo "[INFO] Submitted dependent XCOMET scorer ${score_id}"
