#!/bin/bash
set -eo pipefail

cd /dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT
source "Run Scripts/shared/openrouter_job_env.sh"

JOBS=(
  "Run Scripts/openrouter/run_gpt56luna_gitksan_pdf1.sh"
  "Run Scripts/openrouter/run_gpt56luna_gitksan_pdf2.sh"
  "Run Scripts/openrouter/run_gpt56luna_lezgi.sh"
  "Run Scripts/openrouter/run_gpt56luna_natugu.sh"
  "Run Scripts/openrouter/run_gpt56luna_tsez.sh"
)

job_ids=()
for job in "${JOBS[@]}"; do
  echo "[INFO] Submitting ${job}"
  job_id="$(sbatch --parsable "${job}")"
  echo "[INFO] Submitted ${job_id}"
  job_ids+=("${job_id}")
done

dependency="$(IFS=:; echo "${job_ids[*]}")"
score_id="$(sbatch --parsable --dependency="afterok:${dependency}" "Run Scripts/openrouter/scoring/run_gpt56luna_score_xcomet_all.sh")"
echo "[INFO] Submitted dependent XCOMET scorer ${score_id}"
