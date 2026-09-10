#!/bin/bash
set -eo pipefail
cd /dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT

JOBS=(
  "scripts/jobs/qwen3/lezgi/run_lezgi_qwen_curated_v1_pdfpages_impactful_jpg.sh"
  "scripts/jobs/qwen3/lezgi/run_lezgi_qwen_modelgloss_curated_v1_pdfpages_impactful_jpg.sh"
  "scripts/jobs/qwen35/lezgi/run_lezgi_qwen35_curated_v1_pdfpages_impactful_jpg.sh"
  "scripts/jobs/qwen35/lezgi/run_lezgi_qwen35_modelgloss_curated_v1_pdfpages_impactful_jpg.sh"
)

for job in "${JOBS[@]}"; do
  echo "[INFO] Submitting ${job}"
  sbatch "${job}"
done

echo "[INFO] Submit XCOMET scoring after the Qwen 3.5 Lezgi page jobs finish:"
echo "sbatch \"scripts/scoring/run_lezgi_qwen35_score_xcomet_pdfpages_impactful.sh\""
