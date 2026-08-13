#!/bin/bash
set -eo pipefail
cd /dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT

JOBS=(
  "Run Scripts/qwen3/lezgi/run_lezgi_qwen_newmaterials_pdfpages_impactful_jpg.sh"
  "Run Scripts/qwen3/lezgi/run_lezgi_qwen_modelgloss_newmaterials_pdfpages_impactful_jpg.sh"
  "Run Scripts/qwen35/lezgi/run_lezgi_qwen35_newmaterials_pdfpages_impactful_jpg.sh"
  "Run Scripts/qwen35/lezgi/run_lezgi_qwen35_modelgloss_newmaterials_pdfpages_impactful_jpg.sh"
)

for job in "${JOBS[@]}"; do
  echo "[INFO] Submitting ${job}"
  sbatch "${job}"
done

echo "[INFO] Submit XCOMET scoring after the Qwen 3.5 Lezgi page jobs finish:"
echo "sbatch \"Run Scripts/qwen35/scoring/run_lezgi_qwen35_score_xcomet_pdfpages_impactful.sh\""
