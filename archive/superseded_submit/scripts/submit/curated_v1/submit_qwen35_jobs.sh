#!/bin/bash
set -eo pipefail

cd /dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT

echo "[INFO] The Qwen 3.5 curated_v1 jobs are split by language/PDF to avoid LRZ QOS submit limits."
echo "[INFO] Submit one group at a time:"
echo "[INFO]   bash \"scripts/submit/curated_v1/submit_qwen35_gitksan_pdf1.sh\""
echo "[INFO]   bash \"scripts/submit/curated_v1/submit_qwen35_gitksan_pdf2.sh\""
echo "[INFO]   bash \"scripts/submit/curated_v1/submit_qwen35_lezgi.sh\""
echo "[INFO]   bash \"scripts/submit/curated_v1/submit_qwen35_natugu.sh\""
echo "[INFO]   bash \"scripts/submit/curated_v1/submit_qwen35_tsez.sh\""
echo "[INFO]"
echo "[INFO] Newmaterials group sizes are Gitksan PDF1=12, Gitksan PDF2=12, Lezgi=12, Natugu=12, Tsez=12 jobs."
echo "[INFO] Old/original jobs are separate:"
echo "[INFO]   bash \"scripts/submit/legacy/submit_old_qwen35_jobs.sh\""
echo "[INFO] Wait for some jobs to finish before submitting the next group if LRZ reaches the submit limit."
echo "[INFO] After generation finishes, submit XCOMET scoring with one of:"
echo "[INFO]   sbatch \"scripts/scoring/run_gitksan_qwen35_score_xcomet.sh\""
echo "[INFO]   sbatch \"scripts/scoring/run_lezgi_qwen35_score_xcomet.sh\""
echo "[INFO]   sbatch \"scripts/scoring/run_natugu_qwen35_score_xcomet.sh\""
echo "[INFO]   sbatch \"scripts/scoring/run_tsez_qwen35_score_xcomet.sh\""
echo "[INFO]   sbatch \"scripts/scoring/run_qwen35_score_xcomet_all.sh\""
