#!/bin/bash
set -eo pipefail
cd /dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT

echo "[INFO] Qwen 3 newmaterials jobs are split by language:"
echo "[INFO]   bash \"Run Scripts/submit/submit_qwen3_gitksan.sh\""
echo "[INFO]   bash \"Run Scripts/submit/submit_qwen3_lezgi.sh\""
echo "[INFO]   bash \"Run Scripts/submit/submit_qwen3_natugu.sh\""
echo "[INFO]   bash \"Run Scripts/submit/submit_qwen3_tsez.sh\""
echo "[INFO]"
echo "[INFO] Newmaterials group sizes are Gitksan PDF1=12, Gitksan PDF2=12, Lezgi=12, Natugu=12, Tsez=12 jobs."
echo "[INFO] Old/original jobs are separate:"
echo "[INFO]   bash \"Run Scripts/submit/submit_old_qwen3_jobs.sh\""
