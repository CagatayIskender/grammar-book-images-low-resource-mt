#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_outputs/%j.out

set -eo pipefail
source "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/Run Scripts/shared/qwen35_job_env.sh"

RESULT_JSONL="results/results_lezgi_Qwen35_ModelGloss_grammar_pdfpages_all_jpg.jsonl"
METRICS_JSON="metrics/qwen35/lezgi/metrics_lezgi_Qwen35_ModelGloss_grammar_pdfpages_all_jpg.json"

python "Qwen3.5 Experiments/run_grammamt_Qwen35_context.py" \
  --language Lezgi \
  --model_id Qwen/Qwen3.5-9B \
  --support_n 21 \
  --test_n 87 \
  --grammar_image_dir "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/Lezgi Grammar Screenshots/Lezgi Grammar PDF/Pages from the PDF/PDF Pages All JPG" \
  --model_gloss \
  --use_float32 \
  --out_metrics "${METRICS_JSON}" \
  --out_jsonl "${RESULT_JSONL}"

deactivate || true
source venv/bin/activate
export LD_LIBRARY_PATH="/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/venv/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
echo "[INFO] Rescoring with XCOMET using base venv: $(which python)"
python "Qwen3.5 Experiments/score_qwen35_xcomet_from_jsonl.py" --result_jsonl "${RESULT_JSONL}" --overwrite
