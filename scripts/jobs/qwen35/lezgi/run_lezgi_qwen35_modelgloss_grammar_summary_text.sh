#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/slurm_outputs/%j.out

set -eo pipefail
source "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/scripts/shared/qwen35_job_env.sh"
mkdir -p metrics/legacy/qwen35/lezgi/grammar/original results/legacy/qwen35/lezgi/grammar/original

RESULT_JSONL="results/legacy/qwen35/lezgi/grammar/original/results_lezgi_Qwen35_ModelGloss_grammar_summary_text.jsonl"
METRICS_JSON="metrics/legacy/qwen35/lezgi/grammar/original/metrics_lezgi_Qwen35_ModelGloss_grammar_summary_text.json"

python "runners/qwen35/run_grammamt_Qwen35_context.py" \
  --language Lezgi \
  --model_id Qwen/Qwen3.5-9B \
  --support_n 21 \
  --test_n 87 \
  --grammar_text_file "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/materials/legacy/lezgi/grammar/Summary Text/lezgi_grammar_summary_model_readable.txt" \
  --model_gloss \
  --use_float32 \
  --out_metrics "${METRICS_JSON}" \
  --out_jsonl "${RESULT_JSONL}"

deactivate || true
source venv/bin/activate
export LD_LIBRARY_PATH="/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/venv/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
echo "[INFO] Rescoring with XCOMET using base venv: $(which python)"
python "runners/qwen35/score_qwen35_xcomet_from_jsonl.py" --result_jsonl "${RESULT_JSONL}" --overwrite
