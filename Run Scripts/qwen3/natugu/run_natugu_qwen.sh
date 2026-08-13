#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_outputs/%j.out

set -eo pipefail

export XDG_DATA_DIRS="${XDG_DATA_DIRS:-}"
source /etc/profile
if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

set -u

if type module >/dev/null 2>&1; then
  module purge || true
  module load python || true
  module list || true
fi

cd /dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT
source venv/bin/activate
source "Run Scripts/shared/cache_env.sh"
export LD_LIBRARY_PATH="/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/venv/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

echo "[INFO] Python: $(which python)"
python --version

if [ -f "$HF_HOME/token" ]; then
  echo "[INFO] Hugging Face token cache found."
else
  echo "[WARN] No Hugging Face token cache found at $HF_HOME/token"
fi

python run_grammamt_Qwen_grammar_images.py \
  --language Natugu \
  --model_id Qwen/Qwen3-VL-8B-Instruct \
  --support_n 21 \
  --test_n 99 \
  --grammar_image_dir "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/Natugu Grammar Screenshots" \
  --max_grammar_images 4 \
  --use_float32 \
  --out_metrics metrics/qwen3/natugu/metrics_natugu_Qwen_grammar_images.json \
  --out_jsonl results/results_natugu_Qwen_grammar_images.jsonl
