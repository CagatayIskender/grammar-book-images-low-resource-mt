#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/slurm_outputs/%j.out

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
mkdir -p metrics/legacy/qwen35/natugu/grammar/original results/legacy/qwen35/natugu/grammar/original
source venv/bin/activate
source "scripts/shared/cache_env.sh"
export LD_LIBRARY_PATH="/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/venv/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[INFO] Python: $(which python)"
python --version

python "runners/qwen35/run_grammamt_Qwen35_context.py" \
  --language Natugu \
  --model_id Qwen/Qwen3.5-9B \
  --support_n 21 \
  --test_n 99 \
  --grammar_image_dir "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/materials/legacy/natugu/grammar/Summary Combined Image" \
  --model_gloss \
  --use_float32 \
  --out_metrics metrics/legacy/qwen35/natugu/grammar/original/metrics_natugu_Qwen35_ModelGloss_summary_combined_image.json \
  --out_jsonl results/legacy/qwen35/natugu/grammar/original/results_natugu_Qwen35_ModelGloss_summary_combined_image.jsonl
