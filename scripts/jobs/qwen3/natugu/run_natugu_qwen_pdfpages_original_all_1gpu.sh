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
mkdir -p metrics/legacy/qwen3/natugu/grammar/original results/legacy/qwen3/natugu/grammar/original
source venv/bin/activate
source "scripts/shared/cache_env.sh"
export LD_LIBRARY_PATH="/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/venv/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[INFO] Python: $(which python)"
python --version

python runners/qwen3/run_grammamt_Qwen_grammar_images.py \
  --language Natugu \
  --model_id Qwen/Qwen3-VL-8B-Instruct \
  --support_n 21 \
  --test_n 99 \
  --grammar_image_dir "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/materials/legacy/natugu/grammar/Pages from the PDF" \
  --use_float32 \
  --out_metrics metrics/legacy/qwen3/natugu/grammar/original/metrics_natugu_Qwen_pdfpages_original_all_1gpu.json \
  --out_jsonl results/legacy/qwen3/natugu/grammar/original/results_natugu_Qwen_pdfpages_original_all_1gpu.jsonl
