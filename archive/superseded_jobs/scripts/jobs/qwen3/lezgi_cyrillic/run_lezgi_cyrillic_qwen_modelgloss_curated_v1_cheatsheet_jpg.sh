#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
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
source "scripts/shared/cache_env.sh"
export LD_LIBRARY_PATH="/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/venv/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[INFO] Python: $(which python)"
python --version

python runners/qwen3/run_grammamt_Qwen_grammar_images_ModelGloss.py \
  --language Lezgi \
  --model_id Qwen/Qwen3-VL-8B-Instruct \
  --support_n 21 \
  --test_n 87 \
  --grammar_image_dir "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/materials/curated_v1/lezgi/grammar/cyrillic/cheatsheet" \
  --use_float32 \
  --out_metrics metrics/curated_v1/qwen3/lezgi/grammar/cyrillic/metrics_lezgi_Qwen_ModelGloss_curated_v1_cyrillic_cheatsheet_jpg.json \
  --out_jsonl results/curated_v1/qwen3/lezgi/grammar/cyrillic/results_lezgi_Qwen_ModelGloss_curated_v1_cyrillic_cheatsheet_jpg.jsonl
