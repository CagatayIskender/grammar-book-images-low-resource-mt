#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=06:00:00
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
mkdir -p metrics/baseline/apertus/natugu/grammar/original metrics/baseline/llama/natugu/grammar/original metrics/baseline/qwen3/natugu/grammar/original results/baseline/apertus/natugu/grammar/original results/baseline/llama/natugu/grammar/original results/baseline/qwen3/natugu/grammar/original
source venv/bin/activate
source "scripts/shared/cache_env.sh"
export LD_LIBRARY_PATH="/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/venv/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"

echo "[INFO] Python: $(which python)"
python --version

if [ -f "$HF_HOME/token" ]; then
  echo "[INFO] Hugging Face token cache found."
else
  echo "[WARN] No Hugging Face token cache found at $HF_HOME/token"
fi

# 1️⃣ LLaMA
python runners/baseline/run_grammamt.py \
  --language Natugu \
  --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --support_n 21 \
  --test_n 99 \
  --use_float32 \
  --out_metrics metrics/baseline/llama/natugu/grammar/original/metrics_natugu_Llama_3.1.json \
  --out_jsonl results/baseline/llama/natugu/grammar/original/results_natugu_Llama_3.1.jsonl

# 2️⃣ Apertus
python runners/baseline/run_grammamt_Apertus.py \
  --language Natugu \
  --model_id swiss-ai/Apertus-8B-Instruct-2509 \
  --support_n 21 \
  --test_n 99 \
  --use_float32 \
  --out_metrics metrics/baseline/apertus/natugu/grammar/original/metrics_natugu_Apertus.json \
  --out_jsonl results/baseline/apertus/natugu/grammar/original/results_natugu_Apertus.jsonl

# 3️⃣ Qwen
python runners/qwen3/run_grammamt_Qwen.py \
  --language Natugu \
  --model_id Qwen/Qwen3-VL-8B-Instruct \
  --support_n 21 \
  --test_n 99 \
  --use_float32 \
  --out_metrics metrics/baseline/qwen3/natugu/grammar/original/metrics_natugu_Qwen.json \
  --out_jsonl results/baseline/qwen3/natugu/grammar/original/results_natugu_Qwen.jsonl

# 4️⃣ ModelGloss
python runners/baseline/run_grammamt_ModelGloss.py \
  --language Natugu \
  --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --support_n 21 \
  --test_n 99 \
  --use_float32 \
  --out_metrics metrics/baseline/llama/natugu/grammar/original/metrics_natugu_ModelGloss_Llama_3.1.json \
  --out_jsonl results/baseline/llama/natugu/grammar/original/results_natugu_ModelGloss_Llama_3.1.jsonl

python runners/baseline/run_grammamt_ModelGloss.py \
  --language Natugu \
  --model_id swiss-ai/Apertus-8B-Instruct-2509 \
  --support_n 21 \
  --test_n 99 \
  --use_float32 \
  --out_metrics metrics/baseline/apertus/natugu/grammar/original/metrics_natugu_ModelGloss_Apertus.json \
  --out_jsonl results/baseline/apertus/natugu/grammar/original/results_natugu_ModelGloss_Apertus.jsonl

python runners/baseline/run_grammamt_ModelGloss.py \
  --language Natugu \
  --model_id Qwen/Qwen3-VL-8B-Instruct \
  --support_n 21 \
  --test_n 99 \
  --use_float32 \
  --out_metrics metrics/baseline/qwen3/natugu/grammar/original/metrics_natugu_ModelGloss_Qwen.json \
  --out_jsonl results/baseline/qwen3/natugu/grammar/original/results_natugu_ModelGloss_Qwen.jsonl
