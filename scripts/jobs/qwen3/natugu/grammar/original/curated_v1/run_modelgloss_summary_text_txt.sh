#!/bin/bash
#SBATCH --job-name=qwen3_natugu_grammar_original_modelgloss_summary_text_txt
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/slurm_outputs/%j.out
set -eo pipefail
cd "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT"
source "venv/bin/activate"
source "scripts/shared/cache_env.sh"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python runners/run_audited_context.py --config "configs/experiments/qwen3_natugu_grammar_original_modelgloss_summary_text_txt.json" 
