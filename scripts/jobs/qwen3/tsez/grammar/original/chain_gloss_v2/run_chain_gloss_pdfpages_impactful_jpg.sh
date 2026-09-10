#!/bin/bash
#SBATCH --job-name=qwen3_tsez_grammar_original_chain_gloss_pdfpages_impactful_jpg
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/slurm_outputs/%j.out
set -eo pipefail
cd "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT"
source "venv/bin/activate"
source "scripts/shared/cache_env.sh"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python runners/run_audited_context.py --config "configs/experiments/qwen3_tsez_grammar_original_chain_gloss_pdfpages_impactful_jpg.json" 
