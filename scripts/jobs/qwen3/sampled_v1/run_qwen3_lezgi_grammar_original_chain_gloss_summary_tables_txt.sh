#!/bin/bash
#SBATCH --job-name=sampled_v1_qwen3_lezgi_grammar_original_chain_gloss_summary_tables_txt
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --exclude=lrz-hgx-h100-026
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/slurm_outputs/%j.out
set -eo pipefail
cd "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT"
source "venv/bin/activate"
source "scripts/shared/cache_env.sh"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python runners/run_sampled_context.py --config "configs/experiments/qwen3_lezgi_grammar_original_chain_gloss_summary_tables_txt.json" --preflight
python runners/run_sampled_context.py --config "configs/experiments/qwen3_lezgi_grammar_original_chain_gloss_summary_tables_txt.json"
