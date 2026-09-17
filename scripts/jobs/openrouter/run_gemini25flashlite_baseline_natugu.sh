#!/bin/bash
#SBATCH --job-name=gemini_baseline_natugu
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/slurm_outputs/%j.out
set -eo pipefail
source "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/scripts/shared/openrouter_job_env.sh"
for mode in baseline modelgloss_baseline; do
  extra=()
  if [[ "$mode" == modelgloss_baseline ]]; then extra+=(--model_gloss); fi
  python -u runners/openrouter/run_grammamt_openrouter_context.py \
    --baseline --language Natugu --support_n 21 --test_n 99 \
    --model google/gemini-2.5-flash-lite --model_label gemini25flashlite \
    --reasoning_effort none --temperature 0 --seed 42 --max_tokens 512 \
    --out_jsonl "results/baseline/gemini25flashlite/natugu/grammar/original/results_gemini25flashlite_natugu_${mode}.jsonl" \
    --out_metrics "metrics/baseline/gemini25flashlite/natugu/grammar/original/metrics_gemini25flashlite_natugu_${mode}.json" \
    "${extra[@]}"
done
