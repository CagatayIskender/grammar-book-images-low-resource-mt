#!/bin/bash
set -euo pipefail
export GRAMMAMT_COMPARISON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$GRAMMAMT_COMPARISON_ROOT"
if [[ "${1:-}" == "--dry-run" ]]; then
    printf '%s\n' '90 paired comparisons; 360 BLEU/chrF tests, both views, Holm correction.' \
        '1 CPU, 24 GB RAM, 02:00:00; no GPU, model loading or translation generation.'
    exit 0
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
../../../venv/bin/python -c 'from compare import checked_scores; checked_scores()'
queued="$(squeue -h -u "$USER" -o '%j')"
if grep -Fxq mtob_vs_grammamt <<< "$queued"; then
    printf '%s\n' '[ERROR] This comparison is already queued.' >&2
    exit 1
fi
mkdir -p slurm_outputs submissions
job="$(sbatch --parsable --output="$PWD/slurm_outputs/%j.out" run_analysis.sh)"
job="${job%%;*}"
printf 'job_id\tresources\n%s\t1 CPU; 24 GB RAM; 02:00:00; no GPU\n' "$job" > "submissions/$(date -u +%Y%m%dT%H%M%SZ).tsv"
printf '[INFO] Comparison analysis submitted: %s; time limit 02:00:00; no GPU.\n' "$job"
