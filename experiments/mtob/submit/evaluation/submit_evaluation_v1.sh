#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ "${1:-}" == "--dry-run" ]]; then
    printf '%s\n' 'Score: 1 CPU, 24 GB RAM, 02:00:00, no GPU' \
        'Analyze: 1 CPU, 24 GB RAM, 02:00:00, no GPU; afterok dependency on score' \
        'No generation, model loading, or inference API calls.'
    exit 0
fi
export PYTHONPATH="$ROOT/vendor:$ROOT"
../../venv/bin/python -c 'from mtob_grammar.closure_audit import require_audit; require_audit()'
queued="$(squeue -h -u "$USER" -o '%j')"
if grep -Eq '^mtob_eval_v1_(score|analyze)$' <<< "$queued"; then
    printf '%s\n' '[ERROR] Evaluation already queued; refusing duplicate submission.' >&2
    exit 1
fi
mkdir -p slurm_outputs evaluation_v1/submissions
receipt="evaluation_v1/submissions/$(date -u +%Y%m%dT%H%M%SZ).tsv"
score="$(sbatch --parsable jobs/evaluation/run_score.sh)"
score="${score%%;*}"
printf 'stage\tjob_id\tdependency\nscore\t%s\tnone\n' "$score" > "$receipt"
printf '[INFO] Score job: %s\n' "$score"
if ! analyze="$(sbatch --parsable --dependency="afterok:$score" jobs/evaluation/run_analyze.sh)"; then
    printf '[ERROR] Analysis submission failed. Score job %s remains queued. Receipt: %s\n' "$score" "$receipt" >&2
    exit 1
fi
analyze="${analyze%%;*}"
printf 'analyze\t%s\tafterok:%s\n' "$analyze" "$score" >> "$receipt"
printf '[INFO] Analysis job: %s (afterok:%s)\n[INFO] Receipt: %s\n' "$analyze" "$score" "$receipt"
