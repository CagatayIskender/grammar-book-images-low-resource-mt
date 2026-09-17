#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
jobs=(
  "baseline_v1/jobs/qwen3/gitksan_pdf1.sh"
  "baseline_v1/jobs/qwen3/gitksan_pdf2.sh"
  "baseline_v1/jobs/qwen3/natugu.sh"
  "baseline_v1/jobs/qwen3/lezgi.sh"
  "baseline_v1/jobs/qwen3/tsez.sh"
  "baseline_v1/jobs/qwen35/gitksan_pdf1.sh"
  "baseline_v1/jobs/qwen35/gitksan_pdf2.sh"
  "baseline_v1/jobs/qwen35/natugu.sh"
  "baseline_v1/jobs/qwen35/lezgi.sh"
  "baseline_v1/jobs/qwen35/tsez.sh"
)
if [[ "${1:-}" == "--dry-run" ]]; then
    printf '%s\n' "${jobs[@]}" 'Then: baseline_v1/jobs/analyze.sh (CPU, afterok all generation jobs)'
    exit 0
fi
export PYTHONPATH="$ROOT/vendor:$ROOT"
../../venv/bin/python -c 'from mtob_grammar.closure_audit import require_audit; require_audit()'
queued="$(squeue -h -u "$USER" -o '%j')"
if grep -q '^mtob_base_' <<< "$queued"; then
    printf '%s\n' 'Baseline campaign already queued; refusing duplicate submissions.' >&2
    exit 1
fi
mkdir -p baseline_v1/submissions slurm_outputs
receipt="baseline_v1/submissions/$(date -u +%Y%m%dT%H%M%SZ).tsv"
printf 'script\tjob_id\n' > "$receipt"
ids=()
for job in "${jobs[@]}"; do
    if ! id="$(sbatch --parsable "$job")"; then
        printf '[ERROR] Submission stopped at %s. Prior submitted IDs are in %s\n' "$job" "$receipt" >&2
        exit 1
    fi
    id="${id%%;*}"
    ids+=("$id")
    printf '%s\t%s\n' "$job" "$id" >> "$receipt"
    printf '[SUBMITTED] %s %s\n' "$id" "$job"
done
dependency="$(IFS=:; echo "${ids[*]}")"
id="$(sbatch --parsable --dependency="afterok:$dependency" baseline_v1/jobs/analyze.sh)"
printf 'baseline_v1/jobs/analyze.sh\t%s\n' "${id%%;*}" >> "$receipt"
printf '[SUBMITTED] Analysis %s; after all ten generation jobs succeed. Receipt: %s\n' "${id%%;*}" "$receipt"
