#!/usr/bin/env python3
"""Generate isolated baseline jobs from the existing working Qwen job environments."""
import re
from pathlib import Path

from mtob_grammar.closure_audit import ROOT, MODELS, sources


def main():
    target = ROOT / "baseline_v1"
    jobs = []
    for model in MODELS:
        for source in sources():
            original = (ROOT / "jobs" / model / source / "run_ge.sh").read_text()
            minutes = 30 if source.startswith("gitksan") else 60
            if source == "tsez" and model == "qwen3":
                minutes = 120
            text = re.sub(r"^#SBATCH --job-name=.*$", f"#SBATCH --job-name=mtob_base_{model}_{source}", original, flags=re.M)
            text = re.sub(r"^#SBATCH --time=.*$", f"#SBATCH --time={minutes // 60:02d}:{minutes % 60:02d}:00", text, flags=re.M)
            text = text.replace(f"python run_experiment.py --model {model} --source {source} --condition ge",
                                f"export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\npython -u baseline.py generate --model {model} --source {source}")
            path = target / "jobs" / model / f"{source}.sh"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            jobs.append(str(path.relative_to(ROOT)))
    cpu = (ROOT / "jobs" / "evaluation" / "run_analyze.sh").read_text()
    cpu = cpu.replace("mtob_eval_v1_analyze", "mtob_base_analyze")
    cpu = cpu.replace("finalize_results.py --stage analyze", "baseline.py analyze")
    (target / "jobs" / "analyze.sh").write_text(cpu)
    submission = '''#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
jobs=(
'''
    submission += "".join(f'  "{j}"\n' for j in jobs)
    submission += ''')
if [[ "${1:-}" == "--dry-run" ]]; then
    printf '%s\\n' "${jobs[@]}" 'Then: baseline_v1/jobs/analyze.sh (CPU, afterok all generation jobs)'
    exit 0
fi
export PYTHONPATH="$ROOT/vendor:$ROOT"
../../venv/bin/python -c 'from mtob_grammar.closure_audit import require_audit; require_audit()'
queued="$(squeue -h -u "$USER" -o '%j')"
if grep -q '^mtob_base_' <<< "$queued"; then
    printf '%s\\n' 'Baseline campaign already queued; refusing duplicate submissions.' >&2
    exit 1
fi
mkdir -p baseline_v1/submissions slurm_outputs
receipt="baseline_v1/submissions/$(date -u +%Y%m%dT%H%M%SZ).tsv"
printf 'script\\tjob_id\\n' > "$receipt"
ids=()
for job in "${jobs[@]}"; do
    if ! id="$(sbatch --parsable "$job")"; then
        printf '[ERROR] Submission stopped at %s. Prior submitted IDs are in %s\\n' "$job" "$receipt" >&2
        exit 1
    fi
    id="${id%%;*}"
    ids+=("$id")
    printf '%s\\t%s\\n' "$job" "$id" >> "$receipt"
    printf '[SUBMITTED] %s %s\\n' "$id" "$job"
done
dependency="$(IFS=:; echo "${ids[*]}")"
id="$(sbatch --parsable --dependency="afterok:$dependency" baseline_v1/jobs/analyze.sh)"
printf 'baseline_v1/jobs/analyze.sh\\t%s\\n' "${id%%;*}" >> "$receipt"
printf '[SUBMITTED] Analysis %s; after all ten generation jobs succeed. Receipt: %s\\n' "${id%%;*}" "$receipt"
'''
    (target / "submit.sh").write_text(submission)
    print(f"Prepared {len(jobs)} single-GPU jobs and one dependent CPU analysis job")


if __name__ == "__main__":
    main()
