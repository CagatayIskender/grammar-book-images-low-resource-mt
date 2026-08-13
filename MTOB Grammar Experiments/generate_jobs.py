#!/usr/bin/env python3
from pathlib import Path
import json

from mtob_grammar.config import ROOT, models, output_path, sources
from mtob_grammar.experiment import CONDITIONS


PROJECT = ROOT.parent
QWEN35_ENV = Path("/dss/dssfs05/lwp-dss-0003/pn39je/pn39je-dss-0004/ge92kun2/envs/grammamt-qwen35")
TIME_BY_SOURCE = {
    "gitksan_pdf1": "04:00:00",
    "gitksan_pdf2": "04:00:00",
    "natugu": "10:00:00",
    "lezgi": "10:00:00",
    "tsez": "14:00:00",
}
SHORT_MODEL = {"qwen3": "q3", "qwen35": "q35", "gemini25flashlite": "gem", "gpt56luna": "luna"}
SHORT_SOURCE = {"gitksan_pdf1": "git1", "gitksan_pdf2": "git2", "natugu": "nat", "lezgi": "lez", "tsez": "tsez"}


def common_environment(model: str) -> str:
    if model == "qwen35":
        activate = f'source "{QWEN35_ENV}/bin/activate"'
    else:
        activate = f'source "{PROJECT}/venv/bin/activate"'
    openrouter = ""
    if models()[model]["backend"] == "openrouter":
        openrouter = f'''\nENV_FILE="${{OPENROUTER_ENV_FILE:-{ROOT.parents[2]}/.config/grammamt/openrouter.env}}"
if [ -f "${{ENV_FILE}}" ]; then
  set -a
  source "${{ENV_FILE}}"
  set +a
fi
if [ -z "${{OPENROUTER_API_KEY:-}}" ]; then
  echo "[ERROR] OPENROUTER_API_KEY is not set" >&2
  exit 2
fi'''
    cache = ""
    if models()[model]["backend"] == "qwen":
        cache = f'\nsource "{PROJECT}/Run Scripts/shared/cache_env.sh"'
    return f'''export XDG_DATA_DIRS="${{XDG_DATA_DIRS:-}}"
source /etc/profile
if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
if type module >/dev/null 2>&1; then
  module purge || true
  module load python || true
fi
{activate}{cache}
export PYTHONPATH="{ROOT}/vendor:{ROOT}:${{PYTHONPATH:-}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True{openrouter}
cd "{ROOT}"'''


def job_text(model: str, source: str, condition: str) -> str:
    is_qwen = models()[model]["backend"] == "qwen"
    resources = (
        "#SBATCH --partition=lrz-hgx-h100-94x4\n#SBATCH --gres=gpu:1\n#SBATCH --mem=80G"
        if is_qwen
        else "#SBATCH --partition=lrz-cpu\n#SBATCH --qos=cpu\n#SBATCH --cpus-per-task=2\n#SBATCH --mem=8G"
    )
    name = f"mtob_{SHORT_MODEL[model]}_{SHORT_SOURCE[source]}_{condition}"
    return f'''#!/bin/bash
#SBATCH --job-name={name}
{resources}
#SBATCH --time={TIME_BY_SOURCE[source]}
#SBATCH --output=slurm_outputs/%j.out

set -euo pipefail
{common_environment(model)}

python run_experiment.py --model {model} --source {source} --condition {condition}
'''


def main() -> None:
    for directory in ("jobs", "submit", "results", "metrics", "slurm_outputs"):
        output_path(directory).mkdir(parents=True, exist_ok=True)
    generated = []
    matrix = []
    for model in models():
        for source in sources():
            jobs = []
            for condition in CONDITIONS:
                path = output_path("jobs", model, source, f"run_{condition}.sh")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(job_text(model, source, condition), encoding="utf-8")
                path.chmod(0o755)
                jobs.append(path)
                generated.append(path)
                matrix.append(
                    {
                        "model_key": model,
                        "model_id": models()[model]["model_id"],
                        "source_id": source,
                        "language": sources()[source]["language"],
                        "condition": condition,
                        "direction": "language_to_english",
                        "expected_examples": sources()[source]["test_n"],
                        "job_script": str(path.relative_to(ROOT)),
                        "result_file": f"results/{model}/{source}/results_{condition}.jsonl",
                        "metrics_file": f"metrics/{model}/{source}/metrics_{condition}.json",
                    }
                )
            submit = output_path("submit", f"submit_{model}_{source}.sh")
            lines = [
                "#!/bin/bash",
                "set -euo pipefail",
                f'ROOT="{ROOT}"',
                'cd "${ROOT}"',
                "",
            ]
            lines.extend(f'sbatch "${{ROOT}}/{path.relative_to(ROOT)}"' for path in jobs)
            submit.write_text("\n".join(lines) + "\n", encoding="utf-8")
            submit.chmod(0o755)
    output_path("configs", "experiment_matrix.json").write_text(
        json.dumps(matrix, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[OK] Generated {len(generated)} jobs and {len(models()) * len(sources())} submit scripts")


if __name__ == "__main__":
    main()
