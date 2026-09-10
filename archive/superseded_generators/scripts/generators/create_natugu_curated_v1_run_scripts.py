from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_LAYOUT_ROOT = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _layout_dir in ("", "runners/baseline", "runners/qwen3", "runners/qwen35", "runners/openrouter", "scripts/generators"):
    _layout_sys.path.insert(0, str(_LAYOUT_ROOT / _layout_dir))


from pathlib import Path


PROJECT = Path("/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT")
RUN_DIR = PROJECT / "scripts"
QWEN3_RUN_DIR = RUN_DIR / "qwen3" / "natugu"
QWEN35_RUN_DIR = RUN_DIR / "qwen35" / "natugu"
SUBMIT_DIR = RUN_DIR / "submit"
MATERIALS = PROJECT / "Natugu Grammar Materials" / "Boerger Natqgu grammar sketch final"

LANGUAGE = "Natugu"
SUPPORT_N = 21
TEST_N = 99

TEXT_VARIANTS = [
    ("summary_text_txt", MATERIALS / "Summary Text" / "summary_model_readable.txt"),
    ("summary_tables_txt", MATERIALS / "Summary Image with Tables" / "summary_tables.txt"),
    ("cheatsheet_txt", MATERIALS / "Cheat Sheet from PDF" / "compact_cheatsheet.txt"),
]

IMAGE_VARIANTS = [
    ("summary_tables_jpg", MATERIALS / "Summary Image with Tables"),
    ("cheatsheet_jpg", MATERIALS / "Cheat Sheet from PDF"),
    ("pdfpages_impactful_jpg", MATERIALS / "Pages from the PDF" / "Selected impactful pages JPG"),
]


def qwen3_header() -> str:
    return """#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=%j.out

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
source venv/bin/activate
source "scripts/shared/cache_env.sh"
export LD_LIBRARY_PATH="/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/venv/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[INFO] Python: $(which python)"
python --version
"""


def qwen35_header() -> str:
    return """#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=%j.out

set -eo pipefail
source "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/scripts/shared/qwen35_job_env.sh"
"""


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    path.chmod(0o755)


def qwen3_script(variant: str, context_path: Path, context_kind: str, model_gloss: bool) -> str:
    label = f"ModelGloss_curated_v1_{variant}" if model_gloss else f"curated_v1_{variant}"
    runner = "run_grammamt_Qwen_grammar_images" if context_kind == "image" else "run_grammamt_Qwen_grammar_text"
    if model_gloss:
        runner += "_ModelGloss"
    arg_name = "grammar_image_dir" if context_kind == "image" else "grammar_text_file"
    return (
        qwen3_header()
        + f"""
python {runner}.py \\
  --language {LANGUAGE} \\
  --model_id Qwen/Qwen3-VL-8B-Instruct \\
  --support_n {SUPPORT_N} \\
  --test_n {TEST_N} \\
  --{arg_name} "{context_path}" \\
  --use_float32 \\
  --out_metrics metrics/qwen3/natugu/metrics_natugu_Qwen_{label}.json \\
  --out_jsonl results/results_natugu_Qwen_{label}.jsonl
"""
    )


def qwen35_script(variant: str, context_path: Path, context_kind: str, model_gloss: bool) -> str:
    label = f"ModelGloss_curated_v1_{variant}" if model_gloss else f"curated_v1_{variant}"
    arg_name = "grammar_image_dir" if context_kind == "image" else "grammar_text_file"
    model_gloss_arg = " \\\n  --model_gloss" if model_gloss else ""
    return (
        qwen35_header()
        + f"""
RESULT_JSONL="results/results_natugu_Qwen35_{label}.jsonl"
METRICS_JSON="metrics/qwen35/natugu/metrics_natugu_Qwen35_{label}.json"

python "runners/qwen35/run_grammamt_Qwen35_context.py" \\
  --language {LANGUAGE} \\
  --model_id Qwen/Qwen3.5-9B \\
  --support_n {SUPPORT_N} \\
  --test_n {TEST_N} \\
  --{arg_name} "{context_path}"{model_gloss_arg} \\
  --use_float32 \\
  --out_metrics "${{METRICS_JSON}}" \\
  --out_jsonl "${{RESULT_JSONL}}"
"""
    )


def main() -> None:
    generated: list[str] = []
    variants = [("text", name, path) for name, path in TEXT_VARIANTS]
    variants += [("image", name, path) for name, path in IMAGE_VARIANTS]

    for context_kind, variant, context_path in variants:
        for model_gloss in (False, True):
            qwen3_name = "run_natugu_qwen"
            qwen35_name = "run_natugu_qwen35"
            if model_gloss:
                qwen3_name += "_modelgloss"
                qwen35_name += "_modelgloss"
            qwen3_name += f"_curated_v1_{variant}.sh"
            qwen35_name += f"_curated_v1_{variant}.sh"

            qwen3_path = QWEN3_RUN_DIR / qwen3_name
            qwen35_path = QWEN35_RUN_DIR / qwen35_name
            write(qwen3_path, qwen3_script(variant, context_path, context_kind, model_gloss))
            write(qwen35_path, qwen35_script(variant, context_path, context_kind, model_gloss))
            generated.extend([str(qwen3_path.relative_to(PROJECT)), str(qwen35_path.relative_to(PROJECT))])

    submit = SUBMIT_DIR / "submit_all_natugu_curated_v1_qwen_jobs.sh"
    job_lines = "\n".join(f'  "{item}"' for item in generated)
    write(
        submit,
        f"""#!/bin/bash
set -eo pipefail
cd /dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT

JOBS=(
{job_lines}
)

for job in "${{JOBS[@]}}"; do
  echo "[INFO] Submitting ${{job}}"
  sbatch "${{job}}"
done
""",
    )
    generated.append(str(submit.relative_to(PROJECT)))

    print("Generated scripts:")
    for item in generated:
        print(item)


if __name__ == "__main__":
    main()
