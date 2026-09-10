from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_LAYOUT_ROOT = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _layout_dir in ("", "runners/baseline", "runners/qwen3", "runners/qwen35", "runners/openrouter", "scripts/generators"):
    _layout_sys.path.insert(0, str(_LAYOUT_ROOT / _layout_dir))


from pathlib import Path


PROJECT = Path("/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT")
RUN_DIR = PROJECT / "scripts"
QWEN3_ROOT = RUN_DIR / "qwen3" / "gitksan"
QWEN35_ROOT = RUN_DIR / "qwen35" / "gitksan"
SUBMIT_DIR = RUN_DIR / "submit"
MATERIAL_ROOT = PROJECT / "Gitskan Grammar Materials"

LANGUAGE = "Gitksan"
SUPPORT_N = 21
TEST_N = 37

PDFS = [
    ("pdf1_brown", MATERIAL_ROOT / "gitksan Grammar PDF 1"),
    ("pdf2_rigsby", MATERIAL_ROOT / "gitksan Grammar PDF 2(1986 Rigsby)"),
]


def text_variants(base: Path) -> list[tuple[str, Path]]:
    return [
        ("summary_text_txt", base / "Summary Text" / "summary_model_readable.txt"),
        ("summary_tables_txt", base / "Summary Image with Tables" / "summary_tables.txt"),
        ("cheatsheet_txt", base / "Cheat Sheet from PDF" / "compact_cheatsheet.txt"),
    ]


def image_variants(base: Path) -> list[tuple[str, Path]]:
    return [
        ("summary_tables_jpg", base / "Summary Image with Tables"),
        ("cheatsheet_jpg", base / "Cheat Sheet from PDF"),
        ("pdfpages_impactful_jpg", base / "Pages from the PDF" / "Selected impactful pages JPG"),
    ]


def qwen3_header() -> str:
    return """#!/bin/bash
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
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
#SBATCH --time=02:00:00
#SBATCH --output=%j.out

set -eo pipefail
source "/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT/scripts/shared/qwen35_job_env.sh"
"""


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")
    path.chmod(0o755)


def result_label(pdf_label: str, variant: str, model_gloss: bool) -> str:
    if model_gloss:
        return f"ModelGloss_{pdf_label}_curated_v1_{variant}"
    return f"{pdf_label}_curated_v1_{variant}"


def qwen3_script(pdf_label: str, variant: str, context_path: Path, context_kind: str, model_gloss: bool) -> str:
    label = result_label(pdf_label, variant, model_gloss)
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
  --out_metrics metrics/qwen3/gitksan/{pdf_label}/metrics_gitksan_Qwen_{label}.json \\
  --out_jsonl results/results_gitksan_Qwen_{label}.jsonl
"""
    )


def qwen35_script(pdf_label: str, variant: str, context_path: Path, context_kind: str, model_gloss: bool) -> str:
    label = result_label(pdf_label, variant, model_gloss)
    arg_name = "grammar_image_dir" if context_kind == "image" else "grammar_text_file"
    model_gloss_arg = " \\\n  --model_gloss" if model_gloss else ""
    return (
        qwen35_header()
        + f"""
RESULT_JSONL="results/results_gitksan_Qwen35_{label}.jsonl"
METRICS_JSON="metrics/qwen35/gitksan/{pdf_label}/metrics_gitksan_Qwen35_{label}.json"

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

    for pdf_label, base in PDFS:
        variants = [("text", name, path) for name, path in text_variants(base)]
        variants += [("image", name, path) for name, path in image_variants(base)]

        for context_kind, variant, context_path in variants:
            for model_gloss in (False, True):
                qwen3_name = f"run_gitksan_qwen_{pdf_label}"
                qwen35_name = f"run_gitksan_qwen35_{pdf_label}"
                if model_gloss:
                    qwen3_name += "_modelgloss"
                    qwen35_name += "_modelgloss"
                qwen3_name += f"_curated_v1_{variant}.sh"
                qwen35_name += f"_curated_v1_{variant}.sh"

                qwen3_path = QWEN3_ROOT / pdf_label / qwen3_name
                qwen35_path = QWEN35_ROOT / pdf_label / qwen35_name
                write(qwen3_path, qwen3_script(pdf_label, variant, context_path, context_kind, model_gloss))
                write(qwen35_path, qwen35_script(pdf_label, variant, context_path, context_kind, model_gloss))
                generated.extend([str(qwen3_path.relative_to(PROJECT)), str(qwen35_path.relative_to(PROJECT))])

    submit = SUBMIT_DIR / "submit_all_gitksan_curated_v1_qwen_jobs.sh"
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
