from pathlib import Path


ROOT = Path("/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT")
QWEN3_DIR = ROOT / "Run Scripts" / "qwen3" / "tsez"
QWEN35_DIR = ROOT / "Run Scripts" / "qwen35" / "tsez"
SUBMIT_DIR = ROOT / "Run Scripts" / "submit"
SCORING_DIR = ROOT / "Run Scripts" / "qwen35" / "scoring"

LANGUAGE = "Tsez"
SLUG = "tsez"
SUPPORT_N = 21
TEST_N = 445
TIME = "10:00:00"
MEM = "80G"
PARTITION = "lrz-hgx-h100-94x4"

MATERIALS = ROOT / "Tsez Grammar Materials" / "Tsez_Grammar_PDF"

CONTEXTS = [
    {
        "name": "cheatsheet_jpg",
        "kind": "image",
        "path": MATERIALS / "Cheat Sheet from PDF",
    },
    {
        "name": "cheatsheet_txt",
        "kind": "text",
        "path": MATERIALS / "Cheat Sheet from PDF" / "compact_cheatsheet.txt",
    },
    {
        "name": "summary_tables_jpg",
        "kind": "image",
        "path": MATERIALS / "Summary Image with Tables",
    },
    {
        "name": "summary_tables_txt",
        "kind": "text",
        "path": MATERIALS / "Summary Image with Tables" / "summary_tables.txt",
    },
    {
        "name": "summary_text_txt",
        "kind": "text",
        "path": MATERIALS / "Summary Text" / "summary_model_readable.txt",
    },
    {
        "name": "pdfpages_impactful_jpg",
        "kind": "image",
        "path": MATERIALS / "Pages from the PDF" / "Selected impactful pages JPG",
    },
]


def qwen3_script(context: dict[str, object], model_gloss: bool) -> str:
    name = context["name"]
    kind = context["kind"]
    path = context["path"]
    model_part = "_ModelGloss" if model_gloss else ""
    result_model = "Qwen_ModelGloss" if model_gloss else "Qwen"
    runner = {
        ("text", False): "run_grammamt_Qwen_grammar_text.py",
        ("text", True): "run_grammamt_Qwen_grammar_text_ModelGloss.py",
        ("image", False): "run_grammamt_Qwen_grammar_images.py",
        ("image", True): "run_grammamt_Qwen_grammar_images_ModelGloss.py",
    }[(kind, model_gloss)]
    context_arg = "--grammar_text_file" if kind == "text" else "--grammar_image_dir"
    return f"""#!/bin/bash
#SBATCH --partition={PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --mem={MEM}
#SBATCH --time={TIME}
#SBATCH --output=slurm_outputs/%j.out

set -eo pipefail

export XDG_DATA_DIRS="${{XDG_DATA_DIRS:-}}"
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

cd {ROOT}
source venv/bin/activate
source "Run Scripts/shared/cache_env.sh"
export LD_LIBRARY_PATH="{ROOT}/venv/lib/python3.10/site-packages/torch/lib:${{LD_LIBRARY_PATH:-}}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p metrics/qwen3/{SLUG} results slurm_outputs

echo "[INFO] Python: $(which python)"
python --version

python {runner} \\
  --language {LANGUAGE} \\
  --model_id Qwen/Qwen3-VL-8B-Instruct \\
  --support_n {SUPPORT_N} \\
  --test_n {TEST_N} \\
  {context_arg} "{path}" \\
  --use_float32 \\
  --out_metrics metrics/qwen3/{SLUG}/metrics_{SLUG}_{result_model}_newmaterials_{name}.json \\
  --out_jsonl results/results_{SLUG}_{result_model}_newmaterials_{name}.jsonl
"""


def qwen35_script(context: dict[str, object], model_gloss: bool) -> str:
    name = context["name"]
    kind = context["kind"]
    path = context["path"]
    result_model = "Qwen35_ModelGloss" if model_gloss else "Qwen35"
    context_arg = "--grammar_text_file" if kind == "text" else "--grammar_image_dir"
    model_gloss_line = "  --model_gloss \\\n" if model_gloss else ""
    return f"""#!/bin/bash
#SBATCH --partition={PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --mem={MEM}
#SBATCH --time={TIME}
#SBATCH --output=slurm_outputs/%j.out

set -eo pipefail
source "{ROOT}/Run Scripts/shared/qwen35_job_env.sh"

mkdir -p metrics/qwen35/{SLUG} results slurm_outputs

RESULT_JSONL="results/results_{SLUG}_{result_model}_newmaterials_{name}.jsonl"
METRICS_JSON="metrics/qwen35/{SLUG}/metrics_{SLUG}_{result_model}_newmaterials_{name}.json"

python "Qwen3.5 Experiments/run_grammamt_Qwen35_context.py" \\
  --language {LANGUAGE} \\
  --model_id Qwen/Qwen3.5-9B \\
  --support_n {SUPPORT_N} \\
  --test_n {TEST_N} \\
  {context_arg} "{path}" \\
{model_gloss_line}  --use_float32 \\
  --out_metrics "${{METRICS_JSON}}" \\
  --out_jsonl "${{RESULT_JSONL}}"
"""


def submit_wrapper(model: str) -> str:
    display = "Qwen 3.5" if model == "qwen35" else "Qwen 3"
    return f"""#!/bin/bash
set -eo pipefail
cd {ROOT}
bash "Run Scripts/submit/submit_{model}_language.sh" {SLUG}
"""


def scoring_script() -> str:
    return f"""#!/bin/bash
#SBATCH --partition={PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --mem={MEM}
#SBATCH --time=12:00:00
#SBATCH --output=slurm_outputs/%j.out

set -eo pipefail

export XDG_DATA_DIRS="${{XDG_DATA_DIRS:-}}"
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

cd {ROOT}
source venv/bin/activate
source "Run Scripts/shared/cache_env.sh"
export LD_LIBRARY_PATH="{ROOT}/venv/lib/python3.10/site-packages/torch/lib:${{LD_LIBRARY_PATH:-}}"

mkdir -p metrics/qwen35/{SLUG} slurm_outputs

echo "[INFO] Python: $(which python)"
python --version

python "Qwen3.5 Experiments/score_qwen35_xcomet_from_jsonl.py" \\
  --results_dir results \\
  --metrics_dir metrics \\
  --pattern_prefix results_{SLUG}_Qwen35 \\
  --overwrite
"""


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def main() -> None:
    for directory in [QWEN3_DIR, QWEN35_DIR, SUBMIT_DIR, SCORING_DIR, ROOT / "metrics" / "qwen3" / SLUG, ROOT / "metrics" / "qwen35" / SLUG]:
        directory.mkdir(parents=True, exist_ok=True)

    for context in CONTEXTS:
        name = context["name"]
        write(QWEN3_DIR / f"run_{SLUG}_qwen_newmaterials_{name}.sh", qwen3_script(context, False))
        write(QWEN3_DIR / f"run_{SLUG}_qwen_modelgloss_newmaterials_{name}.sh", qwen3_script(context, True))
        write(QWEN35_DIR / f"run_{SLUG}_qwen35_newmaterials_{name}.sh", qwen35_script(context, False))
        write(QWEN35_DIR / f"run_{SLUG}_qwen35_modelgloss_newmaterials_{name}.sh", qwen35_script(context, True))

    write(SUBMIT_DIR / "submit_qwen3_tsez.sh", submit_wrapper("qwen3"))
    write(SUBMIT_DIR / "submit_qwen35_tsez.sh", submit_wrapper("qwen35"))
    write(SCORING_DIR / "run_tsez_qwen35_score_xcomet.sh", scoring_script())

    print("Created Tsez Qwen3/Qwen3.5 newmaterials run scripts, submit scripts, and Qwen3.5 scorer.")


if __name__ == "__main__":
    main()
