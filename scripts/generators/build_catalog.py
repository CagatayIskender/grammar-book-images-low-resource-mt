"""Generate the single source of truth for curated and corrected-chain jobs."""
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runners"))
from experiment_io import COUNTS, atomic_json, dataset, read_records, validate_records

SOURCES = [("gitksan", "pdf1_brown", "original"), ("gitksan", "pdf2_rigsby", "original"),
           ("lezgi", "grammar", "original"), ("natugu", "grammar", "original"),
           ("tsez", "grammar", "original"), ("lezgi", "grammar", "cyrillic")]
MATERIALS = {"cheatsheet_txt":"cheatsheet/compact_cheatsheet.txt", "cheatsheet_jpg":"cheatsheet",
             "summary_tables_txt":"summary_tables/summary_tables.txt", "summary_tables_jpg":"summary_tables",
             "summary_text_txt":"summary_text/summary_model_readable.txt", "pdfpages_impactful_jpg":"selected_pages"}


def write(path, text):
    path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def job_text(config, preflight=False):
    hours = 1 if preflight else 10 if config["language"] == "Tsez" else 4
    env = 'source "scripts/shared/qwen35_job_env.sh"' if config["model"] == "qwen35" else 'source "venv/bin/activate"\nsource "scripts/shared/cache_env.sh"'
    return f'''#!/bin/bash
#SBATCH --job-name={('pf_' if preflight else '') + config['id']}
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time={hours:02}:00:00
#SBATCH --output={ROOT}/slurm_outputs/%j.out
set -eo pipefail
cd "{ROOT}"
{env}
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python runners/run_audited_context.py --config "{config['config']}" {'--preflight' if preflight else ''}
'''


def main():
    catalog = []
    for model, model_id in (("qwen3", "Qwen/Qwen3-VL-8B-Instruct"), ("qwen35", "Qwen/Qwen3.5-9B")):
        for language, source, variant in SOURCES:
            for material, rel in MATERIALS.items():
                if variant == "cyrillic" and material == "pdfpages_impactful_jpg":
                    continue
                base = ROOT / "materials/curated_v1" / language / source / variant
                location = base / rel
                files = sorted(location.glob("*.jpg")) if material.endswith("jpg") else [location]
                if not files or any(not x.is_file() for x in files):
                    raise ValueError(f"Missing material: {location}")
                for condition in ("shot", "modelgloss", "chain_gloss"):
                    family = "chain_gloss_v2" if condition == "chain_gloss" else "curated_v1"
                    key = f"{model}_context_{condition}"
                    stem = f"{condition}_{material}"
                    tree = f"{family}/{model}/{language}/{source}/{variant}"
                    ident = f"{model}_{language}_{source}_{variant}_{stem}"
                    cfg = {"id":ident, "family":family, "model":model, "model_id":model_id,
                           "language":language.title(), "source":source, "variant":variant, "condition":condition,
                           "material":material, "context_kind":"image" if material.endswith("jpg") else "text",
                           "context_files":[str(x.relative_to(ROOT)) for x in files], "expected_records":COUNTS[language.title()],
                           "prediction_key":key, "dtype":"float32", "gpu_count":1, "prompt_version":family,
                           "results":f"results/{tree}/{stem}.jsonl", "metrics":f"metrics/{tree}/{stem}.json",
                           "config":f"configs/experiments/{ident}.json",
                           "job":f"scripts/jobs/{model}/{language}/{source}/{variant}/{family}/run_{stem}.sh"}
                    cfg["repair"] = condition != "chain_gloss" and material == "pdfpages_impactful_jpg" and (
                        (model == "qwen3" and language in ("natugu", "tsez")) or
                        (model == "qwen35" and language == "tsez" and condition == "modelgloss"))
                    cfg["preflight_job"] = f"scripts/jobs/{model}/preflight/run_{ident}.sh"
                    atomic_json(ROOT / cfg["config"], cfg)
                    write(cfg["job"], job_text(cfg))
                    if cfg["repair"] or (condition == "chain_gloss" and material == "pdfpages_impactful_jpg"):
                        write(cfg["preflight_job"], job_text(cfg, preflight=True))
                    catalog.append(cfg)
    atomic_json(ROOT / "configs/experiment_catalog.json", catalog)
    targets = [{k:c[k] for k in ("language", "results", "metrics")} for c in catalog if c["repair"] or c["condition"] == "chain_gloss"]
    atomic_json(ROOT / "configs/repair_scoring_targets.json", targets)
    pending = []
    for metric in sorted((ROOT / "metrics/curated_v1/qwen35/lezgi").rglob("*.json")):
        if "_md." in metric.name:
            continue
        value = json.loads(metric.read_text())
        blocks = [v for k,v in value.items() if k.startswith("qwen35_") and isinstance(v,dict)]
        if not blocks or all(isinstance(b.get("xcomet"), (int,float)) for b in blocks):
            continue
        result = ROOT / "results" / metric.relative_to(ROOT / "metrics").parent / metric.name.replace("metrics_", "results_", 1).replace(".json", ".jsonl")
        validate_records(read_records(result), dataset("Lezgi"), complete=True)
        pending.append({"language":"Lezgi", "results":str(result.relative_to(ROOT)), "metrics":str(metric.relative_to(ROOT))})
    atomic_json(ROOT / "configs/lezgi_missing_xcomet.json", pending)
    (ROOT / "docs").mkdir(exist_ok=True)
    with (ROOT / "docs/experiment_catalog.tsv").open("w") as f:
        columns = ["id", "family", "model", "language", "source", "variant", "condition", "material", "expected_records", "image_count", "job", "results", "metrics"]
        writer = csv.DictWriter(f, columns, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for c in catalog:
            writer.writerow(dict(c, image_count=len(c["context_files"]) if c["context_kind"]=="image" else 0))
    for family in ("curated_v1", "chain_gloss_v2"):
        for model in ("qwen3", "qwen35"):
            for group in ("gitksan_pdf1", "gitksan_pdf2", "lezgi", "natugu", "tsez", "lezgi_cyrillic"):
                write(f"scripts/submit/{family}/submit_{model}_{group}.sh",
                      f'#!/bin/bash\nset -euo pipefail\ncd "{ROOT}"\npython3 scripts/submit_jobs.py --family {family} --model {model} --group {group} --submit "$@"\n')
    for name, config in (("lezgi_missing_xcomet", "lezgi_missing_xcomet"), ("repair_xcomet", "repair_scoring_targets")):
        write(f"scripts/scoring/run_{name}.sh", f'''#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition=lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --output={ROOT}/slurm_outputs/%j.out
set -eo pipefail
cd "{ROOT}"
source venv/bin/activate
source scripts/shared/cache_env.sh
python -u runners/score_verified.py --targets configs/{config}.json
''')
    print(f"Generated {len(catalog)} conditions; chain={sum(c['condition']=='chain_gloss' for c in catalog)}, repairs={sum(c['repair'] for c in catalog)}, missing Lezgi XCOMET={len(pending)}")


if __name__ == "__main__":
    main()
