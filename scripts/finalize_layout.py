"""Finish mechanical path migration and archive superseded entry points."""
from pathlib import Path
import csv
import json
import os
import re

ROOT = Path(__file__).resolve().parents[1]


def archive(p, category):
    dest = ROOT / "archive" / category / p.relative_to(ROOT)
    if dest.exists():
        raise ValueError(f"Archive collision: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    p.rename(dest)
    return [str(p.relative_to(ROOT)), str(dest.relative_to(ROOT))]


def main():
    if (ROOT / "docs/migration_followup.tsv").exists():
        raise SystemExit("Finalization already completed; refusing to repeat one-time edits")
    moves = []
    for p in (ROOT / "scripts/jobs").rglob("*.sh"):
        if "curated_v1" in p.name:
            moves.append(archive(p, "superseded_jobs"))
    keep = set()
    for family in ("curated_v1", "chain_gloss_v2"):
        for model in ("qwen3", "qwen35"):
            for group in ("gitksan_pdf1", "gitksan_pdf2", "lezgi", "natugu", "tsez", "lezgi_cyrillic"):
                keep.add(f"submit_{model}_{group}.sh")
    for p in (ROOT / "scripts/submit/curated_v1").glob("*.sh"):
        if p.name not in keep and not any(x in p.name for x in ("gemini", "luna")):
            moves.append(archive(p, "superseded_submit"))
    for p in (ROOT / "scripts/generators").glob("*.py"):
        if "grammar_context_materials" not in p.name and p.name != "build_catalog.py":
            moves.append(archive(p, "superseded_generators"))
    # Obsolete compatibility wrappers inside the old Qwen3.5 folder.
    for p in (ROOT / "runners/qwen35/Run Scripts").glob("*"):
        if p.is_file():
            moves.append(archive(p, "superseded_jobs"))
    for p in (ROOT / "inputs/grammar_pdfs/gitksan").glob("*.pdf"):
        dest = p.parent / ("pdf1_brown" if "PDF 1" in p.name else "pdf2_rigsby") / p.name
        dest.parent.mkdir(exist_ok=True)
        p.rename(dest)
        moves.append([str(p.relative_to(ROOT)), str(dest.relative_to(ROOT))])
    for root in ("runners", "scripts", "experiments/mtob"):
        for directory, dirs, files in os.walk(ROOT / root):
            dirs[:] = [d for d in dirs if d not in ("cache", "vendor", "__pycache__")]
            for name in files:
                p = Path(directory) / name
                if p.suffix not in (".py", ".sh", ".ps1", ".json") or p.name in ("migrate_layout.py", "finalize_layout.py"):
                    continue
                text = p.read_text()
                for old, new in moves:
                    if old.startswith("inputs/"):
                        text = text.replace(old, new)
                if p.suffix == ".sh":
                    text = re.sub(r"(#SBATCH\s+--(?:gpus|gpus-per-node|gpus-per-task)[= ])\d+", r"\g<1>1", text)
                    text = re.sub(r"(#SBATCH\s+--gres=gpu:)\d+", r"\g<1>1", text)
                    text = text.replace("#SBATCH --output=slurm_outputs/", f"#SBATCH --output={ROOT}/slurm_outputs/")
                if p.suffix == ".py" and p.parent in (ROOT / "runners/qwen3", ROOT / "runners/qwen35", ROOT / "runners/baseline"):
                    if 'ap.add_argument("--src_lang", default="Gitksan")' in text:
                        text = text.replace('ap.add_argument("--src_lang", default="Gitksan")', 'ap.add_argument("--src_lang", default=None)')
                        text = text.replace("args = ap.parse_args()", "args = ap.parse_args()\n    if args.src_lang is None:\n        args.src_lang = args.language")
                    if p.parent.name in ("qwen3", "qwen35") or p.name == "run_grammamt_ModelGloss.py":
                        text = text.replace('device_map="auto"', 'device_map={"": 0}')
                        text = text.replace("torch.float32 if use_float32 else torch.float16", "torch.float32")
                        text = text.replace("torch.float32 if use_float32 else torch.bfloat16", "torch.float32")
                        text = text.replace('if use_4bit:\n', 'if use_4bit:\n' + ' ' * 12 + 'raise ValueError("Quantization disabled: single-GPU FP32 policy")\n', 1) if '        if use_4bit:\n' in text else text
                        # Legacy generation jobs no longer instantiate XCOMET alongside Qwen.
                        text = text.replace('    comet_model, comet_name = load_comet_model_with_fallback()', '    comet_model, comet_name = None, None\n    print("[INFO] XCOMET must run in the separate scoring job")')
                p.write_text(text)
    src = ROOT / "experiments/mtob/configs/sources.json"
    config = json.loads(src.read_text())
    for value in config.values():
        value["pdf"] = "../" + value["pdf"]
        value["test_file"] = "../" + value["test_file"]
    src.write_text(json.dumps(config, indent=2))
    p = ROOT / "experiments/mtob/mtob_grammar/config.py"
    p.write_text(p.read_text().replace("PROJECT_ROOT = ROOT.parent", "PROJECT_ROOT = ROOT.parents[1]"))
    p = ROOT / "experiments/mtob/mtob_grammar/backends.py"
    p.write_text(p.read_text().replace('device_map="auto"', 'device_map={"": 0}').replace("dtype=torch.bfloat16", "dtype=torch.float32"))
    with (ROOT / "docs/migration_followup.tsv").open("w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["old_path", "new_path"])
        writer.writerows(moves)
    # Only empty directories are removed, never cached content or historical artifacts.
    for directory, dirs, files in os.walk(ROOT, topdown=True):
        dirs[:] = [d for d in dirs if d not in (".git", "venv", "cache", "vendor", "__pycache__")]
        for d in list(dirs):
            try:
                (Path(directory) / d).rmdir()
                dirs.remove(d)
            except OSError:
                pass
    print(f"Archived/relocated {len(moves)} obsolete paths")


if __name__ == "__main__":
    main()
