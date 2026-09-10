"""One-time, hash-audited migration. Run --apply only in an idle worktree."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = {
    "Gitskan Grammar Materials/gitksan Grammar PDF 1": "materials/curated_v1/gitksan/pdf1_brown/original",
    "Gitskan Grammar Materials/gitksan Grammar PDF 2(1986 Rigsby)": "materials/curated_v1/gitksan/pdf2_rigsby/original",
    "Lezgi Grammar Materials/Lezgi Grammar PDF Cyrillic": "materials/curated_v1/lezgi/grammar/cyrillic",
    "Lezgi Grammar Materials/Lezgi Grammar PDF": "materials/curated_v1/lezgi/grammar/original",
    "Natugu Grammar Materials/Boerger Natqgu grammar sketch final": "materials/curated_v1/natugu/grammar/original",
    "Tsez Grammar Materials/Tsez_Grammar_PDF": "materials/curated_v1/tsez/grammar/original",
    "Gitskan Grammar Screenshots/PDF 1 - Brown IPA": "materials/legacy/gitksan/pdf1_brown",
    "Gitskan Grammar Screenshots/PDF 2 - Rigsby Intro": "materials/legacy/gitksan/pdf2_rigsby",
    "Lezgi Grammar Screenshots/Lezgi Grammar PDF": "materials/legacy/lezgi/grammar",
    "Natugu Grammar Screenshots": "materials/legacy/natugu/grammar",
    "Gitskan Grammar PDF's": "inputs/grammar_pdfs/gitksan",
    "Lezgi Grammar PDF": "inputs/grammar_pdfs/lezgi/grammar",
    "Natugu Grammar Sketch": "inputs/grammar_pdfs/natugu/grammar",
    "Tsez Grammar PDF": "inputs/grammar_pdfs/tsez/grammar",
    "Qwen3.5 Experiments": "runners/qwen35",
    "OpenRouter Experiments": "runners/openrouter",
    "MTOB Grammar Experiments": "experiments/mtob",
    "Run Scripts/shared": "scripts/shared",
    "Run Scripts/generators": "scripts/generators",
    "Run Scripts/legacy": "scripts/jobs/legacy",
    "Run Scripts/openrouter": "scripts/jobs/openrouter",
    "Run Scripts/qwen3": "scripts/jobs/qwen3",
    "Run Scripts/qwen35": "scripts/jobs/qwen35",
    "Local Run Scripts": "scripts/local",
    "Latest local files": "archive/local_snapshots",
    "Local Results": "archive/local_results",
    "metric_archives": "archive/metric_archives",
    "Screenshots": "docs/screenshots",
    "OCR Experiment": "docs/papers",
}
SUBDIRS = {
    "Cheat Sheet from PDF": "cheatsheet",
    "Summary Image with Tables": "summary_tables",
    "Summary Text": "summary_text",
    "Pages from the PDF/Selected impactful pages JPG": "selected_pages",
    "Pages from the PDF": "page_notes",
}


def artifact_path(old):
    name = Path(old).name
    low = name.lower()
    language = next((x for x in ("gitksan", "lezgi", "natugu", "tsez") if x in low), None)
    if language is None:
        return "archive/unclassified/" + old
    model = next((x for x in ("gemini25flashlite", "gpt56luna", "qwen35", "qwen", "apertus", "llama") if x in low), "unspecified")
    model = "qwen3" if model == "qwen" else model
    source = "pdf1_brown" if "pdf1" in low else "pdf2_rigsby" if "pdf2" in low else "grammar"
    variant = "cyrillic" if "cyrillic" in low else "original"
    family = "curated_v1" if "newmaterials" in low else "legacy" if any(x in low for x in ("grammar", "pdfpages", "summary", "cheatsheet", "pdf1", "pdf2")) else "baseline"
    if "/smoke/" in old:
        family = "smoke"
    name = name.replace("newmaterials", "curated_v1")
    return f"{old.split('/')[0]}/{family}/{model}/{language}/{source}/{variant}/{name}"


def map_path(old):
    if old.startswith(("results/", "metrics/")):
        return artifact_path(old)
    if old.startswith("Run Scripts/submit/"):
        family = "legacy" if Path(old).name.startswith("submit_old_") else "curated_v1"
        return f"scripts/submit/{family}/" + Path(old).name.replace("newmaterials", "curated_v1")
    if old.startswith("Run Scripts/") and old.endswith(".py"):
        return "scripts/generators/" + Path(old).name.replace("newmaterials", "curated_v1")
    if "/" not in old:
        if old.endswith(".out"):
            return "slurm_outputs/" + old
        if old.startswith("run_grammamt"):
            group = "qwen3" if "Qwen" in old else "baseline"
            return f"runners/{group}/{old}"
        if old.startswith("create_"):
            return "scripts/generators/" + old
        if old in ("hf_login.py", "import_model.py", "llm_call.py", "main.py", "print_dataset_stats.py"):
            return "scripts/local/" + old
    for prefix in sorted(PREFIXES, key=len, reverse=True):
        if old == prefix or old.startswith(prefix + "/"):
            new = PREFIXES[prefix] + old[len(prefix):]
            if new.startswith("materials/curated_v1/"):
                for a, b in SUBDIRS.items():
                    new = new.replace("/" + a, "/" + b)
            if new.startswith("scripts/jobs/"):
                new = new.replace("newmaterials", "curated_v1")
                new = new.replace("_2gpu", "_1gpu")
                if "/scoring/" in new:
                    new = "scripts/scoring/" + Path(new).name
            return new
    return old


def digest(path):
    if path.is_symlink():
        return hashlib.sha256(os.readlink(path).encode()).hexdigest()
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    if (ROOT / "docs/migration_manifest.tsv").exists():
        raise SystemExit("Migration already recorded; refusing a second pass")
    files = []
    for directory, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "venv", "__pycache__", "cache", "vendor")]
        files.extend(Path(directory) / n for n in names)
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    mapping = {old:map_path(old) for old in tracked if old}
    known = set(mapping) | set(mapping.values())
    for p in files:
        old = str(p.relative_to(ROOT))
        if old not in known:
            mapping[old] = map_path(old)
    destinations = {}
    for old, new in mapping.items():
        if new in destinations:
            raise ValueError(f"Destination collision: {old}, {destinations[new]} -> {new}")
        destinations[new] = old
        src, dst = ROOT / old, ROOT / new
        src_present = src.exists() or src.is_symlink()
        dst_present = dst.exists() or dst.is_symlink()
        if old != new and src_present and dst_present:
            raise ValueError(f"Refusing overwrite: {old} -> {new}")
        if not src_present and not dst_present:
            raise ValueError(f"Neither source nor destination exists: {old}")
    print(f"Inventory: {len(files)} files; moves: {sum(a != b for a,b in mapping.items())}")
    if not args.apply:
        return
    queue = subprocess.check_output(["squeue", "-h", "-u", str(__import__('getpass').getuser())], text=True)
    if queue.strip():
        raise SystemExit("Active jobs exist; migration refused")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/migration_journal.json").write_text(json.dumps(mapping, indent=2))
    rows = []
    for old, new in mapping.items():
        path = ROOT / old
        if not path.exists() and not path.is_symlink():
            path = ROOT / new
        before = digest(path)
        if path != ROOT / new:
            (ROOT / new).parent.mkdir(parents=True, exist_ok=True)
            path.rename(ROOT / new)
        rows.append([old, new, before, digest(ROOT / new)])
    # Literal paths are migrated in active code/configuration, never in historical results.
    substitutions = {a:b for a,b in mapping.items() if a != b}
    substitutions.update(PREFIXES)
    for old, new in mapping.items():
        if old.startswith(("Lezgi Grammar Materials/", "Gitskan Grammar Materials/", "Natugu Grammar Materials/", "Tsez Grammar Materials/")):
            a, b = Path(old).parent, Path(new).parent
            while len(a.parts) > 1 and len(b.parts) > 4:
                substitutions[str(a)] = str(b)
                a, b = a.parent, b.parent
    substitutions.update({"Run Scripts/submit":"scripts/submit/curated_v1", "Run Scripts":"scripts"})
    for old, new in list(mapping.items()):
        if old.endswith(".py") and "/" not in old:
            substitutions[old] = new
    pattern = re.compile("|".join(re.escape(x) for x in sorted(substitutions, key=len, reverse=True)))
    for new in mapping.values():
        p = ROOT / new
        if p.suffix not in (".py", ".sh", ".ps1", ".json", ".toml") or new.startswith(("archive/", "results/", "metrics/", "materials/", "inputs/", "docs/", "slurm_outputs/", "scripts/migrate_layout")):
            continue
        text = p.read_text(encoding="utf-8")
        text = pattern.sub(lambda m: substitutions[m[0]], text)
        if new.startswith("scripts/"):
            text = text.replace("newmaterials", "curated_v1")
            text = text.replace("_2gpu", "_1gpu")
            text = re.sub(r"(#SBATCH\s+--gres=gpu:)\d+", r"\g<1>1", text)
        # Bootstrapping moved entry points also makes sibling legacy imports work.
        if p.suffix == ".py" and new.startswith(("runners/", "scripts/generators/", "scripts/local/")):
            bootstrap = ('\nfrom pathlib import Path as _LayoutPath\nimport sys as _layout_sys\n'
                         '_LAYOUT_ROOT = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())\n'
                         'for _layout_dir in ("", "runners/baseline", "runners/qwen3", "runners/qwen35", "runners/openrouter", "scripts/generators"):\n'
                         '    _layout_sys.path.insert(0, str(_LAYOUT_ROOT / _layout_dir))\n')
            future = "from __future__ import annotations"
            if future in text:
                text = text.replace(future, future + "\n" + bootstrap, 1)
            else:
                text = bootstrap + text
            text = text.replace('PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))', 'PROJECT_ROOT = str(_LAYOUT_ROOT.parent)')
            text = text.replace('PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))', 'PROJECT_DIR = str(_LAYOUT_ROOT)')
            text = text.replace('PROJECT_DIR = Path(__file__).resolve().parents[1]', 'PROJECT_DIR = _LAYOUT_ROOT')
        p.write_text(text, encoding="utf-8")
    # Caches remain cached data; move the directories intact without editing content.
    for old, new in PREFIXES.items():
        for cache_name in ("cache", "vendor"):
            src, dst = ROOT / old / cache_name, ROOT / new / cache_name
            if src.is_dir() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                src.rename(dst)
    (ROOT / "docs").mkdir(exist_ok=True)
    with (ROOT / "docs/migration_manifest.tsv").open("w") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["old_path", "new_path", "sha256_before", "sha256_after_move"])
        writer.writerows(rows)
    for old in sorted(PREFIXES, key=len, reverse=True):
        p = ROOT / old
        try:
            p.rmdir()
        except OSError:
            pass
    print("Migration complete; verify runtime path construction before submitting jobs.")


if __name__ == "__main__":
    main()
