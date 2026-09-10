"""Mechanical final path corrections; does not modify research outputs."""
import importlib.util
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("migration", ROOT / "scripts/migrate_layout.py")
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def main():
    for group in ("scripts", "runners"):
        for directory, dirs, files in os.walk(ROOT / group):
            dirs[:] = [x for x in dirs if x not in ("__pycache__", "cache")]
            for name in files:
                p = Path(directory) / name
                if p.suffix not in (".sh", ".py", ".ps1") or p.name in ("migrate_layout.py", "finalize_layout.py", "repair_migrated_paths.py"):
                    continue
                old = p.read_text()
                text = old.replace("Summary docs/screenshots", "Summary Screenshots")
                if p.suffix == ".sh":
                    def replace_output(match):
                        value = match[0]
                        if any(value.startswith(k + "/" + family + "/") for k in ("results", "metrics") for family in ("baseline", "legacy", "curated_v1", "chain_gloss_v2", "smoke")):
                            return value
                        return migration.artifact_path(value)
                    text = re.sub(r"(?:results|metrics)/(?:[\w./-]*/)?(?:results|metrics)_[\w.-]+\.(?:jsonl|json)", replace_output, text)
                    if p.is_relative_to(ROOT / "scripts/jobs") and "python" in text:
                        # Existing legacy writers need their newly nested parents created.
                        parents = sorted({str(Path(x).parent) for x in re.findall(r"(?:results|metrics)/(?:[\w./-]*/)?(?:results|metrics)_[\w.-]+\.(?:jsonl|json)", text)})
                        if parents and ("mkdir -p " + " ".join(parents)) not in text:
                            marker = "cd /dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT"
                            if marker in text:
                                text = text.replace(marker, marker + "\nmkdir -p " + " ".join(parents), 1)
                            elif 'source "' in text:
                                lines = text.splitlines()
                                idx = next((i for i,l in enumerate(lines) if "qwen35_job_env.sh" in l), None)
                                if idx is not None:
                                    lines.insert(idx+1, "mkdir -p " + " ".join(parents))
                                    text = "\n".join(lines) + "\n"
                if p.name.startswith("create_") and "grammar_context_materials" in p.name:
                    text = text.replace('.glob("*.pdf")', '.rglob("*.pdf")')
                if text != old:
                    p.write_text(text)
    # Old directories now contain only bytecode or empty directory scaffolding.
    for name in ("Gitskan Grammar Materials", "Gitskan Grammar PDF's", "Gitskan Grammar Screenshots", "Lezgi Grammar Materials", "Lezgi Grammar Screenshots", "MTOB Grammar Experiments", "Natugu Grammar Materials", "Natugu Grammar Screenshots", "OpenRouter Experiments", "Qwen3.5 Experiments", "Run Scripts", "Tsez Grammar Materials", "__pycache__"):
        old = ROOT / name
        if not old.exists():
            continue
        remaining = [p for p in old.rglob("*") if p.is_file() or p.is_symlink()]
        if any(p.suffix not in (".pyc", ".md") for p in remaining):
            raise RuntimeError(f"Unexpected unprocessed files: {old}")
        if remaining:
            dest = ROOT / "archive/old_directory_notes" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            old.rename(dest)
        else:
            for d in sorted(old.rglob("*"), key=lambda p:len(p.parts), reverse=True):
                d.rmdir()
            old.rmdir()


if __name__ == "__main__":
    main()
