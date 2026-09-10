from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_LAYOUT_ROOT = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _layout_dir in ("", "runners/baseline", "runners/qwen3", "runners/qwen35", "runners/openrouter", "scripts/generators"):
    _layout_sys.path.insert(0, str(_LAYOUT_ROOT / _layout_dir))


from pathlib import Path


PROJECT = Path("/dss/dsshome1/07/ge92kun2/grammamt/GRAMMAMT")
RUN_DIR = PROJECT / "scripts"
QWEN3_RUN_DIR = RUN_DIR / "qwen3"
QWEN35_RUN_DIR = RUN_DIR / "qwen35"
SUBMIT_DIR = RUN_DIR / "submit"


LANGUAGE_PATTERNS = {
    "lezgi": {
        "submit": "submit_lezgi_modelgloss_curated_v1_qwen_jobs.sh",
        "qwen3": "run_lezgi_qwen_modelgloss_curated_v1_*.sh",
        "qwen35": "run_lezgi_qwen35_modelgloss_curated_v1_*.sh",
    },
    "natugu": {
        "submit": "submit_natugu_modelgloss_curated_v1_qwen_jobs.sh",
        "qwen3": "run_natugu_qwen_modelgloss_curated_v1_*.sh",
        "qwen35": "run_natugu_qwen35_modelgloss_curated_v1_*.sh",
    },
    "gitksan": {
        "submit": "submit_gitksan_modelgloss_curated_v1_qwen_jobs.sh",
        "qwen3": "run_gitksan_qwen_*modelgloss_curated_v1_*.sh",
        "qwen35": "run_gitksan_qwen35_*modelgloss_curated_v1_*.sh",
    },
}


def relative_jobs(qwen3_pattern: str, qwen35_pattern: str) -> list[str]:
    qwen3_jobs = sorted(QWEN3_RUN_DIR.rglob(qwen3_pattern))
    qwen35_jobs = sorted(QWEN35_RUN_DIR.rglob(qwen35_pattern))
    jobs = [
        str(path.relative_to(PROJECT))
        for path in qwen3_jobs + qwen35_jobs
        if "_md.sh" not in path.name
    ]
    if not jobs:
        raise FileNotFoundError(f"No jobs found for patterns {qwen3_pattern!r}, {qwen35_pattern!r}")
    return jobs


def write_submit(path: Path, jobs: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    job_lines = "\n".join(f'  "{job}"' for job in jobs)
    path.write_text(
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
        encoding="utf-8",
    )
    path.chmod(0o755)


def main() -> None:
    for language, config in LANGUAGE_PATTERNS.items():
        jobs = relative_jobs(config["qwen3"], config["qwen35"])
        out_path = SUBMIT_DIR / config["submit"]
        write_submit(out_path, jobs)
        print(f"{language}: {len(jobs)} jobs -> {out_path.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
