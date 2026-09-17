"""Prepare/list 13 sampled-policy checks; submit only with --submit."""
import argparse
import getpass
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runners"))
sys.path.insert(0, str(ROOT / "scripts/generators"))
from experiment_io import atomic_json
from run_sampled_context import REVISION, preflight_ready
from build_catalog import job_text


def selected_configs():
    catalog = json.loads((ROOT / "configs/experiment_catalog.json").read_text())
    return [c for c in catalog if
            (c["condition"] == "chain_gloss" and c["material"] == "pdfpages_impactful_jpg") or
            (c["language"] == "Tsez" and c["condition"] == "chain_gloss" and c["material"] == "summary_text_txt") or
            (c["model"], c["language"], c["condition"], c["material"]) ==
            ("qwen3", "Tsez", "shot", "pdfpages_impactful_jpg")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()
    configs = selected_configs()
    if len(configs) != 13 or len({c["id"] for c in configs}) != 13:
        raise ValueError("Unexpected validation matrix")
    queued = set(subprocess.check_output(["squeue", "-u", getpass.getuser(), "-h", "-o", "%j"], text=True).splitlines()) if args.submit else set()
    log = []
    log_path = ROOT / "docs/submissions" / f"sampled_preflights_{time.time_ns()}.json"
    for cfg in configs:
        name = f"pf_{REVISION}_{cfg['id']}"
        path = ROOT / "scripts/jobs" / cfg["model"] / "preflight" / REVISION / f"run_{cfg['id']}.sh"
        text = job_text(cfg, preflight=True).replace(
            "python runners/run_audited_context.py", "python runners/run_sampled_context.py"
        ).replace(f"--job-name=pf_{cfg['id']}", f"--job-name={name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        subprocess.run(["bash", "-n", str(path)], check=True)
        print(str(path.relative_to(ROOT)), flush=True)
        if not args.submit:
            continue
        if name in queued or preflight_ready(cfg):
            print("[SKIP] Queued or already verified", flush=True)
            continue
        p = subprocess.run(["sbatch", "--parsable", str(path)], capture_output=True, text=True)
        if p.returncode:
            log.append({"id":cfg["id"], "error":p.stderr})
            atomic_json(log_path, log)
            raise SystemExit(p.stderr)
        ident = p.stdout.strip().split(";")[0]
        log.append({"id":cfg["id"], "job_id":ident, "script":str(path.relative_to(ROOT))})
        atomic_json(log_path, log)
        queued.add(name)
        print(f"[SUBMITTED] {ident}", flush=True)
    print(f"[SUMMARY] considered={len(configs)} submitted={len(log)}", flush=True)


if __name__ == "__main__":
    main()
