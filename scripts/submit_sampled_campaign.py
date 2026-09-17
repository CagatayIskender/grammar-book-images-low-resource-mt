"""Bounded sampled campaign: exact preflight must succeed before full generation."""
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
from experiment_io import atomic_json, read_records, sha256, validate_records
from run_sampled_context import REVISION, fingerprint_of, preflight_ready, provenance_for, result_path
from build_catalog import job_text


def campaign_configs():
    catalog = json.loads((ROOT / "configs/experiment_catalog.json").read_text())
    return [c for c in catalog if c["condition"] == "chain_gloss" or
            (c["model"], c["language"], c["condition"], c["material"]) ==
            ("qwen3", "Tsez", "shot", "pdfpages_impactful_jpg")]


def state(cfg):
    provenance = provenance_for(cfg)
    full = result_path(cfg, False)
    if full.exists():
        rows = read_records(full)
        validate_records(rows, provenance["test"], fingerprint=fingerprint_of(provenance))
        if any(r[cfg["prediction_key"]].get("error") for r in rows):
            return "blocked_full_errors"
        if len(rows) == len(provenance["test"]):
            return "complete"
    if preflight_ready(cfg, provenance):
        return "ready"
    probe = result_path(cfg, True)
    if probe.exists() or probe.with_suffix(".status.json").exists():
        return "blocked_preflight"
    return "needs_preflight"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--validate-first", action="store_true", help="Allow untested conditions with a preflight command before generation")
    ap.add_argument("--max-jobs", type=int, default=20)
    ap.add_argument("--exclude-nodes", default="")
    ap.add_argument("--retry-load-oom", action="store_true", help="Archive zero-record model-load OOM checks; requires excluded node")
    args = ap.parse_args()
    if args.max_jobs < 1:
        raise SystemExit("--max-jobs must be positive")
    configs = campaign_configs()
    if args.retry_load_oom:
        if not args.exclude_nodes:
            raise SystemExit("--retry-load-oom requires --exclude-nodes")
        if args.submit:
            active = subprocess.check_output(["squeue", "-u", getpass.getuser(), "-h", "-o", "%j"], text=True)
            if active.strip():
                raise SystemExit("Archive/retry requires an empty user queue")
            archived = []
            destination = ROOT / "archive" / f"model_load_oom_{time.time_ns()}"
            for cfg in configs:
                path = result_path(cfg, True)
                status_file = path.with_suffix(".status.json")
                if not status_file.exists():
                    continue
                record = json.loads(status_file.read_text())
                error = record.get("error", "")
                if (record.get("status") != "failed" or record.get("records") != 0
                        or "CUDA out of memory" not in error or "Process " not in error
                        or "35.04 GiB" not in error or read_records(path)):
                    continue
                destination.mkdir(parents=True, exist_ok=True)
                for original in (path, status_file, path.with_suffix(".provenance.json")):
                    if original.exists():
                        moved = destination / original.name
                        checksum = sha256(original)
                        original.rename(moved)
                        archived.append({"old":str(original.relative_to(ROOT)), "new":str(moved.relative_to(ROOT)), "sha256":checksum})
            if archived:
                atomic_json(destination / "manifest.json", {"reason":"zero-record model-load OOM with other-process memory; retry on different node", "excluded_nodes":args.exclude_nodes, "files":archived})
                print(f"[ARCHIVED] {len(archived)} artifacts: {destination}")
    states = {c["id"]:state(c) for c in configs}
    configs.sort(key=lambda c:(states[c["id"]] != "ready", c["id"]))
    queued = set(subprocess.check_output(["squeue", "-u", getpass.getuser(), "-h", "-o", "%j"], text=True).splitlines()) if args.submit else set()
    log, count = [], 0
    log_path = ROOT / "docs/submissions" / f"sampled_campaign_{time.time_ns()}.json"
    for cfg in configs:
        status = states[cfg["id"]]
        name = f"{REVISION}_{cfg['id']}"
        if status != "ready" and not (status == "needs_preflight" and args.validate_first):
            print(f"[{status.upper()}] {cfg['id']}")
            continue
        if name in queued:
            print(f"[QUEUED] {cfg['id']}")
            continue
        if count >= args.max_jobs:
            print(f"[NEXT_BATCH] {cfg['id']}")
            continue
        path = ROOT / "scripts/jobs" / cfg["model"] / "sampled_v1" / f"run_{cfg['id']}.sh"
        text = job_text(cfg).replace(f"--job-name={cfg['id']}", f"--job-name={name}")
        if args.exclude_nodes:
            if not all(ch.isalnum() or ch in "-,[]" for ch in args.exclude_nodes):
                raise ValueError("Invalid node exclusion")
            text = text.replace("#SBATCH --gres=gpu:1", "#SBATCH --gres=gpu:1\n#SBATCH --exclude=" + args.exclude_nodes)
        original = f'python runners/run_audited_context.py --config "{cfg["config"]}" '
        command = f'python runners/run_sampled_context.py --config "{cfg["config"]}"'
        # set -e stops the job if its exact preflight fails. The runner checks the gate again.
        text = text.replace(original, command + " --preflight\n" + command)
        if "run_audited_context.py" in text or text.count(command) != 2:
            raise ValueError("Unexpected job template")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        subprocess.run(["bash", "-n", str(path)], check=True)
        print(f"[{status.upper()}] {path.relative_to(ROOT)}", flush=True)
        count += 1
        if not args.submit:
            continue
        proc = subprocess.run(["sbatch", "--parsable", str(path)], capture_output=True, text=True)
        if proc.returncode:
            log.append({"id":cfg["id"], "error":proc.stderr})
            atomic_json(log_path, log)
            raise SystemExit(proc.stderr)
        job_id = proc.stdout.strip().split(";")[0]
        log.append({"id":cfg["id"], "job_id":job_id, "script":str(path.relative_to(ROOT)),
                    "initial_state":status, "script_sha256":sha256(path)})
        atomic_json(log_path, log)
        queued.add(name)
        print(f"[SUBMITTED] {job_id}", flush=True)
    print(f"[SUMMARY] matrix={len(configs)} batch={count} submitted={len(log)}")


if __name__ == "__main__":
    main()
