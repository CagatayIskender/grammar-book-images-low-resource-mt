"""List by default; submit explicitly with duplicate protection and recorded IDs."""
import argparse
import getpass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runners"))
from experiment_io import CHAIN_PROMPT_REVISION, dataset, preflight_stem, read_records, sha256, validate_records
from fp32_attention import ATTENTION_TAG


def preflight_ready(cfg):
    result = ROOT / "results/preflight" / (preflight_stem(cfg, ATTENTION_TAG) + ".jsonl")
    try:
        status = json.loads(result.with_suffix(".status.json").read_text())
        provenance = json.loads(result.with_suffix(".provenance.json").read_text())
        fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if status.get("status") != "complete" or status.get("fingerprint") != fingerprint:
            return False
        if provenance.get("config") != cfg or provenance.get("attention") != ATTENTION_TAG or provenance.get("dtype") != "float32":
            return False
        if cfg["condition"] == "chain_gloss" and provenance.get("chain_prompt_revision") != CHAIN_PROMPT_REVISION:
            return False
        for key, filename in (("runner_sha256", "run_audited_context.py"), ("io_sha256", "experiment_io.py"), ("attention_sha256", "fp32_attention.py")):
            if provenance.get(key) != sha256(ROOT / "runners" / filename):
                return False
        hashes = {path:sha256(ROOT / path) for path in cfg["context_files"]}
        if provenance.get("context_hashes") != hashes:
            return False
        prompt_hashes = {str(p.relative_to(ROOT)):sha256(p) for group in ("qwen3", "qwen35", "baseline") for p in (ROOT / "runners" / group).glob("run_grammamt*.py")}
        if provenance.get("prompt_module_hashes") != prompt_hashes:
            return False
        expected = dataset(cfg["language"])
        if provenance.get("test") != expected or provenance.get("support") != dataset(cfg["language"], "train")[:21]:
            return False
        target = sorted(expected, key=lambda x:len(x["source"]), reverse=True)[:3]
        rows = read_records(result)
        validate_records(rows, target, complete=True, fingerprint=fingerprint)
        return len(rows) == status.get("records") == status.get("expected") == 3 and all(
            r[cfg["prediction_key"]].get("prediction", "").strip() and not r[cfg["prediction_key"]].get("error") for r in rows
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["curated_v1", "chain_gloss_v2"])
    ap.add_argument("--model", choices=["qwen3", "qwen35"])
    ap.add_argument("--group")
    ap.add_argument("--repairs", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    configs = json.loads((ROOT / "configs/experiment_catalog.json").read_text())
    selected = []
    for c in configs:
        if args.family and c["family"] != args.family or args.model and c["model"] != args.model:
            continue
        if args.repairs and not c["repair"]:
            continue
        if args.group:
            group = c["language"].lower()
            if group == "gitksan":
                group += "_pdf1" if c["source"]=="pdf1_brown" else "_pdf2"
            elif c["variant"] == "cyrillic":
                group += "_cyrillic"
            if group != args.group:
                continue
        if args.preflight and not (ROOT / c["preflight_job"]).is_file():
            continue
        selected.append(c)
    if not selected:
        raise SystemExit("No matching experiments")
    submit = args.submit and not args.dry_run
    queued = set(subprocess.check_output(["squeue", "-h", "-u", getpass.getuser(), "-o", "%j"], text=True).splitlines()) if submit else set()
    log = []
    for c in selected:
        path = c["preflight_job"] if args.preflight else c["job"]
        name = ("pf_" if args.preflight else "") + c["id"]
        print(path)
        if not submit:
            continue
        if name in queued:
            print(f"[SKIP] Already queued: {name}")
            continue
        if not args.preflight:
            # Require successful image stress testing for the same model/language/source.
            candidates = [x for x in configs if x["model"]==c["model"] and x["language"]==c["language"] and x["source"]==c["source"]
                          and x["condition"]=="chain_gloss" and x["material"]=="pdfpages_impactful_jpg"]
            gate_cfg = c if c["repair"] else candidates[0] if candidates else None
            if gate_cfg is None or not preflight_ready(gate_cfg):
                print(f"[BLOCKED] Current successful single-H100 FP32 preflight required: {c['id']}")
                log.append({"id":c["id"], "status":"blocked_preflight"})
                continue
        proc = subprocess.run(["sbatch", "--parsable", str(ROOT / path)], text=True, capture_output=True)
        if proc.returncode:
            print(proc.stderr, file=sys.stderr)
            log.append({"id":c["id"], "status":"submission_failed", "error":proc.stderr})
            break
        job_id = proc.stdout.strip().split(";")[0]
        log.append({"id":c["id"], "job_id":job_id, "script":path})
        queued.add(name)
        print(f"[SUBMITTED] {job_id}", flush=True)
    if submit:
        output = ROOT / "docs/submissions" / f"{time.time_ns()}.json"
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(log, indent=2))
    print(f"[SUMMARY] considered={len(selected)}, submitted={sum('job_id' in x for x in log)}")


if __name__ == "__main__":
    main()
