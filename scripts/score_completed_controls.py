"""Score full error-free sampled/baseline outputs; persist every exclusion."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runners"))
from experiment_io import atomic_json, dataset, read_records, sha256, validate_records
from score_verified import score_targets


def main():
    paths = list((ROOT / "results/baseline/qwen35").rglob("*.jsonl"))
    for family in ("chain_gloss_v2", "curated_v1"):
        paths += list((ROOT / "results" / family).rglob("*_sampled_v1.jsonl"))
    targets, skipped = [], []
    for path in sorted(paths):
        if ".preflight." in path.name:
            continue
        try:
            language = path.relative_to(ROOT / "results").parts[2].title()
            rows = read_records(path)
            validate_records(rows, dataset(language), complete=True)
            status = json.loads(path.with_suffix(".status.json").read_text())
            if status.get("status") != "complete" or status.get("results_sha256") != sha256(path):
                raise ValueError("Incomplete, error-marked or unverified output")
            keys = [k for k,v in rows[0].items() if isinstance(v,dict) and "prediction" in v]
            if not keys or any(not r[k]["prediction"].strip() or r[k].get("error") for r in rows for k in keys):
                raise ValueError("Invalid predictions; retain failure report, do not drop sentences")
            metric = ROOT / "metrics" / path.relative_to(ROOT / "results").with_suffix(".json")
            targets.append({"language":language, "results":str(path.relative_to(ROOT)), "metrics":str(metric.relative_to(ROOT))})
        except (ValueError, OSError, KeyError) as exc:
            skipped.append({"path":str(path.relative_to(ROOT)), "reason":str(exc)})
    manifest = ROOT / "docs/scoring_snapshots" / f"controls_{time.time_ns()}.json"
    atomic_json(manifest, {"targets":targets, "excluded":skipped})
    print(f"[INFO] full validated outputs={len(targets)}, excluded={len(skipped)}; {manifest}", flush=True)
    score_targets(targets)


if __name__ == "__main__":
    main()
