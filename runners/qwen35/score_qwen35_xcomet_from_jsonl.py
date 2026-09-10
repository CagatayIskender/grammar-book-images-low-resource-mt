"""Compatibility CLI for the hash-checked, complete-dataset XCOMET scorer."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runners"))
from score_verified import score_targets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default=str(ROOT / "results"))
    p.add_argument("--metrics_dir", default=str(ROOT / "metrics"))
    p.add_argument("--pattern_prefix", default="results_")
    p.add_argument("--result_jsonl", action="append", default=[])
    p.add_argument("--overwrite", action="store_true", help="Compatibility flag; provenance controls skipping")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    files = [Path(x).resolve() for x in args.result_jsonl] or [x.resolve() for x in Path(args.results_dir).rglob(args.pattern_prefix + "*.jsonl") if "qwen35" in str(x).lower()]
    targets = []
    for result in files:
        rel = result.relative_to(ROOT / "results")
        language = next((x.title() for x in ("gitksan", "lezgi", "natugu", "tsez") if x in str(rel).lower()), None)
        if not language:
            raise ValueError(f"Unidentified language: {result}")
        metric = Path(args.metrics_dir).resolve() / rel.parent / rel.name.replace("results_", "metrics_", 1).replace(".jsonl", ".json")
        targets.append({"language":language, "results":str(result), "metrics":str(metric)})
    if not targets:
        raise FileNotFoundError("No matching Qwen3.5 results")
    score_targets(targets, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
