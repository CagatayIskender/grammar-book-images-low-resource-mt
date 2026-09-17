#!/usr/bin/env python3
import argparse

from mtob_grammar.config import ROOT, input_path, models, sources
from mtob_grammar.data import parse_igt
from mtob_grammar.io import read_jsonl, sha256
from mtob_grammar.closure_audit import CONDITIONS, OUT, artifact, load, require_audit, result_path, validate_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="List validated record cohorts; file existence is not completion.")
    parser.add_argument("--all-models", action="store_true", help="Include API conditions outside the Qwen closure scope")
    args = parser.parse_args()
    try:
        fingerprint = require_audit()["fingerprint"]
    except (OSError, ValueError, KeyError):
        fingerprint = None
    completion = OUT / "scoring_complete.json"
    metric_hashes = {}
    if completion.exists():
        try:
            completed = load(completion)
            if completed.get("fingerprint") == fingerprint:
                metric_hashes = completed.get("metrics_sha256", {})
        except (ValueError, KeyError, TypeError):
            pass
    print("MODEL\tSOURCE\tCONDITION\tRECORDS\tRESULT\tMETRICS")
    count = 0
    for model in models() if args.all_models else ("qwen3", "qwen35"):
        for source, config in sources().items():
            limit = 99 if model not in {"qwen3", "qwen35"} and source == "tsez" else config["test_n"]
            examples = parse_igt(input_path(config["test_file"]), limit=limit)
            for condition in CONDITIONS:
                result = result_path(model, source, condition)
                n = 0
                result_status = "missing"
                if result.exists():
                    try:
                        rows = read_jsonl(result)
                        n = len(rows)
                        result_status, _ = validate_rows(rows, examples, model, source, condition)
                    except (ValueError, KeyError, TypeError):
                        result_status = "invalid"
                legacy = ROOT / "metrics" / model / source / f"metrics_{condition}.json"
                metric_status = "historical_unverified" if legacy.exists() else "missing"
                metric = artifact("metrics", model, source, condition)
                records = artifact("records", model, source, condition)
                if metric.exists():
                    metric_status = "stale_or_error"
                    try:
                        value = load(metric)
                        if fingerprint and result_status == "complete" and value.get("state") == "complete" and records.exists():
                            if value.get("binding") == {"fingerprint": fingerprint, "records_sha256": sha256(records)}:
                                metric_status = "scored_pending_full_verification"
                                if metric_hashes.get(str(metric.relative_to(OUT))) == sha256(metric):
                                    metric_status = "verified"
                    except (ValueError, KeyError, TypeError):
                        metric_status = "invalid"
                print(f"{model}\t{source}\t{condition}\t{n}/{len(examples)}\t{result_status}\t{metric_status}")
                count += 1
    print(f"\nTotal: {count} experiments")


if __name__ == "__main__":
    main()
