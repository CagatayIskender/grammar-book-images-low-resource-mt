#!/usr/bin/env python3
import argparse

from mtob_grammar.config import models, output_path, sources
from mtob_grammar.evaluation import evaluate
from mtob_grammar.experiment import CONDITIONS, output_paths
from mtob_grammar.io import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute paper metrics from saved raw results.")
    parser.add_argument("--model", choices=models())
    parser.add_argument("--source", choices=sources())
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all and not (args.model and args.source and args.condition):
        parser.error("choose --all or provide --model, --source, and --condition")
    matrix = (
        [(m, s, c) for m in models() for s in sources() for c in CONDITIONS]
        if args.all
        else [(args.model, args.source, args.condition)]
    )
    for model, source, condition in matrix:
        result_path, metric_path = output_paths(model, source, condition)
        if not result_path.is_file():
            print(f"[SKIP] missing {result_path.relative_to(output_path())}")
            continue
        records = read_jsonl(result_path)
        valid = [r for r in records if r.get("status") == "ok"]
        metrics = evaluate([r["prediction"] for r in valid], [r["reference"] for r in valid])
        statuses = {}
        for record in records:
            status = record.get("status", "unknown")
            statuses[status] = statuses.get(status, 0) + 1
        metrics.update({"model_key": model, "source_id": source, "condition": condition, "status_counts": statuses})
        write_json(metric_path, metrics)
        print(f"[OK] {model}/{source}/{condition}: {len(valid)} scored")


if __name__ == "__main__":
    main()
