from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from run_grammamt_Qwen import compute_metrics, load_comet_model_with_fallback  # noqa: E402


DEFAULT_RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
DEFAULT_METRICS_DIR = os.path.join(PROJECT_DIR, "metrics")


def iter_result_files(results_dir: str, pattern_prefix: str):
    for name in sorted(os.listdir(results_dir)):
        if name.startswith(pattern_prefix) and name.endswith(".jsonl"):
            yield os.path.join(results_dir, name)


def metric_output_path(result_path: str, metrics_dir: str) -> str:
    stem = os.path.basename(result_path)
    if not stem.startswith("results_") or not stem.endswith(".jsonl"):
        raise ValueError(f"Unexpected result filename: {stem}")
    metric_name = "metrics_" + stem[len("results_") : -len(".jsonl")] + ".json"
    metric_stem = metric_name[:-len(".json")]

    if metric_stem.startswith("metrics_gitksan_Qwen35"):
        if "pdf1_brown" in metric_stem:
            return os.path.join(metrics_dir, "qwen35", "gitksan", "pdf1_brown", metric_name)
        if "pdf2_rigsby" in metric_stem:
            return os.path.join(metrics_dir, "qwen35", "gitksan", "pdf2_rigsby", metric_name)
        return os.path.join(metrics_dir, "qwen35", "gitksan", metric_name)
    if metric_stem.startswith("metrics_lezgi_Qwen35"):
        return os.path.join(metrics_dir, "qwen35", "lezgi", metric_name)
    if metric_stem.startswith("metrics_natugu_Qwen35"):
        return os.path.join(metrics_dir, "qwen35", "natugu", metric_name)
    if metric_stem.startswith("metrics_tsez_Qwen35"):
        return os.path.join(metrics_dir, "qwen35", "tsez", metric_name)
    return os.path.join(metrics_dir, metric_name)


def has_complete_xcomet_scores(metric_path: str) -> bool:
    try:
        with open(metric_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    score_blocks = [
        value
        for key, value in metrics.items()
        if key.startswith("qwen35_") and isinstance(value, dict)
    ]
    if not score_blocks:
        return False

    return all(
        isinstance(block.get("xcomet"), (int, float))
        and math.isfinite(block["xcomet"])
        for block in score_blocks
    )


def has_records(result_path: str) -> bool:
    with open(result_path, "r", encoding="utf-8") as f:
        return any(line.strip() for line in f)


def prediction_keys(records: list[dict]) -> list[str]:
    keys = []
    for key, value in records[0].items():
        if key.startswith("qwen35_") and isinstance(value, dict) and "prediction" in value:
            keys.append(key)
    if not keys:
        raise ValueError("No qwen35 prediction keys found in JSONL")
    return keys


def score_file(
    result_path: str,
    metrics_dir: str,
    comet_model,
    comet_name: str | None,
    overwrite: bool,
) -> str | None:
    out_path = metric_output_path(result_path, metrics_dir)
    if not overwrite and has_complete_xcomet_scores(out_path):
        print(f"[SKIP] Complete XCOMET scores already exist: {out_path}")
        return out_path

    with open(result_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        print(f"[WARN] Skipping empty result file: {result_path}")
        return None

    refs = [record["reference"] for record in records]
    sources = [record["source"] for record in records]

    start = time.time()
    metrics = {}
    for key in prediction_keys(records):
        hyps = [record[key]["prediction"] for record in records]
        metrics[key] = compute_metrics(refs, hyps, sources, comet_model)

    elapsed = time.time() - start
    metrics["comet_model_used"] = comet_name
    metrics["scored_from_jsonl"] = result_path
    metrics["runtime_seconds"] = elapsed
    metrics["runtime_hms"] = {
        "hours": int(elapsed // 3600),
        "minutes": int((elapsed % 3600) // 60),
        "seconds": int(elapsed % 60),
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--metrics_dir", default=DEFAULT_METRICS_DIR)
    parser.add_argument("--pattern_prefix", default="results_gitksan_Qwen35")
    parser.add_argument("--result_jsonl", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.result_jsonl:
        result_files = args.result_jsonl
    else:
        result_files = list(iter_result_files(args.results_dir, args.pattern_prefix))
    if not result_files:
        raise FileNotFoundError("No Qwen3.5 result JSONL files found")

    pending_files = []
    for result_path in result_files:
        out_path = metric_output_path(result_path, args.metrics_dir)
        if not args.overwrite and has_complete_xcomet_scores(out_path):
            print(f"[SKIP] Complete XCOMET scores already exist: {out_path}")
        elif not has_records(result_path):
            print(f"[WARN] Skipping empty result file: {result_path}")
        else:
            pending_files.append(result_path)

    if not pending_files:
        print("[OK] No result files require XCOMET scoring")
        return

    comet_model, comet_name = load_comet_model_with_fallback()
    for result_path in pending_files:
        score_file(result_path, args.metrics_dir, comet_model, comet_name, args.overwrite)


if __name__ == "__main__":
    main()
