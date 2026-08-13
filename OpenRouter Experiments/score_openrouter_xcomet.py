from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from run_grammamt_Qwen import compute_metrics, load_comet_model_with_fallback  # noqa: E402


def metric_path_for(result_path: Path, metrics_root: Path) -> Path:
    if not result_path.name.startswith("results_") or result_path.suffix != ".jsonl":
        raise ValueError(f"Unexpected result filename: {result_path}")
    metric_name = "metrics_" + result_path.name[len("results_") : -len(".jsonl")] + ".json"
    return metrics_root / result_path.parent.name / metric_name


def prediction_keys(records: list[dict], prediction_prefix: str) -> list[str]:
    keys = [
        key
        for key, value in records[0].items()
        if key == prediction_prefix or key.startswith(f"{prediction_prefix}_")
        and isinstance(value, dict)
        and "prediction" in value
    ]
    if not keys:
        raise ValueError(f"No {prediction_prefix} prediction keys found")
    return keys


def has_complete_xcomet(metric_path: Path, keys: list[str]) -> bool:
    try:
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return all(
        isinstance(metrics.get(key, {}).get("xcomet"), (int, float))
        and math.isfinite(metrics[key]["xcomet"])
        for key in keys
    )


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_label", default="gemini25flashlite")
    parser.add_argument("--results_root", default=str(PROJECT_DIR / "results" / "openrouter"))
    parser.add_argument(
        "--metrics_root",
        default="",
    )
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    metrics_root = Path(args.metrics_root) if args.metrics_root else (
        PROJECT_DIR / "metrics" / "openrouter" / args.model_label
    )
    result_prefix = f"results_{args.model_label}_"
    groups = args.group or [path.name for path in results_root.iterdir() if path.is_dir() and path.name != "smoke"]
    candidates = []
    for group in sorted(groups):
        candidates.extend(sorted((results_root / group).glob(f"{result_prefix}*.jsonl")))
    if not candidates:
        raise FileNotFoundError(f"No completed {args.model_label} result files found")

    pending: list[tuple[Path, Path, list[dict], list[str]]] = []
    for result_path in candidates:
        records = load_records(result_path)
        if not records:
            print(f"[WARN] Skipping empty result file: {result_path}")
            continue
        keys = prediction_keys(records, args.model_label)
        metric_path = metric_path_for(result_path, metrics_root)
        if not args.overwrite and has_complete_xcomet(metric_path, keys):
            print(f"[SKIP] Complete XCOMET scores already exist: {metric_path}")
            continue
        pending.append((result_path, metric_path, records, keys))

    if not pending:
        print(f"[OK] No {args.model_label} result files require XCOMET scoring")
        return

    comet_model, comet_name = load_comet_model_with_fallback()
    if comet_model is None:
        raise RuntimeError("XCOMET could not be loaded")
    for result_path, metric_path, records, keys in pending:
        print(f"[INFO] Scoring {result_path}")
        refs = [record["reference"] for record in records]
        sources = [record["source"] for record in records]
        try:
            metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metrics = {}
        for key in keys:
            hyps = [record[key]["prediction"] for record in records]
            metrics[key] = compute_metrics(refs, hyps, sources, comet_model)
        metrics["comet_model_used"] = comet_name
        metrics["scored_from_jsonl"] = str(result_path)
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        metric_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[OK] Wrote {metric_path}")


if __name__ == "__main__":
    main()
