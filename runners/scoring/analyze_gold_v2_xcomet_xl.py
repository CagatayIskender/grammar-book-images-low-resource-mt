"""CPU-only paired analysis of complete, prediction-bound XL sentence scores."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_gold_v2_xcomet_xxl import analyze, ROOT
from protocol import verify_inputs, validate_rows
from experiment_io import atomic_json, read_records, sha256
from score_verified import MODEL, valid_segment_scores
from artifact_layout import metric_path

OUT = ROOT / "docs/matched_gold_v2/analysis_xcomet_xl"


def verify_metric(metrics, cfg, result_hash):
    provenance = metrics.get("xcomet_provenance", {})
    if (metrics.get("fingerprint") != cfg["fingerprint"]
            or provenance.get("model") != MODEL
            or provenance.get("results_sha256") != result_hash
            or provenance.get("records") != len(cfg["test"])
            or len(provenance.get("checkpoint_sha256", "")) != 64
            or provenance.get("scorer_sha256") != sha256(ROOT / "runners/score_verified.py")
            or provenance.get("comet_version") != importlib.metadata.version("unbabel-comet")
            or not valid_segment_scores(metrics, ["translation"], len(cfg["test"]))):
        raise ValueError(f"Missing/stale XL segment scores: {cfg['id']}")
    return provenance["checkpoint_sha256"]


def main(samples=100000, dry_run=False):
    systems, inputs, checkpoints = {}, {}, set()
    catalog = json.loads((ROOT / "configs/matched_gold_v2/catalog.json").read_text())
    for item in catalog:
        config_path = ROOT / item["config"]
        cfg = json.loads(config_path.read_text())
        verify_inputs(cfg)
        if dry_run:
            continue
        result = ROOT / cfg["results"]
        result_hash = sha256(result)
        rows = read_records(result)
        validate_rows(rows, cfg, complete=True)
        path = metric_path(ROOT, cfg["metrics"])
        metric_hash = sha256(path)
        metrics = json.loads(path.read_text())
        checkpoints.add(verify_metric(metrics, cfg, result_hash))
        if cfg["id"] in systems:
            raise ValueError("Duplicate condition")
        systems[cfg["id"]] = (cfg, [s["score"] for s in metrics["xcomet_segments"]["translation"]])
        for p, h in ((config_path, sha256(config_path)), (result, result_hash), (path, metric_hash)):
            inputs[str(p.relative_to(ROOT))] = h
    if dry_run:
        print(f"Validated {len(catalog)} configurations; full run requires verified XL segment scores")
        return
    if len(systems) != 351 or len(checkpoints) != 1:
        raise ValueError("Expected all 351 conditions with one XL checkpoint")
    analyze(systems, inputs, samples, OUT, MODEL, "XL", "xcomet_xl",
            {str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__))})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.samples, args.dry_run)
