"""Separate XXL evaluation of frozen gold-support results; never run generation."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runners/matched_gold_v2"))
from protocol import validate_rows, verify_inputs
from experiment_io import atomic_json, read_records, sha256

MODEL = "Unbabel/XCOMET-XXL"
OUTPUT = ROOT / "metrics/matched_gold_v2_xcomet_xxl"


def output_path(cfg):
    relative = Path(cfg["metrics"]).relative_to("metrics/matched_gold_v2")
    path = (OUTPUT / relative).resolve()
    if not path.is_relative_to(OUTPUT.resolve()):
        raise ValueError("Output outside isolated XXL directory")
    return path


def provenance(cfg, result_hash, checkpoint_hash):
    return {"model": MODEL, "checkpoint_sha256": checkpoint_hash,
            "results_sha256": result_hash, "fingerprint": cfg["fingerprint"],
            "records": len(cfg["test"]), "scorer_sha256": sha256(Path(__file__)),
            "comet_version": importlib.metadata.version("unbabel-comet"),
            "batch_size": 1, "gpus": 1, "precision": "float32",
            "failure_policy": "all expected rows, including empty predictions",
            "scale": "native COMET score; multiply by 100 for percent-scale tables"}


def reusable(saved, expected):
    scores = saved.get("segments", [])
    value = saved.get("xcomet_xxl")
    return (saved.get("provenance") == expected
            and len(scores) == expected["records"]
            and [s.get("idx") for s in scores] == list(range(expected["records"]))
            and all(isinstance(s.get("score"), (int, float))
                    and math.isfinite(s["score"]) for s in scores)
            and isinstance(value, (int, float)) and math.isfinite(value)
            and abs(value - sum(s["score"] for s in scores) / len(scores)) < 1e-12)


def main(dry_run=False):
    catalog = json.loads((ROOT / "configs/matched_gold_v2/catalog.json").read_text())
    targets, problems = [], []
    for item in catalog:
        try:
            cfg = json.loads((ROOT / item["config"]).read_text())
            verify_inputs(cfg)
            path = ROOT / cfg["results"]
            before = sha256(path) if path.exists() else None
            rows = read_records(path)
            validate_rows(rows, cfg, complete=True)
            if before != sha256(path):
                raise ValueError("Predictions changed during validation")
            targets.append((cfg, sorted(rows, key=lambda r: r["idx"]), before))
        except (ValueError, OSError, KeyError) as exc:
            problems.append({"config": item["config"], "error": str(exc)})
    print(f"XXL: {len(targets)} complete; {len(problems)} incomplete/invalid", flush=True)
    if dry_run:
        return
    if problems:
        atomic_json(ROOT / "docs/matched_gold_v2/xcomet_xxl_incomplete.json", problems)
        raise RuntimeError("Full matrix is not valid and complete; no model loaded")

    import torch
    if torch.cuda.device_count() != 1 or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Exactly one H100 required; no CPU/offload fallback")
    from comet import download_model, load_from_checkpoint
    checkpoint = Path(download_model(MODEL))
    checkpoint_hash = sha256(checkpoint)
    pending = []
    for cfg, rows, result_hash in targets:
        identity = provenance(cfg, result_hash, checkpoint_hash)
        path = output_path(cfg)
        if path.exists() and reusable(json.loads(path.read_text()), identity):
            print(f"[SKIP verified XXL] {cfg['id']}", flush=True)
        else:
            pending.append((cfg, rows, result_hash, identity, path))
    if not pending:
        print("All XXL scores already verified", flush=True)
        return
    model = load_from_checkpoint(str(checkpoint)).float().eval()
    for cfg, rows, result_hash, identity, path in pending:
        start = time.time()
        data = [{"src": r["source"], "mt": r["translation"]["prediction"],
                 "ref": r["reference"]} for r in rows]
        output = model.predict(data, batch_size=1, gpus=1)
        scores = [float(s) for s in output["scores"]]
        if len(scores) != len(rows) or not all(math.isfinite(s) for s in scores):
            raise RuntimeError("Incomplete/nonfinite XXL scores")
        if sha256(ROOT / cfg["results"]) != result_hash:
            raise RuntimeError("Predictions changed during scoring; refusing write")
        artifact = {"id": cfg["id"], "provenance": identity,
                    "xcomet_xxl": sum(scores) / len(scores),
                    "segments": [{"idx": r["idx"], "score": s} for r, s in zip(rows, scores)],
                    "runtime_seconds": time.time() - start}
        if not reusable(artifact, identity):
            raise RuntimeError("XXL artifact failed its own validation")
        atomic_json(path, artifact)
        print(f"[SCORED XXL] {cfg['id']}: {artifact['xcomet_xxl']:.6f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Validate only; no downloads or inference")
    main(parser.parse_args().dry_run)
