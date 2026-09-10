"""XCOMET-XL only. Validate full datasets and keep existing metric metadata."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
from experiment_io import ROOT, atomic_json, dataset, read_records, sha256, validate_records

MODEL = "Unbabel/XCOMET-XL"


def score_targets(targets, dry_run=False):
    pending, problems = [], []
    for target in targets:
        result, metric = ROOT / target["results"], ROOT / target["metrics"]
        try:
            rows = read_records(result)
            expected = dataset(target["language"])
            if "expected_records" in target:
                count = int(target["expected_records"])
                if not 0 < count <= len(expected):
                    raise ValueError("Invalid configured evaluation subset")
                expected = expected[:count]
            validate_records(rows, expected, complete=True)
            keys = [k for k,v in rows[0].items() if isinstance(v, dict) and "prediction" in v]
            if not keys or any(not isinstance(row.get(k, {}).get("prediction"), str) for row in rows for k in keys):
                raise ValueError("Missing prediction fields")
            metrics = json.loads(metric.read_text()) if metric.exists() else {}
            checksum = sha256(result)
            provenance = metrics.get("xcomet_provenance", {})
            if (provenance.get("results_sha256") == checksum and provenance.get("model") == MODEL
                    and provenance.get("records") == len(rows)
                    and all(isinstance(metrics.get(k, {}).get("xcomet"), (int,float))
                            and math.isfinite(metrics[k]["xcomet"]) for k in keys)):
                print(f"[SKIP] {result}")
                continue
            pending.append((target, rows, keys, metrics, checksum))
            print(f"[PENDING] {result}: {len(rows)} records, {len(keys)} conditions")
        except (ValueError, OSError, KeyError) as exc:
            problems.append({"results":str(result), "error":str(exc)})
            print(f"[REJECT] {result}: {exc}")
    if dry_run:
        return len(pending), problems
    if not pending:
        if problems:
            raise RuntimeError(f"{len(problems)} incomplete/unverifiable files")
        return 0, []
    import torch
    if torch.cuda.device_count() != 1 or "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Scoring requires exactly one H100")
    from comet import download_model, load_from_checkpoint
    model = load_from_checkpoint(download_model(MODEL))
    import sacrebleu
    for target, rows, keys, metrics, checksum in pending:
        start = time.time()
        for key in keys:
            data = [{"src":r["source"], "mt":r[key]["prediction"], "ref":r["reference"]} for r in rows]
            scores = model.predict(data, batch_size=1, gpus=1)
            values = scores["scores"]
            if len(values) != len(rows) or not all(math.isfinite(float(x)) for x in values):
                raise RuntimeError("Incomplete/non-finite XCOMET scores")
            block = metrics.setdefault(key, {})
            refs, hyps = [x["ref"] for x in data], [x["mt"] for x in data]
            block.setdefault("bleu", sacrebleu.corpus_bleu(hyps, [refs]).score)
            block.setdefault("chrf", sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score)
            block["xcomet"] = sum(float(x) for x in values) / len(values)
        if sha256(ROOT / target["results"]) != checksum:
            raise RuntimeError("Predictions changed during scoring; refusing metrics write")
        metrics["xcomet_provenance"] = {"model":MODEL, "results_sha256":checksum, "records":len(rows), "runtime_seconds":time.time()-start}
        metrics["comet_model_used"] = MODEL
        atomic_json(ROOT / target["metrics"], metrics)
    if problems:
        raise RuntimeError(f"Scored valid files, but {len(problems)} files were rejected")
    return len(pending), problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    targets = json.loads((ROOT / args.targets).read_text())
    count, problems = score_targets(targets, args.dry_run)
    print(f"[SUMMARY] pending/scored={count}, rejected={len(problems)}")
    if args.dry_run and problems:
        sys.exit(1)
