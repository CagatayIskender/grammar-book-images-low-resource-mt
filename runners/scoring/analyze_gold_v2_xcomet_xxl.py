"""Paired sentence bootstrap of verified XXL scores, without model loading."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.stats import bootstrap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_gold_v2_xcomet_xxl import ROOT, MODEL, output_path, provenance, reusable
from protocol import verify_inputs, validate_rows
from experiment_io import atomic_json, read_records, sha256

OUT = ROOT / "docs/matched_gold_v2/analysis_xcomet_xxl"
SEED = 20260919


def paired_test(left, right, samples, seed):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError("Aligned score vectors with at least two sentences required")
    if not np.isfinite(left).all() or not np.isfinite(right).all() or samples < 1:
        raise ValueError("Nonfinite scores or invalid resample count")
    difference = right - left
    delta = float(difference.mean())
    result = bootstrap((difference,), np.mean, vectorized=True, batch=256,
                       n_resamples=samples, method="percentile", confidence_level=0.95,
                       rng=np.random.default_rng(seed))
    # Center paired bootstrap differences at the null, then test both directions.
    null = result.bootstrap_distribution - delta
    tolerance = 1e-12
    p = 1.0 if abs(delta) <= tolerance else (
        1 + np.count_nonzero(np.abs(null) >= abs(delta) - tolerance)) / (samples + 1)
    return {"baseline_score": float(left.mean()), "score": float(right.mean()),
            "delta": delta, "p": float(p), "delta_ci_low": float(result.confidence_interval.low),
            "delta_ci_high": float(result.confidence_interval.high), "records": len(left), "seed": seed}


def holm(values):
    adjusted = [1.0] * len(values)
    maximum = 0.0
    for rank, index in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        maximum = max(maximum, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, maximum)
    return adjusted


def check_pair(left, right):
    for key in ("model", "language", "method", "policy", "support", "test", "predicted_glosses"):
        if left.get(key) != right.get(key):
            raise ValueError(f"Unmatched XXL contrast: {key}")


def load_inputs():
    inputs, systems, checkpoint_hashes = {}, {}, set()
    for item in json.loads((ROOT / "configs/matched_gold_v2/catalog.json").read_text()):
        config_path = ROOT / item["config"]
        cfg = json.loads(config_path.read_text())
        verify_inputs(cfg)
        result_path = ROOT / cfg["results"]
        result_hash = sha256(result_path)
        rows = read_records(result_path)
        validate_rows(rows, cfg, complete=True)
        score_path = output_path(cfg)
        score_hash = sha256(score_path)
        artifact = json.loads(score_path.read_text())
        checkpoint = artifact.get("provenance", {}).get("checkpoint_sha256", "")
        if len(checkpoint) != 64 or artifact.get("id") != cfg["id"]:
            raise ValueError("Invalid XXL checkpoint/condition identity")
        expected = provenance(cfg, result_hash, checkpoint)
        if not reusable(artifact, expected):
            raise ValueError(f"Stale or invalid XXL scores: {cfg['id']}")
        checkpoint_hashes.add(checkpoint)
        if cfg["id"] in systems:
            raise ValueError("Duplicate condition")
        systems[cfg["id"]] = (cfg, [s["score"] for s in artifact["segments"]])
        for path, checksum in ((config_path, sha256(config_path)), (result_path, result_hash),
                               (score_path, score_hash)):
            inputs[str(path.relative_to(ROOT))] = checksum
    if len(systems) != 351 or len(checkpoint_hashes) != 1:
        raise ValueError("Expected 351 conditions using the same XXL checkpoint")
    return systems, inputs


def analyze(systems, inputs, samples, output, model, label, metric, extra_code_hashes=None):
    baselines = {(c["model"], c["language"], c["method"]): (c, s)
                 for c, s in systems.values() if c["material"] == "baseline"}
    cases = {(c["model"], c["language"], c["source"], c["variant"], c["method"], c["material"]): (c, s)
             for c, s in systems.values()}
    comparisons, material_pairs, scores, memo = [], [], [], {}

    def contrast(left, right, cohort):
        lc, ls = left; rc, rs = right
        check_pair(lc, rc)
        n = 99 if cohort == "common99" and rc["language"] == "Tsez" else len(rs)
        key = (lc["id"], rc["id"], n)
        if key not in memo:
            seed = SEED + int(hashlib.sha256(json.dumps(key).encode()).hexdigest()[:8], 16)
            memo[key] = paired_test(ls[:n], rs[:n], samples, seed)
        return dict(memo[key], cohort=cohort, baseline_id=lc["id"], id=rc["id"],
                    model=rc["model"], language=rc["language"], method=rc["method"],
                    source=rc["source"], variant=rc["variant"], material=rc["material"])

    for cohort in ("native", "common99"):
        for cfg, values in systems.values():
            n = 99 if cohort == "common99" and cfg["language"] == "Tsez" else len(values)
            scores.append(dict(cohort=cohort, id=cfg["id"], records=n,
                               **{metric: float(np.mean(values[:n]))}))
            if cfg["material"] != "baseline":
                comparisons.append(contrast(baselines[(cfg["model"], cfg["language"], cfg["method"])],
                                            (cfg, values), cohort))
            if cfg["material"] in ("cheatsheet_txt", "summary_tables_txt"):
                case = (cfg["model"], cfg["language"], cfg["source"], cfg["variant"], cfg["method"],
                        cfg["material"].replace("_txt", "_jpg"))
                material_pairs.append(contrast((cfg, values), cases[case], cohort))
            print(f"[{label} BOOTSTRAP] {cohort} {cfg['id']}", flush=True)
    if len(comparisons) != 630 or len(material_pairs) != 216:
        raise ValueError(f"Incomplete planned {label} test families")
    for family in (comparisons, material_pairs):
        for row, p in zip(family, holm([r["p"] for r in family])):
            row["p_holm"] = p
    for path, checksum in inputs.items():
        if sha256(ROOT / path) != checksum:
            raise ValueError("Inputs changed during analysis")
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("comparisons", comparisons), ("material_pairs", material_pairs), ("scores", scores)):
        with (output / f"{name}.tsv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader(); writer.writerows(rows)
    atomic_json(output / "results.json", dict(model=model, samples=samples, seed=SEED,
        test="two-sided null-centered paired sentence bootstrap; plus-one Monte Carlo correction",
        confidence_interval="95% percentile interval for paired mean difference; unadjusted",
        baseline_holm_tests=630, material_holm_tests=216, cohorts=["native", "common99"],
        correction=f"Separate supplemental {label} families, each pooling both cohorts; not joint with other metrics",
        comparisons=comparisons, material_pairs=material_pairs, inputs=inputs,
        analysis_sha256=sha256(Path(__file__)), extra_code_hashes=extra_code_hashes or {},
        numpy_version=np.__version__, scipy_version=scipy.__version__))
    report = [f"# XCOMET-{label} Paired Analysis", "", f"{samples} bootstrap resamples per unique contrast.",
              f"All 351 conditions validated. Empty predictions remain represented by their {label} scores.",
              "630 baseline contrasts and 216 TXT/JPG contrasts across native/common99; separate Holm families.",
              f"Supplemental {label} analysis, not jointly multiplicity-adjusted with other metrics.", "",
              "Delta is context minus baseline, or JPG minus TXT. Scores use native COMET scale.",
              "95% intervals describe paired deltas, are unadjusted, and are not the Holm decision rule.",
              "The two cohorts overlap and are not independent replications. Repeated Lezgi rows are not clustered.",
              "No significance means insufficient evidence, not equivalence. No model-interaction tests are performed."]
    for name, family in (("baseline/context", comparisons), ("TXT/JPG", material_pairs)):
        report.append(f"{name}: {sum(r['p_holm'] < .05 and r['delta'] > 0 for r in family)} positive and "
                      f"{sum(r['p_holm'] < .05 and r['delta'] < 0 for r in family)} negative significant tests.")
    (output / "report.md").write_text("\n".join(report) + "\n")


def main(samples=100000, dry_run=False):
    if dry_run:
        catalog = json.loads((ROOT / "configs/matched_gold_v2/catalog.json").read_text())
        for item in catalog:
            verify_inputs(json.loads((ROOT / item["config"]).read_text()))
        print(f"Validated {len(catalog)} configurations; XXL artifacts checked only for the full run")
        return
    systems, inputs = load_inputs()
    analyze(systems, inputs, samples, OUT, MODEL, "XXL", "xcomet_xxl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.samples, args.dry_run)
