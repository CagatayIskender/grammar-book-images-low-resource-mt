"""Paired sacreBLEU bootstrap tests on complete, sentence-aligned predictions."""
import argparse
import collections
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runners"))
from experiment_io import atomic_json, dataset, read_records, sha256, validate_records


def baseline_key(key):
    if "model_gloss" in key or "modelgloss" in key:
        return "model_gloss"
    if "chain" in key:
        return "chain_gloss"
    if "shot" in key:
        return "gloss_shot"
    raise ValueError(f"Unmapped prediction key: {key}")


def holm(p_values):
    adjusted = [None] * len(p_values)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=lambda i:p_values[i])):
        running = max(running, (len(p_values)-rank)*p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def aligned_predictions(path, key, expected):
    rows = read_records(path)
    validate_records(rows, expected, complete=True)
    predictions, errors, empty = [], 0, 0
    for row in rows:
        block = row.get(key)
        if not isinstance(block, dict) or not isinstance(block.get("prediction"), str):
            raise ValueError(f"Missing prediction field: {key}")
        predictions.append(block["prediction"])
        errors += bool(block.get("error"))
        empty += not block["prediction"].strip()
    # Never drop failed/empty sentences from an otherwise complete file.
    return predictions, {"recorded_errors":errors, "empty_predictions":empty}


def run(args):
    import sacrebleu
    from sacrebleu.metrics import BLEU, CHRF
    from sacrebleu.significance import PairedTest
    output = (ROOT / args.output).resolve()
    if ROOT not in output.parents:
        raise ValueError("Output must be inside project root")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["SACREBLEU_SEED"] = str(args.seed)
    excluded, comparisons, inputs, groups = [], [], {}, collections.defaultdict(list)
    baselines = {}
    for language in ("Gitksan", "Lezgi", "Natugu", "Tsez"):
        expected = dataset(language)
        baseline_paths = sorted(p for model in args.models for p in (ROOT / "results/baseline" / model / language.lower()).rglob("*.jsonl"))
        for path in baseline_paths:
            if ".preflight." in path.name:
                continue
            model = path.relative_to(ROOT / "results/baseline").parts[0]
            policy = "sampled_v1" if path.stem.endswith("sampled_v1") else "greedy" if model=="qwen35" else "historical"
            rows = read_records(path)
            if not rows:
                continue
            for key, block in rows[0].items():
                if not isinstance(block, dict) or "prediction" not in block:
                    continue
                try:
                    predictions, quality = aligned_predictions(path, key, expected)
                except (ValueError, KeyError) as exc:
                    excluded.append({"path":str(path.relative_to(ROOT)), "reason":"invalid baseline: "+str(exc)})
                    continue
                ident = (model, language, baseline_key(key), policy)
                if ident in baselines:
                    raise ValueError(f"Ambiguous baseline: {ident}")
                baselines[ident] = (path, predictions, quality)
                inputs[str(path.relative_to(ROOT))] = sha256(path)
    files = sorted(p for family in ("legacy", "curated_v1", "chain_gloss_v2")
                   for p in (ROOT / "results" / family).rglob("*.jsonl"))
    for path in files:
        rel = path.relative_to(ROOT)
        parts = path.relative_to(ROOT / "results").parts
        if len(parts) != 6:
            excluded.append({"path":str(rel), "reason":"unrecognized layout"})
            continue
        family, model, lang, source, variant = parts[:5]
        if model not in args.models:
            excluded.append({"path":str(rel), "reason":"model outside requested analysis"})
            continue
        if path.stem.endswith("_md"):
            excluded.append({"path":str(rel), "reason":"deprecated Markdown counterpart; TXT retained"})
            continue
        language = lang.title()
        try:
            expected = dataset(language)
            rows = read_records(path)
            validate_records(rows, expected, complete=True)
            keys = [k for k,v in rows[0].items() if isinstance(v, dict) and "prediction" in v]
            if not keys:
                raise ValueError("No prediction blocks")
        except (ValueError, OSError, KeyError) as exc:
            excluded.append({"path":str(rel), "reason":str(exc)})
            continue
        for key in keys:
            try:
                base_key = baseline_key(key)
                policy = "sampled_v1" if "sampled_v1" in path.stem else "greedy"
                if model=="qwen35" and "chain" in key and family != "chain_gloss_v2":
                    excluded.append({"path":str(rel), "key":key, "reason":"historical Qwen3.5 chain used shot prompt; not distinct explicit-chain condition"})
                    continue
                group = (model, language, base_key, policy if model=="qwen35" else "historical")
                if group not in baselines:
                    raise ValueError(f"No complete same-model/policy baseline: {group}")
                base = baselines[group]
                predictions, quality = aligned_predictions(path, key, expected)
            except (ValueError, KeyError) as exc:
                excluded.append({"path":str(rel), "key":key, "reason":str(exc)})
                continue
            scope = "sampled_decoding_confounded" if "sampled_v1" in path.stem else "historical_same_model"
            if model=="qwen35":
                scope = "qwen35_sampled_matched_decoding" if policy=="sampled_v1" else "qwen35_historical_settings_not_fully_verified"
            record = {"experiment":str(rel)+"::"+key, "family":family, "model":model,
                      "language":language, "source":source, "variant":variant, "prediction_key":key,
                      "baseline":str(base[0].relative_to(ROOT)), "baseline_key":base_key,
                      "records":len(expected), "scope":scope, **quality,
                      "baseline_recorded_errors":base[2]["recorded_errors"],
                      "baseline_empty_predictions":base[2]["empty_predictions"]}
            groups[group].append((record, predictions))
            inputs[str(rel)] = sha256(path)
    atomic_json(output / "input_manifest.json", {"files":inputs, "excluded":excluded})
    for (model, language, key, policy), systems in groups.items():
        print(f"[TEST] {model}/{language}/{key}/{policy}: {len(systems)} systems, {args.samples} paired bootstrap resamples", flush=True)
        base = baselines[(model, language, key, policy)]
        names = [("Baseline", base[1])] + [(r["experiment"], predictions) for r,predictions in systems]
        refs = [[r["reference"] for r in dataset(language)]]
        metrics = {"BLEU":BLEU(tokenize="13a"), "chrF++":CHRF(word_order=2)}
        test = PairedTest(names, metrics, refs, test_type="bs", n_samples=args.samples, n_jobs=1)
        signatures, results = test()
        for name, values in results.items():
            if name == "System":
                continue
            for index, (record, _) in enumerate(systems, 1):
                value, baseline = values[index], values[0]
                # Corpus scores are independently recomputed through the public API.
                direct = test.metrics[name].corpus_score(names[index][1], refs).score
                if abs(direct-value.score) > 1e-8:
                    raise ValueError("Paired-test score mismatch")
                comparisons.append(dict(record, metric=name, baseline_score=float(baseline.score),
                                        score=float(value.score), delta=float(value.score-baseline.score),
                                        p_value=float(value.p_value), bootstrap_mean=float(value.mean),
                                        bootstrap_ci_half_width=float(value.ci),
                                        baseline_bootstrap_mean=float(baseline.mean),
                                        baseline_bootstrap_ci_half_width=float(baseline.ci),
                                        signature=str(signatures[name])))
        atomic_json(output / "partial_results.json", comparisons)
    for scope in sorted({r["scope"] for r in comparisons}):
        subset = [r for r in comparisons if r["scope"]==scope]
        for row, adjusted in zip(subset, holm([r["p_value"] for r in subset])):
            row["p_holm"] = adjusted
            row["holm_family_size"] = len(subset)
            row["significant_improvement"] = adjusted < 0.05 and row["delta"] > 0
            row["significant_degradation"] = adjusted < 0.05 and row["delta"] < 0
    for path, checksum in inputs.items():
        if sha256(ROOT / path) != checksum:
            raise ValueError(f"Input changed during analysis: {path}")
    metadata = {"sacrebleu_version":sacrebleu.__version__, "test":"paired bootstrap", "samples":args.samples,
                "requested_models":args.models,
                "baseline_coverage":{model:{"available_conditions":sum(k[0]==model for k in baselines),
                                             "expected_conditions":24 if model=="qwen35" else 12} for model in args.models},
                "seed":args.seed, "alpha":0.05, "holm_families":"all metrics/languages/materials within each declared scope",
                "inputs":inputs, "excluded":excluded, "comparisons":comparisons,
                "cleaning":"stored prediction strings unchanged; same single canonical reference; case sensitive",
                "limitations":["Conditional on this sentence test set and single model runs; not training-seed uncertainty.",
                               "Historical baseline prompt language/precision and context prompt designs may differ; not a randomized causal grammar ablation.",
                               "Sampled results versus greedy baselines have a decoding confound.",
                               "Empty predictions retained in full test sets; incomplete files excluded, with selection-bias risk.",
                               "CIs are sacreBLEU per-system score intervals, NOT confidence intervals of the paired delta.",
                               "No minimum practically important delta specified; significance alone does not establish usefulness.",
                               "Sentence-level bootstrap assumes independent units; document grouping is unavailable.",
                               "This is an interim snapshot; later experiments require rerunning tests and Holm adjustment."]}
    atomic_json(output / "results.json", metadata)
    if comparisons:
        with (output / "comparisons.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(comparisons)
    lines = ["# Baseline Statistical Significance", "", f"sacreBLEU {sacrebleu.__version__}; paired bootstrap, {args.samples} resamples, seed {args.seed}.",
             "BLEU: 13a, mixed case, exp smoothing. chrF++: character order 6, word order 2.",
             "Holm correction is applied across both metrics and all language/material comparisons within each declared scope.", "",
             "## Summary", "", "| Scope | Language | Metric | Tests | Significant gains | Significant losses |", "| --- | --- | --- | --- | --- | --- |"]
    for scope, language, metric in sorted({(r["scope"],r["language"],r["metric"]) for r in comparisons}):
        selected = [r for r in comparisons if (r["scope"],r["language"],r["metric"])==(scope,language,metric)]
        lines.append(f"| {scope} | {language} | {metric} | {len(selected)} | {sum(r['significant_improvement'] for r in selected)} | {sum(r['significant_degradation'] for r in selected)} |")
    lines += ["", "## Baseline Coverage", ""]
    for model, coverage in metadata["baseline_coverage"].items():
        available, expected = coverage["available_conditions"], coverage["expected_conditions"]
        lines.append(f"- {model}: {available}/{expected} full baseline conditions available." + (" PARTIAL ANALYSIS: missing controls are excluded." if available < expected else ""))
    lines += ["", "## Interpretation", ""] + ["- "+x for x in metadata["limitations"]]
    lines += ["", "Models are never paired with a different-model baseline. Missing or incomplete baselines are listed as exclusions.",
              "Complete raw p-values, adjusted p-values, deltas, quality flags, filenames and signatures are in `comparisons.tsv`.",
              "All excluded files and reasons are recorded in `input_manifest.json`.",
              "A positive significant difference is statistical evidence on these outputs, not proof of practical benefit or grammar-only causation.",
              "", "Source: https://github.com/mjpost/sacrebleu#paired-significance-tests-for-multi-system-evaluation"]
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"[DONE] {len(comparisons)} metric comparisons; {len(excluded)} exclusions; {output}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260913)
    ap.add_argument("--models", nargs="+", choices=["qwen3", "qwen35"], default=["qwen3"])
    ap.add_argument("--output", default="docs/statistical_significance/2026-09-13")
    args = ap.parse_args()
    if args.samples < 1000:
        ap.error("Use at least 1000 resamples")
    run(args)
