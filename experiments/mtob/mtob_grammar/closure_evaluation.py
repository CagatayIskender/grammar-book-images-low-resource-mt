"""CPU-only scoring and paired SacreBLEU tests; never imports generation backends."""
from __future__ import annotations

import itertools
import math
import os

from .closure_audit import (CONDITIONS, MODELS, OUT, ROOT, SETTINGS, artifact, digest,
                            input_hashes, load, matrix, require_audit, save, sources)
from .evaluation import evaluate, paper_clean
from .io import sha256


def scalar_metrics(metrics):
    return {"bleu": metrics["bleu"]["bleu"], "chrf": metrics["chrf"]["score"],
            "rouge1": metrics["rouge"]["rouge1"], "rouge2": metrics["rouge"]["rouge2"],
            "rougeL": metrics["rouge"]["rougeL"], "CharacTER": metrics["character"]["cer_score"]}


def difference(after, before):
    return {k: after[k] - before[k] for k in after}


def metric_objects():
    from sacrebleu.metrics import BLEU, CHRF
    return {"BLEU": BLEU(tokenize="13a", smooth_method="none", effective_order=False),
            "chrF": CHRF(char_order=6, word_order=0, beta=2)}


def verify_score_inputs(predictions, references, metrics):
    p, r = [paper_clean(s) for s in predictions], [paper_clean(s) for s in references]
    actual = metric_objects()
    for name, expected in (("BLEU", metrics["bleu"]["bleu"] * 100), ("chrF", metrics["chrf"]["score"])):
        value = actual[name].corpus_score(p, [r]).score
        if not math.isclose(value, expected, abs_tol=1e-9):
            raise ValueError(f"Evaluator/bootstrap mismatch: {name}: {value} != {expected}")
    return p, r


def reusable(path, binding):
    if not path.exists():
        return False
    value = load(path)
    return value.get("binding") == binding and value.get("state") == "complete"


def score():
    frozen = require_audit()
    for model, source, condition in matrix():
        path = artifact("metrics", model, source, condition)
        record_path = artifact("records", model, source, condition)
        binding = {"fingerprint": frozen["fingerprint"], "records_sha256": sha256(record_path)}
        if reusable(path, binding):
            print(f"[REUSE] {model}/{source}/{condition}", flush=True)
            continue
        print(f"[SCORE] {model}/{source}/{condition}", flush=True)
        try:
            data = load(record_path)
            if data["fingerprint"] != frozen["fingerprint"]:
                raise ValueError("Stale response views")
            rows = data["records"]
            if len(rows) != sources()[source]["test_n"]:
                raise ValueError("Scoring cohort incomplete")
            references = [r["reference"] for r in rows]
            views = {}
            for view in ("raw", "extracted"):
                predictions = [r[view] for r in rows]
                if view == "extracted" and predictions == [r["raw"] for r in rows]:
                    value = dict(views["raw"])
                else:
                    value = evaluate(predictions, references)
                verify_score_inputs(predictions, references, value)
                value["evaluation_implementation"] = SETTINGS
                value["input_sha256"] = digest({"predictions": predictions, "references": references})
                value["empty_predictions"] = sum(not p.strip() for p in predictions)
                views[view] = value
            old = load(ROOT / "metrics" / model / source / f"metrics_{condition}.json")
            valid = [r for r in rows if r["status"] == "ok"]
            old_denominator = len(valid)
            reproduced = views["raw"] if len(valid) == len(rows) else evaluate([r["raw"] for r in valid], [r["reference"] for r in valid])
            hist = scalar_metrics(old)
            rep, raw, ext = map(scalar_metrics, (reproduced, views["raw"], views["extracted"]))
            output = {"state": "complete", "binding": binding, "model": model, "source": source, "condition": condition,
                      "views": views, "historical_comparison": {
                          "historical_num_examples": old.get("num_examples", old.get("valid_examples")),
                          "reproduced_status_ok_denominator": old_denominator,
                          "full_denominator": len(rows), "historical_scores": hist,
                          "historical_reproduction_difference": difference(rep, hist),
                          "denominator_effect_raw_minus_status_ok": difference(raw, rep),
                          "extraction_effect_extracted_minus_raw": difference(ext, raw),
                          "total_effect_extracted_minus_historical": difference(ext, hist)}}
            save(path, output)
        except Exception as exc:
            save(path, {"state": "error", "binding": binding, "error": f"{type(exc).__name__}: {exc}"})
            raise
    if input_hashes() != frozen["inputs"]:
        raise ValueError("Original inputs changed during scoring")
    save(OUT / "scoring_complete.json", {"fingerprint": frozen["fingerprint"], "conditions": len(matrix()),
                                         "original_hashes_unchanged": True,
                                         "metrics_sha256": {str(artifact("metrics", *key).relative_to(OUT)): sha256(artifact("metrics", *key)) for key in matrix()}})
    write_report()


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    corrected = [0.0] * len(pvalues)
    previous = 0.0
    for rank, i in enumerate(order):
        previous = max(previous, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        corrected[i] = previous
    return corrected


def paired_test(left, right, references):
    from sacrebleu.significance import PairedTest
    os.environ["SACREBLEU_SEED"] = str(SETTINGS["bootstrap"]["seed"])
    metrics = metric_objects()
    test = PairedTest([("left", left), ("right", right)], metrics, [references],
                      test_type="bs", n_samples=SETTINGS["bootstrap"]["samples"], n_jobs=1)
    signatures, values = test()
    result = []
    for name, returned_name in (("BLEU", "BLEU"), ("chrF", "chrF2")):
        a, b = values[returned_name]
        zero_effect = math.isclose(a.score, b.score, abs_tol=1e-12, rel_tol=0)
        # SacreBLEU's strict tail can give a tiny p for identical systems.
        # Preserve that package value, but never reject on a zero observed effect.
        result.append({"metric": name, "left_score": a.score, "right_score": b.score,
                       "delta_right_minus_left": b.score - a.score, "p_raw": 1.0 if zero_effect else b.p_value,
                       "p_sacrebleu": b.p_value, "zero_effect_guard": zero_effect,
                       "right_bootstrap_mean": float(b.mean), "right_ci95_halfwidth": float(b.ci),
                       "signature": str(signatures[returned_name]), "scale": "0-100"})
    return result


def analyze():
    frozen = require_audit()
    completion = load(OUT / "scoring_complete.json")
    if completion["fingerprint"] != frozen["fingerprint"]:
        raise ValueError("Scoring completion manifest stale")
    for rel, expected in completion["metrics_sha256"].items():
        if sha256(OUT / rel) != expected:
            raise ValueError(f"Metrics changed: {rel}")
    tests, errors = [], []
    for model in MODELS:
        for source in sources():
            for left, right in itertools.combinations(CONDITIONS, 2):
                for view in ("raw", "extracted"):
                    name = f"{model}_{source}_{left}_{right}_{view}"
                    path = OUT / "analysis" / "pairs" / f"{name}.json"
                    paths = [artifact("records", model, source, c) for c in (left, right)]
                    metric_paths = [artifact("metrics", model, source, c) for c in (left, right)]
                    binding = {"fingerprint": frozen["fingerprint"], "records": [sha256(p) for p in paths],
                               "metrics": [sha256(p) for p in metric_paths]}
                    try:
                        if reusable(path, binding):
                            tests.extend(load(path)["tests"])
                            continue
                        rows_a, rows_b = [load(p)["records"] for p in paths]
                        if [(r["index"], r["source"], r["reference"]) for r in rows_a] != [(r["index"], r["source"], r["reference"]) for r in rows_b]:
                            raise ValueError("Unpaired cohorts; intersection filtering forbidden")
                        metrics_a, metrics_b = [load(p)["views"][view] for p in metric_paths]
                        p_a = [r[view] for r in rows_a]
                        p_b = [r[view] for r in rows_b]
                        refs = [r["reference"] for r in rows_a]
                        for p, met in ((p_a, metrics_a), (p_b, metrics_b)):
                            if met["input_sha256"] != digest({"predictions": p, "references": refs}):
                                raise ValueError("Bootstrap inputs differ from scored predictions")
                        clean_a, clean_refs = verify_score_inputs(p_a, refs, metrics_a)
                        clean_b, _ = verify_score_inputs(p_b, refs, metrics_b)
                        print(f"[BOOTSTRAP] {name}: 100000 resamples, BLEU and chrF", flush=True)
                        values = [{"model": model, "source": source, "left": left, "right": right,
                                   "view": view, "num_examples": len(refs), **v}
                                  for v in paired_test(clean_a, clean_b, clean_refs)]
                        if any(not math.isfinite(v["p_raw"]) or not 0 <= v["p_raw"] <= 1 for v in values):
                            raise ValueError("Invalid p-value")
                        save(path, {"state": "complete", "binding": binding, "tests": values})
                        tests.extend(values)
                    except Exception as exc:
                        error = {"pair": name, "error": f"{type(exc).__name__}: {exc}"}
                        errors.append(error)
                        save(path, {"state": "error", "binding": binding, **error})
                    save(OUT / "analysis" / "progress.json", {"completed_tests": len(tests), "planned_tests": 120, "errors": errors})
    complete = len(tests) == 120 and not errors
    if complete:
        for row, p in zip(tests, holm([t["p_raw"] for t in tests]), strict=True):
            row.update(p_holm=p, significant_holm=p < 0.05)
    save(OUT / "analysis" / "results.json", {"state": "complete" if complete else "incomplete", "fingerprint": frozen["fingerprint"],
                                             "settings": SETTINGS["bootstrap"], "tests": tests, "errors": errors})
    if input_hashes() != frozen["inputs"]:
        raise ValueError("Original inputs changed during analysis")
    write_report()
    if not complete:
        raise ValueError(f"Only {len(tests)}/120 tests completed; see explicit errors")


def write_report():
    audit = load(OUT / "audit.json")
    lines = ["# MTOB Qwen Closure Report", "", "This is a separate supplementary study, not part of the 351-condition matched_v1 matrix.",
             "No new generation, API calls, prompt changes or historical result/metric replacement.", "",
             f"Audit: {'PASS' if audit['passed'] else 'FAIL'}; {audit['total_records']} records across 30 conditions.", "",
             "## Protocol and limitations", "",
             "- Only Ge versus Gs versus Gl is tested. There is no grammar-free baseline; no causal claim about adding grammar is supported.",
             "- PDF extraction replaces the paper's LaTeX preprocessing; Gl on both Qwen models extends the original setup.",
             "- Prompts are reconstructed exactly from Appendix C templates and frozen contexts. Ge/Gs each use two 512-token chunks with 256-token overlap (final chunks may be shorter).",
             "- Embedding model weights are not loaded. Frozen retrieval passages/metadata are checked against the original chunks; embedding retrieval ranking is not recomputed.",
             "- Four Qwen3.5 records have rerun_round=1. Prior round responses are not retained in their final files. Available refusal/reasoning retry histories are counted in audit.json.",
             "- Saved thinking controls and temperature are checked; missing historical runtime parameters cannot be certified. Truncation is unknown where finish metadata was not saved.",
             "- Known refusals/empty/reasoning violations remain in both denominators as empty predictions. Ordinary explanations are not newly classified as hidden reasoning.",
             "- Raw means the saved final response with existing basic label cleanup. Extracted uses reference-blind frozen rules; ambiguous boundaries preserve raw text. Multi-sentence translations are not automatically shortened.",
             "- Both views apply paper punctuation cleanup. BLEU is unsmoothed 13a, 0-1; bootstrap BLEU is the same score times 100. Plain chrF is 0-100, not chrF++.",
             "- ROUGE is the existing local macro-F1 implementation with ASCII tokens; rougeLsum equals rougeL, not official summary-level ROUGE. CharacTER uses word shifts and hypothesis-length normalization, capped at 1; it is not standard CER.",
             "- Corpus-overlap and extraction-quality limitations are inherited; this closure audit does not prove absence of test contamination or independently redo linguistic curation.",
             "- A non-significant difference is not evidence of equivalence. Model-to-model significance and ROUGE/CharacTER significance are not tested.", "",
             "- Zero observed effects receive conservative p=1 before Holm; the unmodified SacreBLEU strict-tail p-value is also retained as p_sacrebleu.", "",
             "## Actual Gl context lengths", "", "| Source | GPT-2 tokens |", "|---|---:|"]
    lines += [f"| {s} | {v['gl_tokens']} |" for s, v in audit["sources"].items()]
    lines += ["", "## Condition matrix", "", "BLEU uses the 0-1 scale below. Metrics pending until the scoring stage finishes.", "",
              "| Model | Source | Condition | Records | State | Raw BLEU | Extracted BLEU | Raw chrF | Extracted chrF | Changed | Ambiguous |", "|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    tsv = ["model\tsource\tcondition\trecords\tstate\traw_bleu\textracted_bleu\traw_chrf\textracted_chrf\tchanged\tambiguous"]
    for row in audit["conditions"]:
        key = row["model"], row["source"], row["condition"]
        path = artifact("metrics", *key)
        values = ["pending"] * 4
        state = row["state"]
        if path.exists():
            metric = load(path)
            if metric.get("state") == "complete" and metric["binding"]["fingerprint"] == audit["fingerprint"] and metric["binding"]["records_sha256"] == sha256(artifact("records", *key)):
                a, b = metric["views"]["raw"], metric["views"]["extracted"]
                values = [f"{v:.6f}" for v in (a["bleu"]["bleu"], b["bleu"]["bleu"], a["chrf"]["score"], b["chrf"]["score"])]
                state = "verified metrics"
        cells = [*key, str(row["count"]), state, *values, str(row.get("changed", 0)), str(row.get("ambiguous", 0))]
        lines.append("| " + " | ".join(cells) + " |")
        tsv.append("\t".join(cells))
    analysis = OUT / "analysis" / "results.json"
    lines += ["", "## Statistical analysis", ""]
    if analysis.exists() and load(analysis)["fingerprint"] == audit["fingerprint"]:
        data = load(analysis)
        lines += [f"State: {data['state']}. Completed tests: {len(data['tests'])}/120."]
        if data["state"] == "complete":
            lines += [f"Holm-significant metric tests: {sum(t['significant_holm'] for t in data['tests'])}/120.",
                      "100,000 paired SacreBLEU bootstrap resamples; seed 20260917; one Holm family across both views and metrics.",
                      "See analysis/results.json for raw/adjusted p-values, score differences, signatures and uncertainty."]
        else:
            lines += ["No complete-family significance conclusions are reported. See explicit errors in analysis/results.json."]
    else:
        lines += ["Pending. No significance conclusion is available until all 120 planned tests complete."]
    lines += ["", "## Audit trail", "", "- provenance.json: original-input SHA256 values, implementation hashes, versions and frozen settings.",
              "- audit.json: identity, context, prompt, retry, status and count checks.",
              "- records/: final original responses, exact character spans, selected response fields and parser rules.",
              "- metrics/: both views and historical-score differences separated into denominator and extraction effects.",
              "- analysis/pairs/: resumable paired tests bound to exact evaluated inputs."]
    save(OUT / "CLOSURE_REPORT.md", "\n".join(lines) + "\n", text=True)
    save(OUT / "condition_matrix.tsv", "\n".join(tsv) + "\n", text=True)
