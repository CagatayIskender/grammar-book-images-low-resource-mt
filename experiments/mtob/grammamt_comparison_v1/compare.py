"""Offline cross-protocol comparison; no generation or model imports."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
PROJECT = ROOT.parent.parent
sys.path[:0] = [str(ROOT / "vendor"), str(ROOT)]

from mtob_grammar.closure_audit import (
    CONDITIONS, MODELS, SETTINGS, artifact, digest, load, matrix, models,
    require_audit, result_path, sources,
)
from mtob_grammar.closure_evaluation import (
    evaluate, holm, paired_test, scalar_metrics, verify_score_inputs,
)
from mtob_grammar.io import read_jsonl, sha256
from mtob_grammar.response_views import response_views

METHODS = ("shot", "chain_gloss", "modelgloss")
VIEWS = ("raw", "extracted")
VERSION = "cross-protocol-1.0.0"
SETTINGS = json.loads(json.dumps(SETTINGS))
SETTINGS["version"] = VERSION
SETTINGS["bootstrap"]["family_size"] = 360
SETTINGS["comparison"] = "existing matched_v1 baseline (left) versus MTOB (right)"
SETTINGS["structural_adapter"] = "Explicit FINAL_TRANSLATION section only; required gloss is never scored. No first-sentence truncation."
FINAL = re.compile(r"^\s*FINAL_TRANSLATION:\s*", re.I | re.M)


def save(path, value, text=False):
    path = path.resolve()
    if OUT not in path.parents:
        raise ValueError(f"Write outside comparison directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(value if text else json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def tsv(path, rows):
    if not rows:
        return
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)
    save(path, buffer.getvalue(), text=True)


def language(source):
    return sources()[source]["language"].lower()


def baseline_paths(model, lang, method):
    ident = f"{model}_{lang}_grammar_original_{method}_baseline"
    config = PROJECT / "configs" / "matched_v1" / f"{ident}.json"
    result = PROJECT / "results" / "matched_v1" / model / lang / "grammar" / "original" / f"{method}_baseline.jsonl"
    return config, result


def baseline_keys():
    return [(m, l, t) for m in MODELS for l in dict.fromkeys(map(language, sources())) for t in METHODS]


def system_id(family, model, source, condition):
    return f"{family}_{model}_{source}_{condition}"


def all_pairs():
    return [{"model": m, "source": s, "language": language(s), "condition": c, "method": t,
             "left": system_id("matched", m, language(s), t), "right": system_id("mtob", m, s, c)}
            for m, s, c in matrix() for t in METHODS]


def identity(rows):
    return [(r["index"], r["source"], r["reference"]) for r in rows]


def validate_baseline(rows, config, expected):
    if digest({k: v for k, v in config.items() if k != "fingerprint"}) != config["fingerprint"]:
        raise ValueError("Baseline config fingerprint is invalid")
    if [(r["source"], r["reference"]) for r in config["test"]] != [(r["source"], r["reference"]) for r in expected]:
        raise ValueError("Baseline config test set differs from audited MTOB cohort")
    seen = set()
    for row in rows:
        idx = row.get("idx")
        if type(idx) is not int or idx in seen or not 0 <= idx < len(expected):
            raise ValueError(f"Invalid or duplicate baseline index {idx!r}")
        seen.add(idx)
        if any(row.get(k) != expected[idx][k] for k in ("source", "reference")):
            raise ValueError(f"Baseline source/reference mismatch at {idx}")
        if row.get("fingerprint") != config["fingerprint"]:
            raise ValueError(f"Baseline row fingerprint mismatch at {idx}")
    if seen != set(range(len(expected))):
        raise ValueError("Incomplete baseline; intersection filtering is forbidden")


def baseline_view(row, method):
    value = row["translation"]
    raw = value["raw"]
    if not isinstance(raw, str):
        raise ValueError("Baseline has no raw response")
    status = "ok"
    error = value.get("error") or ""
    if "reasoning" in error:
        status = "reasoning_violation"
    elif "refusal" in error:
        status = "refusal"
    matches = list(FINAL.finditer(raw))
    if len(matches) > 1:
        raise ValueError("Ambiguous FINAL_TRANSLATION sections; manual protocol audit required")
    start, segment = (matches[0].end(), raw[matches[0].end():]) if matches else (0, raw)
    structural_rule = "explicit_final_translation" if matches else "unmarked_translation"
    if not matches and method == "chain_gloss":
        segment, status, structural_rule = "", "empty", "missing_required_translation_section"
    if not segment.strip() and status == "ok":
        status = "empty"
    parsed = response_views({"initial_response": segment, "status": status})
    for key in ("raw_span", "extracted_span"):
        if parsed[key] is not None:
            parsed[key] = [n + start for n in parsed[key]]
    parsed.update(original_response=raw, response_field="translation.raw", structural_rule=structural_rule,
                  status=status, recorded_error=value.get("error"), recorded_flags=value.get("flags", []),
                  recorded_prediction=value["prediction"],
                  truncation=value.get("attempt", {}).get("truncated", "unknown"),
                  origin=row.get("origin"))
    return {"index": row["idx"], "source": row["source"], "reference": row["reference"], **parsed}


def provenance():
    frozen = require_audit()
    paths = []
    for key in baseline_keys():
        paths.extend(baseline_paths(*key))
    for key in matrix():
        paths.append(artifact("records", *key))
    value = {"mtob_closure_fingerprint": frozen["fingerprint"], "settings": SETTINGS,
             "inputs": {str(p.relative_to(PROJECT)): sha256(p) for p in sorted(set(paths))},
             "implementation_sha256": sha256(Path(__file__))}
    return {**value, "fingerprint": digest(value)}


def audit():
    before = provenance()
    if len(all_pairs()) * len(VIEWS) * 2 != SETTINGS["bootstrap"]["family_size"]:
        raise ValueError("Planned test family does not match the frozen protocol")
    systems = {}
    audit_rows = []
    for model, source, condition in matrix():
        data = load(artifact("records", model, source, condition))
        if data["fingerprint"] != before["mtob_closure_fingerprint"]:
            raise ValueError("MTOB response views are stale")
        rows = data["records"]
        original = sorted(read_jsonl(result_path(model, source, condition)), key=lambda r: r["index"])
        if len(rows) != len(original):
            raise ValueError("MTOB view count mismatch")
        for row, src in zip(rows, original, strict=True):
            expected = {k: src[k] for k in ("index", "source", "reference", "status")}
            expected.update(response_views(src))
            if row != expected:
                raise ValueError("Audited MTOB view differs from deterministic source reconstruction")
        ident = system_id("mtob", model, source, condition)
        systems[ident] = rows
    for model, lang, method in baseline_keys():
        source = next(s for s in sources() if language(s) == lang)
        expected = systems[system_id("mtob", model, source, "ge")]
        cp, rp = baseline_paths(model, lang, method)
        config, rows = load(cp), read_jsonl(rp)
        for key, value in (("model", model), ("model_id", models()[model]["model_id"]),
                           ("language", sources()[source]["language"]), ("method", method),
                           ("material", "baseline"), ("context_kind", "none"), ("source", "grammar"), ("variant", "original")):
            if config.get(key) != value:
                raise ValueError(f"Baseline configuration identity mismatch: {cp}: {key}")
        validate_baseline(rows, config, expected)
        parsed = [baseline_view(r, method) for r in sorted(rows, key=lambda r: r["idx"])]
        systems[system_id("matched", model, lang, method)] = parsed
    for pair in all_pairs():
        if identity(systems[pair["left"]]) != identity(systems[pair["right"]]):
            raise ValueError(f"Unpaired cohort: {pair}")
    for ident, rows in systems.items():
        path = OUT / "records" / f"{ident}.json"
        save(path, {"fingerprint": before["fingerprint"], "records": rows})
        audit_rows.append({"system": ident, "records": len(rows),
                           "empty_raw": sum(not r["raw"].strip() for r in rows),
                           "empty_extracted": sum(not r["extracted"].strip() for r in rows),
                           "extraction_changes": sum(r["raw"] != r["extracted"] for r in rows),
                           "ambiguous_extraction": sum(r["ambiguous_extraction"] for r in rows),
                           "known_truncated": sum(r["truncation"] is True for r in rows),
                           "unknown_truncation": sum(r["truncation"] == "unknown" for r in rows),
                           "baseline_prediction_changes": sum(r.get("recorded_prediction", r["extracted"]) != r["extracted"] for r in rows)})
    if provenance() != before:
        raise ValueError("Inputs changed during audit")
    save(OUT / "provenance.json", before)
    save(OUT / "audit.json", {"state": "complete", "fingerprint": before["fingerprint"],
                              "systems": audit_rows, "pairs": all_pairs(),
                              "records_sha256": {i: sha256(OUT / "records" / f"{i}.json") for i in systems}})
    tsv(OUT / "cohort_audit.tsv", audit_rows)
    tsv(OUT / "planned_comparisons.tsv", all_pairs())
    print(f"[AUDIT] {len(systems)} systems; {sum(len(r) for r in systems.values())} records; {len(all_pairs())} paired comparisons", flush=True)


def checked_audit():
    frozen = load(OUT / "provenance.json")
    if provenance() != frozen:
        raise ValueError("Source/config/parser changed; run audit again")
    audited = load(OUT / "audit.json")
    if audited.get("state") != "complete" or audited["fingerprint"] != frozen["fingerprint"]:
        raise ValueError("Audit missing or stale")
    for ident, value in audited["records_sha256"].items():
        if sha256(OUT / "records" / f"{ident}.json") != value:
            raise ValueError(f"Normalized records changed: {ident}")
    return audited


def metrics_for(ident):
    return OUT / "scores" / f"{ident}.json"


def score():
    audited = checked_audit()
    for ident, record_hash in audited["records_sha256"].items():
        path = metrics_for(ident)
        binding = {"fingerprint": audited["fingerprint"], "records_sha256": record_hash}
        if path.exists() and load(path).get("binding") == binding and load(path).get("state") == "complete":
            continue
        rows = load(OUT / "records" / f"{ident}.json")["records"]
        refs = [r["reference"] for r in rows]
        values = {}
        for view in VIEWS:
            predictions = [r[view] for r in rows]
            if view == "extracted" and predictions == [r["raw"] for r in rows]:
                value = dict(values["raw"])
            else:
                value = evaluate(predictions, refs)
            verify_score_inputs(predictions, refs, value)
            value["input_sha256"] = digest({"predictions": predictions, "references": refs})
            value["empty_predictions"] = sum(not p.strip() for p in predictions)
            values[view] = value
        save(path, {"state": "complete", "binding": binding, "views": values})
        print(f"[SCORE] {ident}: {len(rows)} records", flush=True)
    checked_audit()
    save(OUT / "scoring_complete.json", {"fingerprint": audited["fingerprint"],
                                         "hashes": {i: sha256(metrics_for(i)) for i in audited["records_sha256"]}})
    report()


def checked_scores():
    audited = checked_audit()
    manifest = load(OUT / "scoring_complete.json")
    if manifest["fingerprint"] != audited["fingerprint"] or set(manifest["hashes"]) != set(audited["records_sha256"]):
        raise ValueError("Stale scoring manifest")
    for ident, expected in manifest["hashes"].items():
        if sha256(metrics_for(ident)) != expected:
            raise ValueError(f"Scores changed: {ident}")
    return audited


def analyze():
    audited = checked_scores()
    tests, errors = [], []
    for pair in all_pairs():
        a, b = pair["left"], pair["right"]
        rows_a, rows_b = [load(OUT / "records" / f"{i}.json")["records"] for i in (a, b)]
        if identity(rows_a) != identity(rows_b):
            raise ValueError("Unpaired comparison")
        refs = [r["reference"] for r in rows_a]
        for view in VIEWS:
            name = f"{b}_vs_{a}_{view}"
            path = OUT / "tests" / f"{name}.json"
            binding = {"fingerprint": audited["fingerprint"], "records": [audited["records_sha256"][i] for i in (a, b)],
                       "scores": [sha256(metrics_for(i)) for i in (a, b)], "view": view}
            try:
                if path.exists() and load(path).get("binding") == binding and load(path).get("state") == "complete":
                    tests.extend(load(path)["tests"])
                    continue
                cleaned = []
                for ident, rows in ((a, rows_a), (b, rows_b)):
                    predictions = [r[view] for r in rows]
                    value = load(metrics_for(ident))["views"][view]
                    if value["input_sha256"] != digest({"predictions": predictions, "references": refs}):
                        raise ValueError("Bootstrap inputs differ from scored predictions")
                    pred, clean_refs = verify_score_inputs(predictions, refs, value)
                    cleaned.append(pred)
                print(f"[BOOTSTRAP] {name}: {len(refs)} sentences; 100000 resamples", flush=True)
                values = [{**pair, "view": view, "num_examples": len(refs), **r} for r in paired_test(*cleaned, clean_refs)]
                if any(not math.isfinite(r["p_raw"]) or not 0 <= r["p_raw"] <= 1 for r in values):
                    raise ValueError("Invalid p-value")
                save(path, {"state": "complete", "binding": binding, "tests": values})
                tests.extend(values)
            except Exception as exc:
                error = {"pair": name, "error": f"{type(exc).__name__}: {exc}"}
                errors.append(error)
                save(path, {"state": "error", "binding": binding, **error})
            save(OUT / "analysis_progress.json", {"completed_tests": len(tests), "planned_tests": 360, "errors": errors})
    checked_scores()
    complete = len(tests) == 360 and not errors
    if complete:
        for row, p in zip(tests, holm([r["p_raw"] for r in tests]), strict=True):
            row.update(p_holm=p, significant_holm=p < 0.05)
    save(OUT / "analysis.json", {"state": "complete" if complete else "incomplete", "fingerprint": audited["fingerprint"],
                                 "tests": tests, "errors": errors, "settings": SETTINGS["bootstrap"]})
    if complete:
        tsv(OUT / "significance.tsv", tests)
    report()
    if not complete:
        raise ValueError(f"Only {len(tests)}/360 tests completed; inspect analysis.json")


def relative_change(new, old):
    return 100 * (new - old) / old if old != 0 else None


def report():
    audited = checked_scores()
    analysis_path = OUT / "analysis.json"
    analysis = load(analysis_path) if analysis_path.exists() else {}
    if analysis.get("fingerprint") != audited["fingerprint"]:
        analysis = {}
    tests = analysis.get("tests", []) if analysis.get("state") == "complete" else []
    lookup = {(r["left"], r["right"], r["view"], r["metric"]): r for r in tests}
    rows = []
    for pair in all_pairs():
        for view in VIEWS:
            a, b = [scalar_metrics(load(metrics_for(pair[k]))["views"][view]) for k in ("left", "right")]
            for key in a:
                name = {"bleu": "BLEU", "chrf": "chrF"}.get(key, key)
                scale = 100 if key == "bleu" else 1
                left, right = a[key] * scale, b[key] * scale
                test = lookup.get((pair["left"], pair["right"], view, name), {})
                rows.append({**pair, "view": view, "metric": name, "baseline_score": left, "mtob_score": right,
                             "delta_mtob_minus_baseline": right - left, "relative_change_percent": relative_change(right, left),
                             "higher_is_better": key != "CharacTER", "p_raw": test.get("p_raw"),
                             "p_holm": test.get("p_holm"), "significant_holm": test.get("significant_holm")})
    tsv(OUT / "comparisons.tsv", rows)
    summary = []
    for model in MODELS:
        for method in METHODS:
            for view in VIEWS:
                for metric in ("BLEU", "chrF"):
                    group = [r for r in rows if (r["model"], r["method"], r["view"], r["metric"]) == (model, method, view, metric)]
                    summary.append({"model": model, "baseline_method": method, "view": view, "metric": metric,
                                    "comparisons": len(group), "mtob_higher": sum(r["delta_mtob_minus_baseline"] > 1e-12 for r in group),
                                    "mtob_lower": sum(r["delta_mtob_minus_baseline"] < -1e-12 for r in group),
                                    "equal": sum(abs(r["delta_mtob_minus_baseline"]) <= 1e-12 for r in group),
                                    "significantly_higher": sum(bool(r["significant_holm"]) and r["delta_mtob_minus_baseline"] > 0 for r in group) if tests else None,
                                    "significantly_lower": sum(bool(r["significant_holm"]) and r["delta_mtob_minus_baseline"] < 0 for r in group) if tests else None})
    tsv(OUT / "summary.tsv", summary)
    lines = ["# MTOB versus GRAMMAMT Baselines", "",
             "This is a cross-protocol translation-performance comparison, NOT an isolated estimate of the effect of grammar.", "",
             "## Completion", "",
             f"Cohorts verified: 30 MTOB systems and 24 unique matched_v1 baseline systems; 90 paired comparisons.",
             f"Significance: {len(tests)}/360 verified tests. State: {analysis.get('state', 'pending')}. No significance claim before all tests complete.",
             "No new generation, API calls, GPU allocation, historical output replacement, or changes to the 351-condition primary matrix.", "",
             "## Common Evaluation", "",
             "Every source sentence and reference matches exactly, by index, within each paired comparison. No successful-only subsets.",
             "Gitksan PDF1/PDF2 share the same language baseline and test sentences; they are not independent datasets.",
             "Raw view means the translation response after removal of required output markers/gloss, not the full chain-gloss response.",
             "Extracted view additionally uses the frozen, reference-blind MTOB explanation parser. Multi-sentence translations are preserved.",
             "BLEU and chrF are displayed on 0-100 scales. BLEU: 13a, no smoothing, no effective order; chrF: character order 6, word order 0, beta 2.",
             "Both use paper punctuation cleanup. ROUGE uses local macro F1 (0-1); CharacTER is word-shift-aware, hypothesis-normalized and capped at 1, not standard CER.",
             "Percentage = 100*(MTOB-baseline)/baseline; undefined for zero baselines. Near-zero baselines can produce misleadingly large percentages.",
             "Negative CharacTER changes indicate improvement. Non-significance is not equivalence.",
             "All 360 planned BLEU/chrF tests across both views share one Holm family; 100,000 paired bootstrap resamples, seed 20260917.",
             "Counts below are descriptive comparisons, not independent trials, pooled corpus scores, or a causal analysis.", "",
             "## Descriptive Comparison Counts", "",
             "| Model | Baseline method | View | Metric | Higher / lower / equal | Significant higher / lower |",
             "|---|---|---|---|---|---|"]
    for r in summary:
        sig = f"{r['significantly_higher']} / {r['significantly_lower']}" if tests else "pending"
        lines.append(f"| {r['model']} | {r['baseline_method']} | {r['view']} | {r['metric']} | {r['mtob_higher']} / {r['mtob_lower']} / {r['equal']} | {sig} |")
    for view in VIEWS:
        lines += ["", f"## Absolute Scores: {view.title()} View", "",
                  "Each cell is BLEU / chrF, both on a 0-100 scale. Baseline columns contain no grammar book.", "",
                  "| Model | Source | N | Shot | Chain-gloss | ModelGloss | Ge | Gs | Gl |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for model in MODELS:
            for source in sources():
                ids = [system_id("matched", model, language(source), method) for method in METHODS]
                ids += [system_id("mtob", model, source, condition) for condition in CONDITIONS]
                values = [scalar_metrics(load(metrics_for(i))["views"][view]) for i in ids]
                cells = [f"{v['bleu'] * 100:.2f} / {v['chrf']:.2f}" for v in values]
                lines.append(f"| {model} | {source} | {sources()[source]['test_n']} | " + " | ".join(cells) + " |")
    lines += ["", "## Interpretation Limits", "",
              "- Matched baselines use method-specific GRAMMAMT support/examples or gloss steps; MTOB uses the Appendix C direct translation prompt with grammar passages.",
              "- MTOB: temperature 0.05, output cap 256. Matched_v1: temperature 0.7, output cap 512; seeds and other sampling controls differ.",
              "- MTOB can use refusal/reasoning retries; matched_v1 retains first semantic attempts. Four Qwen3.5 MTOB records were rerun; earlier responses are unavailable.",
              "- MTOB truncation metadata is unavailable. Matched_v1 records it; absence of MTOB metadata is not evidence of no truncation.",
              "- Five Qwen3 chain baseline records have no usable translation; they remain blank predictions. Both views retain all expected sentences.",
              "- Gl contexts differ in actual length; not all are 100K tokens. Different source grammars and retrieval strategies expose different information.",
              "- Common evaluation makes output quality comparable; it does not equalize prompts, computation, information budgets or retry policies.",
              "- Higher scores support a claim about these evaluated system configurations only. They do not show that adding grammar alone caused improvement.",
              "- This analysis does not generate the cancelled MTOB no-book baselines, does not include API models, and does not compare MTOB with every grammar-augmented matched condition.", "",
              "## Artifacts", "",
              "- `comparisons.tsv`: per-condition scores, absolute/percentage differences and adjusted significance.",
              "- `summary.tsv`: descriptive direction counts by model, baseline method and view.",
              "- `cohort_audit.tsv`, `planned_comparisons.tsv`, `audit.json`: matching, exclusions (none), failures and extraction diagnostics.",
              "- `records/`: original response, exact selected span, extraction rules and both evaluation views.",
              "- `scores/`, `tests/`, `significance.tsv`: common metrics and reproducible paired tests.",
              "- `provenance.json`: input/config/normalized-view hashes, parser implementation and evaluator settings."]
    save(OUT / "COMPARISON_REPORT.md", "\n".join(lines) + "\n", text=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("audit", "score", "analyze", "report", "all"))
    args = parser.parse_args()
    if args.stage == "all":
        audit()
        score()
        analyze()
    else:
        globals()[args.stage]()


if __name__ == "__main__":
    main()
