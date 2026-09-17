#!/usr/bin/env python3
"""MTOB no-book Qwen baselines and separate full-denominator comparisons."""
import argparse
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path

from mtob_grammar.closure_audit import (
    ROOT, OUT, MODELS, CONDITIONS, SETTINGS, digest, load, models, sources,
    input_path, input_hashes, offline_encoding, require_audit, artifact, validate_rows,
)
from mtob_grammar.closure_evaluation import (evaluate, scalar_metrics, verify_score_inputs, paired_test, holm)
from mtob_grammar.data import parse_igt
from mtob_grammar.io import sha256, read_jsonl
from mtob_grammar.prompts import appendix_c_prompt
from mtob_grammar.response_views import response_views

BASE = ROOT / "baseline_v1"


def save(path, value):
    path = path.resolve()
    if BASE.resolve() not in path.parents:
        raise ValueError("Baseline writes must stay below baseline_v1")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def result_path(model, source):
    return BASE / "results" / model / source / "results_baseline.jsonl"


def policy(model, source):
    cfg = sources()[source]
    return {"model": models()[model], "source": cfg,
            "entrypoint_sha256": sha256(Path(__file__)),
            "test_sha256": sha256(input_path(cfg["test_file"])),
            "code": {name: sha256(ROOT / "mtob_grammar" / name) for name in ("experiment.py", "backends.py", "prompts.py", "data.py")},
            "initial_seed": "2024 + index", "temperature": 0.05, "max_new_tokens": 256,
            "reasoning_attempt_limit": 3, "refusal_attempt_limit": 3,
            "resume": "Keep every saved terminal record, including failures; no extra rescue rounds",
            "dtype": "float32", "gpus": 1, "thinking": False,
            "context": "Empty grammar block, otherwise unchanged Appendix C prompt"}


def generate(model, source):
    from mtob_grammar.experiment import run
    # Qwen3.5 has its own runtime environment. Evaluator package versions are
    # checked in the CPU evaluator, not confused with generation dependencies.
    if input_hashes() != load(OUT / "provenance.json")["inputs"]:
        raise ValueError("Original MTOB inputs changed")
    manifest = BASE / "manifests" / model / f"{source}.json"
    frozen = policy(model, source)
    versions = {p: importlib.metadata.version(p) for p in ("torch", "transformers")}
    result = result_path(model, source)
    result.parent.mkdir(parents=True, exist_ok=True)
    with result.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest.exists():
            if load(manifest)["policy"] != frozen or load(manifest)["versions"] != versions:
                raise ValueError("Baseline policy changed; refusing mixed-policy resume")
        elif result.exists():
            raise ValueError("Baseline results exist without provenance")
        else:
            save(manifest, {"policy": frozen, "versions": versions})
        print(json.dumps(run(model, source, "baseline", encoding=offline_encoding()), indent=2), flush=True)


def percentage(baseline, condition, lower_is_better=False):
    if baseline == 0:
        return None
    return 100 * ((baseline - condition) if lower_is_better else (condition - baseline)) / baseline


def prepare_scores():
    frozen = require_audit()
    old = load(OUT / "scoring_complete.json")
    for rel, value in old["metrics_sha256"].items():
        if sha256(OUT / rel) != value:
            raise ValueError("Frozen grammar metrics changed")
    result_hashes = {f"{m}/{s}": sha256(result_path(m, s)) for m in MODELS for s in sources()}
    binding = {"grammar_fingerprint": frozen["fingerprint"], "baseline_results": result_hashes,
               "baseline_manifests": {f"{m}/{s}": sha256(BASE / "manifests" / m / f"{s}.json") for m in MODELS for s in sources()},
               "code_sha256": sha256(Path(__file__)), "settings": SETTINGS,
               "comparison_family": "120 baseline-versus-Ge/Gs/Gl tests, separate from original 120 grammar-pair tests"}
    save(BASE / "evaluation" / "provenance.json", binding)
    scored = {}
    comparisons = []
    for model in MODELS:
        for source, cfg in sources().items():
            manifest = load(BASE / "manifests" / model / f"{source}.json")
            if manifest["policy"] != policy(model, source):
                raise ValueError("Generation policy no longer matches source/code")
            rows = read_jsonl(result_path(model, source))
            examples = parse_igt(input_path(cfg["test_file"]))
            state, errors = validate_rows(rows, examples, model, source, "baseline")
            if state != "complete":
                raise ValueError(f"Incomplete/invalid baseline {model}/{source}: {errors}")
            views = []
            for r in sorted(rows, key=lambda x: x["index"]):
                args = (cfg["language"], cfg["location"], r["source"], "")
                if r["prompt"] != appendix_c_prompt(*args):
                    raise ValueError("Baseline prompt mismatch")
                if r.get("refusal_retry_prompt") and r["refusal_retry_prompt"] != appendix_c_prompt(*args, refusal=True):
                    raise ValueError("Baseline refusal prompt mismatch")
                views.append({"index": r["index"], "source": r["source"], "reference": r["reference"], "status": r["status"], **response_views(r)})
            scores = {}
            for view in ("raw", "extracted"):
                predictions, refs = [r[view] for r in views], [r["reference"] for r in views]
                value = evaluate(predictions, refs)
                verify_score_inputs(predictions, refs, value)
                scores[view] = value
                for condition in CONDITIONS:
                    grammar = load(artifact("metrics", model, source, condition))["views"][view]
                    base_vals, cond_vals = scalar_metrics(value), scalar_metrics(grammar)
                    comparisons.append({"model": model, "source": source, "condition": condition, "view": view,
                                        "num_examples": len(views), "baseline": base_vals, "grammar": cond_vals,
                                        "absolute_difference": {k: cond_vals[k] - b for k, b in base_vals.items()},
                                        "relative_improvement_percent": {k: percentage(b, cond_vals[k], k == "CharacTER") for k, b in base_vals.items()},
                                        "zero_baseline_metrics": [k for k, b in base_vals.items() if b == 0]})
            data = {"binding": binding, "records": views, "scores": scores}
            save(BASE / "evaluation" / model / f"{source}.json", data)
            scored[(model, source)] = data
            print(f"[SCORED] {model}/{source}: {len(views)} records", flush=True)
    save(BASE / "evaluation" / "comparisons.json", {"binding": binding, "comparisons": comparisons})
    return binding, scored


def analyze():
    binding, scored = prepare_scores()
    tests, errors = [], []
    for (model, source), data in scored.items():
        for condition in CONDITIONS:
            context = load(artifact("records", model, source, condition))["records"]
            base = data["records"]
            if [(r["index"], r["source"], r["reference"]) for r in base] != [(r["index"], r["source"], r["reference"]) for r in context]:
                raise ValueError("Baseline and context cohorts differ")
            for view in ("raw", "extracted"):
                path = BASE / "evaluation" / "pairs" / f"{model}_{source}_{condition}_{view}.json"
                pair_binding = {"binding": digest(binding), "context_records": sha256(artifact("records", model, source, condition)),
                                "view": view, "condition": condition}
                try:
                    if path.exists() and load(path).get("binding") == pair_binding and load(path).get("state") == "complete":
                        values = load(path)["tests"]
                    else:
                        refs = [r["reference"] for r in base]
                        left, clean_refs = verify_score_inputs([r[view] for r in base], refs, data["scores"][view])
                        right, _ = verify_score_inputs([r[view] for r in context], refs, load(artifact("metrics", model, source, condition))["views"][view])
                        print(f"[BOOTSTRAP] {model}/{source}/baseline-{condition}/{view}", flush=True)
                        values = [{"model": model, "source": source, "condition": condition, "view": view,
                                   "num_examples": len(base), **r} for r in paired_test(left, right, clean_refs)]
                        save(path, {"state": "complete", "binding": pair_binding, "tests": values})
                    tests.extend(values)
                except Exception as exc:
                    errors.append({"model": model, "source": source, "condition": condition, "view": view, "error": str(exc)})
                save(BASE / "evaluation" / "progress.json", {"completed_tests": len(tests), "planned_tests": 120, "errors": errors})
    complete = len(tests) == 120 and not errors
    if complete:
        for row, p in zip(tests, holm([t["p_raw"] for t in tests]), strict=True):
            row.update(p_holm=p, significant_holm=p < 0.05)
    require_audit()
    for key, h in binding["baseline_results"].items():
        if sha256(result_path(*key.split("/"))) != h:
            raise ValueError("Baseline changed during evaluation")
    save(BASE / "evaluation" / "significance.json", {"state": "complete" if complete else "incomplete", "binding": binding, "tests": tests, "errors": errors})
    lines = ["# MTOB Baseline Comparisons", "", f"Statistical analysis: {'complete' if complete else 'incomplete'}; {len(tests)}/120 tests.",
             "", "Percent improvements use baseline as denominator. Zero baselines are undefined, not zero improvement. BLEU below is 0-1; chrF is 0-100.",
             "Only the grammar block is removed from Appendix C. Four historical Qwen3.5 context records had additional rerun rounds; their seed/attempt asymmetry remains a limitation.",
             "These are separate from the main matched_v1 matrix and from the earlier Ge/Gs/Gl-only significance family.", "",
             "| Model | Source | Context | View | Baseline BLEU | Context BLEU | BLEU change % | Baseline chrF | Context chrF | chrF change % |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in load(BASE / "evaluation" / "comparisons.json")["comparisons"]:
        def pct(metric):
            v = row['relative_improvement_percent'][metric]
            return 'undefined' if v is None else f'{v:+.2f}'
        a, b = row['baseline'], row['grammar']
        lines.append(f"| {row['model']} | {row['source']} | {row['condition']} | {row['view']} | {a['bleu']:.6f} | {b['bleu']:.6f} | {pct('bleu')} | {a['chrf']:.4f} | {b['chrf']:.4f} | {pct('chrf')} |")
    if complete:
        lines += ["", f"Holm-significant tests: {sum(t['significant_holm'] for t in tests)}/120. See significance.json for p-values and signed effects.",
                  "A non-significant result is not evidence of equivalence. Percentage gains alone do not establish significance."]
    save(BASE / "evaluation" / "COMPARISON_REPORT.md", "\n".join(lines) + "\n")
    if not complete:
        raise RuntimeError("Baseline comparison family incomplete; see evaluation/significance.json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("generate", "analyze", "list"))
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--source", choices=sources())
    args = p.parse_args()
    if args.stage == "generate":
        if not args.model or not args.source:
            p.error("generate needs --model and --source")
        generate(args.model, args.source)
    elif args.stage == "analyze":
        try:
            analyze()
        except Exception as exc:
            save(BASE / "evaluation" / "error.json", {"error": f"{type(exc).__name__}: {exc}"})
            raise
    else:
        for m in MODELS:
            for s, c in sources().items():
                print(m, s, "baseline", c["test_n"], result_path(m, s))


if __name__ == "__main__":
    main()
