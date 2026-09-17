"""No-grammar Qwen3.5 controls, with separate greedy and sampled provenance."""
import argparse
import fcntl
import hashlib
import json
import os
import re
import time
from types import SimpleNamespace

from experiment_io import ROOT, atomic_json, build_chain_prompt, dataset, parse_chain, read_records, sha256, validate_records
from fp32_attention import ATTENTION_TAG
from run_audited_context import Backend
from run_sampled_context import generate, SEED


def baseline_prompt(condition, support, language, source, predicted_gloss=""):
    if condition == "chain_gloss":
        system, user = build_chain_prompt(support, language, source)
        context = f"Here is a grammar reference summary for {language}:\n\n"
        if user.count(context) != 1:
            raise ValueError("Changed chain context template")
        user = user.replace(context, "", 1).replace(
            "Use the grammar reference to produce the gloss first", "Produce the gloss first", 1)
    else:
        import run_grammamt_Qwen35_context as prompts
        examples = [SimpleNamespace(**r) for r in support]
        if condition == "modelgloss":
            system, user = prompts.build_prompt_model_gloss(examples, language, source, predicted_gloss, "text", "")
            user = user.replace("Use the grammar reference and predicted gloss as supporting context.",
                                "Use the predicted gloss as supporting context.", 1)
        else:
            system, user = prompts.build_prompt_shot(examples, language, source, "text", "")
            user = user.replace("\nUse the grammar reference as supporting context.\n", "\n", 1)
        context = prompts.build_context_intro(language, "text", "")
        if user.count(context) != 1:
            raise ValueError("Changed context template")
        user = user.replace(context, "", 1)
    if "grammar reference" in user or "attached images" in user:
        raise ValueError("Baseline still contains a grammar-context instruction")
    return system, user


def parse(condition, policy, item):
    if item["truncated"]:
        raise ValueError("truncated")
    if re.search(r"<\s*/?think\b", item["raw_with_special_tokens"], re.I):
        raise ValueError("reasoning_violation")
    if policy == "sampled_v1" and item["repetition_warning"]:
        raise ValueError("repeated_output")
    if condition == "chain_gloss":
        return parse_chain(item["raw"])
    match = re.fullmatch(r"\s*FINAL_TRANSLATION:\s*([^\n]+)\s*", item["raw"])
    if not match or not match[1].strip():
        raise ValueError("invalid_translation_format")
    return "", match[1].strip()


def run(language):
    import sacrebleu
    expected, support = dataset(language), dataset(language, "train")[:21]
    from run_grammamt_ModelGloss import load_glosslm_predictions
    glosses = load_glosslm_predictions(language)
    if len(glosses) < len(expected) or any(not x for x in glosses[:len(expected)]):
        raise ValueError("Incomplete predicted glosses")
    base = ROOT / "results/baseline/qwen35" / language.lower() / "grammar/original"
    base.mkdir(parents=True, exist_ok=True)
    code = ["runners/run_qwen35_baselines.py", "runners/run_sampled_context.py", "runners/run_audited_context.py",
            "runners/experiment_io.py", "runners/fp32_attention.py", "runners/diagnose_tsez_chain.py",
            "runners/qwen35/run_grammamt_Qwen35_context.py", "runners/baseline/run_grammamt_ModelGloss.py"]
    backend, problems = None, []
    modes = [("chain_gloss", "sampled_v1"), ("shot", "greedy"), ("modelgloss", "greedy"),
             ("shot", "sampled_v1"), ("modelgloss", "sampled_v1"), ("chain_gloss", "greedy")]
    with (base / "baseline_suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for condition, policy in modes:
            key = f"qwen35_context_{condition}"
            result = base / f"{condition}_{policy}.jsonl"
            provenance = {"model":"Qwen/Qwen3.5-9B", "language":language, "condition":condition,
                          "policy":policy, "test":expected, "support":support, "grammar_context":None,
                          "dtype":"float32", "gpu_count":1, "attention":ATTENTION_TAG, "enable_thinking":False,
                          "max_new_tokens":512, "max_attempts":2, "seed":SEED,
                          "sampling":{"temperature":0.7,"top_p":0.8,"top_k":20,"presence_penalty":1.5,"repetition_penalty":1.0} if policy=="sampled_v1" else {"do_sample":False},
                          "code_hashes":{p:sha256(ROOT / p) for p in code}}
            if condition == "modelgloss":
                provenance["predicted_glosses"] = glosses[:len(expected)]
            fp = hashlib.sha256(json.dumps(provenance, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            old_provenance = result.with_suffix(".provenance.json")
            if old_provenance.exists() and json.loads(old_provenance.read_text()) != provenance:
                raise ValueError("Baseline provenance changed; do not overwrite")
            for preflight in (True, False):
                path = base / f"{condition}_{policy}.preflight.jsonl" if preflight else result
                target = sorted(expected, key=lambda r:len(r["source"]), reverse=True)[:3] if preflight else expected
                rows = read_records(path)
                validate_records(rows, target, fingerprint=fp)
                atomic_json(path.with_suffix(".provenance.json"), provenance)
                started = time.monotonic()
                if len(rows) < len(target) and backend is None:
                    backend = Backend("Qwen/Qwen3.5-9B", "qwen35", ATTENTION_TAG)
                with path.open("a", encoding="utf-8") as handle:
                    for index in range(len(rows), len(target)):
                        ex = target[index]
                        original = expected.index(ex)
                        system, user = baseline_prompt(condition, support, language, ex["source"], glosses[original])
                        attempts, prediction, gloss, error = [], "", "", None
                        for retry in range(2):
                            if policy == "sampled_v1":
                                item = generate(backend, system, user, [], SEED+2*original+retry)
                            else:
                                raw, truncated, special, tokens = backend.generate(system, user, [])
                                item = {"raw":raw,"truncated":truncated,"raw_with_special_tokens":special,"input_tokens":tokens}
                            attempts.append(item)
                            try:
                                gloss, prediction = parse(condition, policy, item)
                                error = None
                                break
                            except ValueError as exc:
                                error = str(exc)
                                user += "\nFollow the requested output format exactly. Do not add analysis."
                        row = dict(idx=index, source=ex["source"], reference=ex["reference"], fingerprint=fp)
                        row[key] = {"prediction":prediction if not error else "", "generated_gloss":gloss,
                                    "error":error, "raw":attempts[-1]["raw"], "attempts":attempts}
                        handle.write(json.dumps(row, ensure_ascii=False)+"\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                        rows.append(row)
                        print(f"[PROGRESS] {language}/{condition}/{policy} preflight={preflight} {len(rows)}/{len(target)} error={error}", flush=True)
                failures = sum(bool(r[key].get("error")) for r in rows)
                atomic_json(path.with_suffix(".status.json"), {"status":"complete_with_errors" if failures else "complete",
                            "records":len(rows), "expected":len(target), "invalid_outputs":failures,
                            "fingerprint":fp, "results_sha256":sha256(path), "runtime_seconds":time.monotonic()-started})
                if preflight and failures:
                    problems.append(f"{condition}/{policy}: failed preflight")
                    break
                if not preflight:
                    refs, hyps = [[r["reference"] for r in rows]], [r[key]["prediction"] for r in rows]
                    metric = ROOT / "metrics" / result.relative_to(ROOT / "results").with_suffix(".json")
                    atomic_json(metric, {key:{"bleu":sacrebleu.corpus_bleu(hyps,refs).score,
                                             "chrf":sacrebleu.corpus_chrf(hyps,refs,word_order=2).score,"xcomet":None},
                                        "records":len(rows),"expected_records":len(expected),"invalid_outputs":failures,
                                        "fingerprint":fp,"results_sha256":sha256(path)})
                    if failures:
                        problems.append(f"{condition}/{policy}: {failures} invalid outputs")
    if problems:
        raise RuntimeError("; ".join(problems))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True, choices=["Gitksan", "Lezgi", "Natugu", "Tsez"])
    run(parser.parse_args().language)
