"""Versioned sampled decoding; frozen greedy diagnostics and repairs stay unchanged."""
import argparse
import fcntl
import hashlib
import json
import os
import re
import time

from experiment_io import ROOT, atomic_json, dataset, parse_chain, read_records, sha256, validate_records
from fp32_attention import ATTENTION_TAG, expanded_sdpa_context
from diagnose_tsez_chain import PresencePenalty, generation_policy, repetition_warning
from run_audited_context import Backend, prompt_for, write_basic_metrics

REVISION = "sampled_v1"
SEED = 20260910


def result_path(cfg, preflight):
    if preflight:
        return ROOT / "results/preflight" / f"{cfg['id']}_{ATTENTION_TAG}_{REVISION}.jsonl"
    path = ROOT / cfg["results"]
    return path.with_name(path.stem + f"_{REVISION}.jsonl")


def provenance_for(cfg):
    files = ["runners/run_sampled_context.py", "runners/run_audited_context.py",
             "runners/experiment_io.py", "runners/fp32_attention.py", "runners/diagnose_tsez_chain.py"]
    files += [str(p.relative_to(ROOT)) for group in ("qwen3", "qwen35", "baseline")
              for p in (ROOT / "runners" / group).glob("run_grammamt*.py")]
    return {"config":cfg, "test":dataset(cfg["language"]), "support":dataset(cfg["language"], "train")[:21],
            "context_hashes":{p:sha256(ROOT / p) for p in cfg["context_files"]},
            "code_hashes":{p:sha256(ROOT / p) for p in files}, "revision":REVISION,
            "policy":dict(generation_policy(True), presence_penalty=1.5, max_new_tokens=512),
            "seed":SEED, "seed_rule":"base + 2 * original_test_index + attempt", "max_attempts":2,
            "dtype":"float32", "gpu_count":1, "attention":ATTENTION_TAG, "enable_thinking":False,
            "support_policy":"first 21 covered training examples; no additional gold gloss"}


def fingerprint_of(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def preflight_ready(cfg, provenance=None):
    path = result_path(cfg, True)
    try:
        provenance = provenance or provenance_for(cfg)
        fingerprint = fingerprint_of(provenance)
        status = json.loads(path.with_suffix(".status.json").read_text())
        if json.loads(path.with_suffix(".provenance.json").read_text()) != provenance:
            return False
        rows = read_records(path)
        target = sorted(provenance["test"], key=lambda ex:len(ex["source"]), reverse=True)[:3]
        validate_records(rows, target, complete=True, fingerprint=fingerprint)
        return (status.get("status") == "complete" and status.get("fingerprint") == fingerprint
                and status.get("records") == status.get("expected") == 3
                and status.get("results_sha256") == sha256(path)
                and all(r[cfg["prediction_key"]].get("prediction", "").strip()
                        and not r[cfg["prediction_key"]].get("error") for r in rows))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def generate(backend, system, user, images, seed):
    torch = backend.torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import LogitsProcessorList
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    messages = [{"role":"system", "content":[{"type":"text", "text":system}]},
                {"role":"user", "content":[{"type":"image", "image":x} for x in images]
                 + [{"type":"text", "text":user}]}]
    prompt = backend.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if backend.model_kind == "qwen35" and not prompt.endswith("<think>\n\n</think>\n\n"):
        raise RuntimeError("Disabled-thinking template missing")
    kwargs = {"text":[prompt], "return_tensors":"pt"}
    if images:
        kwargs["images"] = images
    inputs = backend.processor(**kwargs).to("cuda:0")
    length = inputs.input_ids.shape[1]
    with torch.inference_mode(), expanded_sdpa_context(), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        out = backend.model.generate(**inputs, max_new_tokens=512, **generation_policy(True),
                                     logits_processor=LogitsProcessorList([PresencePenalty(length, 1.5)]))
    ids = out[0, length:]
    eos = backend.model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else (eos or [])
    raw = backend.processor.decode(ids, skip_special_tokens=True).strip()
    return {"raw":raw, "raw_with_special_tokens":backend.processor.decode(ids, skip_special_tokens=False),
            "truncated":len(ids) >= 512 and int(ids[-1]) not in eos, "generated_tokens":len(ids),
            "input_tokens":length, "seed":seed, "repetition_warning":repetition_warning(raw),
            "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest()}


def parse_output(cfg, attempt):
    if attempt["truncated"]:
        raise ValueError("truncated")
    if re.search(r"<\s*/?think\b", attempt["raw_with_special_tokens"], re.I):
        raise ValueError("reasoning_violation")
    if attempt["repetition_warning"]:
        raise ValueError("repeated_output")
    if cfg["condition"] == "chain_gloss":
        return parse_chain(attempt["raw"])
    from run_grammamt_Qwen_grammar_images import extract_translation
    prediction = extract_translation(attempt["raw"])
    if not prediction.strip():
        raise ValueError("empty_output")
    return "", prediction


def run(args):
    cfg = json.loads((ROOT / args.config).read_text())
    if not (cfg["condition"] == "chain_gloss" or
            (cfg["model"], cfg["language"], cfg["condition"], cfg["material"]) ==
            ("qwen3", "Tsez", "shot", "pdfpages_impactful_jpg")):
        raise ValueError("Sampled policy is scoped to corrected chain and the failed Tsez Shot repair")
    provenance = provenance_for(cfg)
    fingerprint = fingerprint_of(provenance)
    if not args.preflight and not preflight_ready(cfg, provenance):
        raise ValueError("This exact condition needs a successful current sampled preflight")
    expected = provenance["test"]
    target = sorted(expected, key=lambda ex:len(ex["source"]), reverse=True)[:3] if args.preflight else expected
    result = result_path(cfg, args.preflight)
    result.parent.mkdir(parents=True, exist_ok=True)
    with result.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = read_records(result)
        validate_records(rows, target, fingerprint=fingerprint)
        atomic_json(result.with_suffix(".provenance.json"), provenance)
        start = time.monotonic()
        backend = None
        try:
            if len(rows) < len(target):
                images, text = [], ""
                if cfg["context_kind"] == "image":
                    from PIL import Image
                    images = [Image.open(ROOT / p).convert("RGB") for p in cfg["context_files"]]
                else:
                    text = (ROOT / cfg["context_files"][0]).read_text(encoding="utf-8").strip()
                backend = Backend(cfg["model_id"], cfg["model"], ATTENTION_TAG)
                print(f"[INFO] {cfg['id']} {REVISION} FP32 H100=1 images={len(images)}", flush=True)
                with result.open("a", encoding="utf-8") as handle:
                    for i in range(len(rows), len(target)):
                        ex = target[i]
                        source_index = expected.index(ex)
                        system, user = prompt_for(cfg, provenance["support"], ex, text, "")
                        attempts, gloss, prediction, error = [], "", "", None
                        for retry in range(2):
                            item = generate(backend, system, user, images, SEED + 2 * source_index + retry)
                            attempts.append(item)
                            try:
                                gloss, prediction = parse_output(cfg, item)
                                error = None
                                break
                            except ValueError as exc:
                                error = str(exc)
                                user += "\nFollow the requested output format exactly. Do not add analysis."
                        row = {"idx":i, "source":ex["source"], "reference":ex["reference"], "fingerprint":fingerprint,
                               cfg["prediction_key"]:{"prediction":prediction if not error else "",
                                   "generated_gloss":gloss, "raw":attempts[-1]["raw"], "attempts":attempts, "error":error}}
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                        rows.append(row)
                        print(f"[PROGRESS] {len(rows)}/{len(target)} error={error}", flush=True)
            validate_records(rows, target, complete=True, fingerprint=fingerprint)
            failures = sum(bool(r[cfg["prediction_key"]].get("error")) for r in rows)
            status = {"status":"complete_with_errors" if failures else "complete", "records":len(rows),
                      "expected":len(target), "invalid_outputs":failures, "fingerprint":fingerprint,
                      "results_sha256":sha256(result), "runtime_seconds":time.monotonic()-start}
            if backend:
                status["peak_gpu_bytes"] = backend.torch.cuda.max_memory_allocated()
            atomic_json(result.with_suffix(".status.json"), status)
            if not args.preflight:
                metrics = ROOT / cfg["metrics"]
                scored_cfg = dict(cfg, metrics=str(metrics.with_name(metrics.stem + f"_{REVISION}.json").relative_to(ROOT)))
                write_basic_metrics(scored_cfg, rows, target, result, fingerprint)
            if failures:
                raise RuntimeError(f"{failures} invalid outputs; do not approve production")
        except Exception as exc:
            if len(rows) != len(target):
                atomic_json(result.with_suffix(".status.json"), {"status":"failed", "error":str(exc),
                            "records":len(rows), "expected":len(target), "fingerprint":fingerprint})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight", action="store_true")
    run(parser.parse_args())
