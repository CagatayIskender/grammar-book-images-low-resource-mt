"""Frozen 2x2 development-set diagnostic; never changes production settings."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import time

from experiment_io import ROOT, DATA, atomic_json, build_chain_prompt, dataset, parse_chain, sha256

BASE = ROOT / "results/diagnostics/tsez_chain_support_decoding_v1"
ARMS = ("covered_greedy", "uncovered_greedy", "covered_sample", "uncovered_sample")


def read_igt(path):
    records = []
    for block in re.split(r"\n\s*\n", path.read_text(encoding="utf-8")):
        row = {}
        for line in block.splitlines():
            for tag, name in (("\\t", "source"), ("\\g", "gloss"), ("\\l", "reference")):
                if line.startswith(tag):
                    row[name] = line[2:].strip()
        if row.get("source") and row.get("reference"):
            row["reference"] = row["reference"].split("||")[0].strip()
            records.append(row)
    return records


def aligned_support(covered, uncovered):
    if len(covered) != len(uncovered) or any(
        (a["source"], a["reference"]) != (b["source"], b["reference"])
        for a, b in zip(covered, uncovered)
    ):
        raise ValueError("Covered/uncovered training source or reference mismatch")
    if any(not x.get("gloss", "").strip() for x in uncovered[:21]):
        raise ValueError("Missing authoritative training gloss")
    return uncovered[:21]


def select_dev(dev, excluded, seed=20260910):
    eligible = [i for i, row in enumerate(dev) if row["source"] not in excluded]
    # Deduplicate by source before the fixed, reference-independent selection.
    seen, unique = set(), []
    for i in eligible:
        if dev[i]["source"] not in seen:
            unique.append(i)
            seen.add(dev[i]["source"])
    longest = sorted(unique, key=lambda i: len(dev[i]["source"]), reverse=True)[:3]
    rest = [i for i in unique if i not in longest]
    if len(longest) != 3 or len(rest) < 5:
        raise ValueError("Insufficient disjoint development examples")
    return sorted(longest + random.Random(seed).sample(rest, 5))


class PresencePenalty:
    """Subtract a constant once per generated token type, ignoring prompt tokens."""
    def __init__(self, prompt_length, penalty):
        self.prompt_length, self.penalty = prompt_length, penalty

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_length:]
        if generated.shape[1] == 0:
            return scores
        seen = scores.new_zeros(scores.shape)
        seen.scatter_(1, generated, 1)
        return scores - self.penalty * seen


def generation_policy(sample):
    if not sample:
        return {"do_sample":False}
    return {"do_sample":True, "temperature":0.7, "top_p":0.8, "top_k":20,
            "min_p":0.0, "repetition_penalty":1.0}


def repetition_warning(text):
    """Diagnostic flag only: four adjacent repetitions of a 3-12 token span."""
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    for width in range(3, 13):
        for start in range(len(tokens) - 4 * width + 1):
            span = tokens[start:start + width]
            if tokens[start:start + 4 * width] == span * 4:
                return True
    return False


def prepare():
    covered = dataset("Tsez", "train")
    train_path = DATA / "Tsez/ddo-train-track1-uncovered"
    dev_path = DATA / "Tsez/ddo-dev-track1-uncovered"
    support = aligned_support(covered, read_igt(train_path))
    dev = read_igt(dev_path)
    excluded = {r["source"] for r in covered} | {r["source"] for r in dataset("Tsez")}
    indices = select_dev(dev, excluded)
    cfgs = {model:json.loads((ROOT / f"configs/experiments/{model}_tsez_grammar_original_chain_gloss_pdfpages_impactful_jpg.json").read_text()) for model in ("qwen3", "qwen35")}
    images = cfgs["qwen3"]["context_files"]
    if images != cfgs["qwen35"]["context_files"] or len(images) != 9:
        raise ValueError("Expected the same nine original images for both models")
    code = ["runners/diagnose_tsez_chain.py", "runners/experiment_io.py", "runners/fp32_attention.py", "runners/run_audited_context.py"]
    value = {"design":"2x2: training gloss availability x decoding policy", "seed":20260910,
             "arms":list(ARMS), "examples":[dict(dev[i], dev_index=i) for i in indices],
             "covered_support":covered[:21], "uncovered_support":support,
             "model_ids":{m:c["model_id"] for m,c in cfgs.items()}, "images":images,
             "input_hashes":{str(p):sha256(p) for p in (train_path, dev_path, DATA / "Tsez/ddo-train-track1-covered", DATA / "Tsez/ddo-test-track1-uncovered")},
             "image_hashes":{p:sha256(ROOT / p) for p in images}, "code_hashes":{p:sha256(ROOT / p) for p in code},
             "max_new_tokens":512, "retries":0, "dtype":"float32", "gpu_count":1,
             "sampling":dict(generation_policy(True), presence_penalty=1.5),
             "sources":["https://huggingface.co/Qwen/Qwen3.5-9B", "https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct"],
             "limitations":"8 dev examples, 1 fixed seed; no guarantee of linguistic correctness or full-test success. No production gate bypass."}
    target = BASE / "manifest.json"
    if target.exists() and json.loads(target.read_text()) != value:
        raise ValueError("Diagnostic already frozen; use a new directory for a new design")
    atomic_json(target, value)
    print(f"Prepared {len(indices)} disjoint dev examples, 4 arms, 32 generations per model: {target}")


def generate(backend, system, user, images, sample, seed):
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import LogitsProcessorList
    from fp32_attention import expanded_sdpa_context
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    messages = [{"role":"system", "content":[{"type":"text", "text":system}]},
                {"role":"user", "content":[{"type":"image", "image":x} for x in images] + [{"type":"text", "text":user}]}]
    prompt = backend.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if backend.model_kind == "qwen35" and not prompt.endswith("<think>\n\n</think>\n\n"):
        raise RuntimeError("Disabled-thinking template missing")
    inputs = backend.processor(text=[prompt], images=images, return_tensors="pt").to("cuda:0")
    length = inputs.input_ids.shape[1]
    kwargs = generation_policy(sample)
    if sample:
        kwargs["logits_processor"] = LogitsProcessorList([PresencePenalty(length, 1.5)])
    started = time.monotonic()
    with torch.inference_mode(), expanded_sdpa_context(), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        out = backend.model.generate(**inputs, max_new_tokens=512, **kwargs)
    ids = out[0, length:]
    raw = backend.processor.decode(ids, skip_special_tokens=True).strip()
    special = backend.processor.decode(ids, skip_special_tokens=False)
    eos = backend.model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else (eos or [])
    truncated = len(ids) >= 512 and int(ids[-1]) not in eos
    return {"raw":raw, "special":special, "truncated":truncated, "generated_tokens":len(ids),
            "input_tokens":length, "runtime_seconds":time.monotonic()-started,
            "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest()}


def run(model):
    import fcntl
    import sacrebleu
    from PIL import Image
    from fp32_attention import ATTENTION_TAG
    from run_audited_context import Backend
    manifest_path = BASE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for field in ("code_hashes", "image_hashes", "input_hashes"):
        for path, expected in manifest[field].items():
            if sha256(ROOT / path) != expected:
                raise ValueError(f"Frozen diagnostic input changed: {path}")
    fingerprint = sha256(manifest_path)
    result = BASE / f"{model}.jsonl"
    with (BASE / f"{model}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = [json.loads(line) for line in result.read_text().splitlines()] if result.exists() else []
        order = [(arm, ex) for arm in ARMS for ex in manifest["examples"]]
        if len(rows) > len(order):
            raise ValueError("Oversized output")
        for row, (arm, ex) in zip(rows, order):
            if row["arm"] != arm or row["dev_index"] != ex["dev_index"] or row["manifest_sha256"] != fingerprint:
                raise ValueError("Invalid diagnostic resume provenance")
        backend = Backend(manifest["model_ids"][model], model, ATTENTION_TAG)
        images = [Image.open(ROOT / p).convert("RGB") for p in manifest["images"]]
        with result.open("a", encoding="utf-8") as handle:
            for arm, ex in order[len(rows):]:
                support = manifest["uncovered_support"] if arm.startswith("uncovered") else manifest["covered_support"]
                # Only the source, never development gloss/reference, enters the prompt.
                system, user = build_chain_prompt(support, "Tsez", ex["source"], images=True)
                row = generate(backend, system, user, images, arm.endswith("sample"), manifest["seed"] + ex["dev_index"])
                row.update(arm=arm, dev_index=ex["dev_index"], source=ex["source"], reference=ex["reference"],
                           gold_gloss=ex.get("gloss", ""), manifest_sha256=fingerprint)
                row.update(prediction="", generated_gloss="", error=None)
                try:
                    if row["truncated"]:
                        raise ValueError("truncated")
                    if "<think>" in row["special"] or "</think>" in row["special"]:
                        raise ValueError("reasoning_violation")
                    row["generated_gloss"], row["prediction"] = parse_chain(row["raw"])
                except ValueError as exc:
                    row["error"] = str(exc)
                row["repetition_warning"] = repetition_warning(row["raw"])
                handle.write(json.dumps(row, ensure_ascii=False)+"\n")
                handle.flush()
                import os
                os.fsync(handle.fileno())
                rows.append(row)
                print(f"[PROGRESS] {len(rows)}/{len(order)} {arm} dev={ex['dev_index']} error={row['error']} tokens={row['generated_tokens']}", flush=True)
        metrics = {}
        for arm in ARMS:
            group = [r for r in rows if r["arm"] == arm]
            metrics[arm] = {"records":len(group), "invalid":sum(bool(r["error"]) for r in group),
                            "truncated":sum(r["truncated"] for r in group),
                            "repetition_warnings":sum(r["repetition_warning"] for r in group),
                            "gloss_chrf":sacrebleu.corpus_chrf([r["generated_gloss"] for r in group], [[r["gold_gloss"] for r in group]], word_order=2).score,
                            "chrf":sacrebleu.corpus_chrf([r["prediction"] for r in group], [[r["reference"] for r in group]], word_order=2).score}
        atomic_json(BASE / f"{model}.summary.json", {"manifest_sha256":fingerprint, "arms":metrics,
                    "status":"diagnostic_complete", "production_approved":False,
                    "peak_gpu_bytes":backend.torch.cuda.max_memory_allocated()})
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--model", choices=["qwen3", "qwen35"])
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.model:
        run(args.model)
    else:
        parser.error("Choose --prepare or --model")
