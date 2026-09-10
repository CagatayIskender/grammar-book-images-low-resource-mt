"""Single-H100 FP32 generation with append-only, provenance-checked resume."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from experiment_io import ROOT, CHAIN_PROMPT_REVISION, atomic_json, build_chain_prompt, dataset, parse_chain, preflight_stem, read_records, sha256, validate_records
from fp32_attention import ATTENTION_TAG, expanded_sdpa_context, verify_efficient_kernel

for group in ("qwen3", "qwen35", "baseline"):
    sys.path.insert(0, str(ROOT / "runners" / group))


class Backend:
    def __init__(self, model_id, model, attention):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        if torch.cuda.device_count() != 1 or "H100" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Exactly one visible H100 is required; CPU/offload is forbidden")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if attention == ATTENTION_TAG:
            verify_efficient_kernel()
        self.torch, self.model_kind, self.attention = torch, model, attention
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, device_map={"": 0}, dtype=torch.float32,
            attn_implementation="sdpa", trust_remote_code=True,
        ).eval()
        for p in self.model.parameters():
            if p.device.type != "cuda" or (p.is_floating_point() and p.dtype != torch.float32):
                raise RuntimeError("Non-FP32 or offloaded model parameter detected")

    def generate(self, system, user, images):
        torch = self.torch
        content = [{"type": "image", "image": x} for x in images] + [{"type": "text", "text": user}]
        messages = [{"role":"system", "content":[{"type":"text", "text":system}]}, {"role":"user", "content":content}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if self.model_kind == "qwen35" and not prompt.endswith("<think>\n\n</think>\n\n"):
            raise RuntimeError("Qwen3.5 disabled-thinking template suffix missing")
        kwargs = {"text":[prompt], "return_tensors":"pt"}
        if images:
            kwargs["images"] = images
        inputs = self.processor(**kwargs).to("cuda:0")
        from torch.nn.attention import SDPBackend, sdpa_kernel
        selected = SDPBackend.MATH if self.attention == "math" else SDPBackend.EFFICIENT_ATTENTION
        # No implicit lower precision, CPU placement, or attention-backend fallback.
        adapter = expanded_sdpa_context() if self.attention == ATTENTION_TAG else contextlib.nullcontext()
        with torch.inference_mode(), adapter, sdpa_kernel(selected):
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=512)
        ids = out[0, inputs.input_ids.shape[1]:]
        raw = self.processor.decode(ids, skip_special_tokens=True).strip()
        special = self.processor.decode(ids, skip_special_tokens=False)
        return raw, len(ids) >= 512, special, inputs.input_ids.shape[1]


def prompt_for(cfg, support, ex, text, gloss):
    if cfg["condition"] == "chain_gloss":
        return build_chain_prompt(support, cfg["language"], ex["source"], text, cfg["context_kind"] == "image")
    examples = [SimpleNamespace(**e) for e in support]
    if cfg["model"] == "qwen35":
        import run_grammamt_Qwen35_context as prompts
        args = [examples, cfg["language"], ex["source"]]
        if cfg["condition"] == "modelgloss":
            return prompts.build_prompt_model_gloss(*args, gloss, cfg["context_kind"], text)
        return prompts.build_prompt_shot(*args, cfg["context_kind"], text)
    if cfg["context_kind"] == "image":
        if cfg["condition"] == "modelgloss":
            from run_grammamt_Qwen_grammar_images_ModelGloss import build_prompt_grammar_model_gloss
            return build_prompt_grammar_model_gloss(examples, cfg["language"], ex["source"], gloss)
        from run_grammamt_Qwen_grammar_images import build_prompt_grammar_shot
        return build_prompt_grammar_shot(examples, cfg["language"], ex["source"])
    if cfg["condition"] == "modelgloss":
        from run_grammamt_Qwen_grammar_text_ModelGloss import build_prompt_grammar_text_model_gloss
        return build_prompt_grammar_text_model_gloss(examples, cfg["language"], ex["source"], gloss, text)
    from run_grammamt_Qwen_grammar_text import build_prompt_grammar_text_shot
    return build_prompt_grammar_text_shot(examples, cfg["language"], ex["source"], text)


def write_basic_metrics(cfg, rows, target, result, fingerprint):
    import sacrebleu
    validate_records(rows, target, complete=True, fingerprint=fingerprint)
    refs = [r["reference"] for r in rows]
    hyps = [r[cfg["prediction_key"]]["prediction"] for r in rows]
    output = ROOT / cfg["metrics"]
    checksum = sha256(result)
    metrics = json.loads(output.read_text()) if output.exists() else {}
    if metrics.get("results_sha256") == checksum:
        return
    atomic_json(output, {cfg["prediction_key"]:{"bleu":sacrebleu.corpus_bleu(hyps,[refs]).score, "chrf":sacrebleu.corpus_chrf(hyps,[refs],word_order=2).score, "xcomet":None},
                        "records":len(rows), "expected_records":len(target),
                        "invalid_outputs":sum(bool(r[cfg["prediction_key"]].get("error")) for r in rows),
                        "results_sha256":checksum, "fingerprint":fingerprint})


def run(args):
    cfg = json.loads((ROOT / args.config).read_text())
    expected, support = dataset(cfg["language"]), dataset(cfg["language"], "train")[:21]
    paths = [ROOT / p for p in cfg["context_files"]]
    if not paths or any(not p.is_file() for p in paths):
        raise ValueError("Missing context inputs")
    text, images = "", []
    if cfg["context_kind"] == "image":
        from PIL import Image
        images = [Image.open(p).convert("RGB") for p in paths]
    else:
        text = paths[0].read_text(encoding="utf-8").strip()
    provenance = {"config":cfg, "context_hashes":{str(p.relative_to(ROOT)):sha256(p) for p in paths},
                  "test":expected, "support":support, "dtype":"float32", "attention":args.attention,
                  "runner_sha256":sha256(__file__), "io_sha256":sha256(ROOT / "runners/experiment_io.py"),
                  "attention_sha256":sha256(ROOT / "runners/fp32_attention.py")}
    if cfg["condition"] == "chain_gloss":
        provenance["chain_prompt_revision"] = CHAIN_PROMPT_REVISION
    provenance["prompt_module_hashes"] = {str(p.relative_to(ROOT)):sha256(p) for group in ("qwen3", "qwen35", "baseline") for p in (ROOT / "runners" / group).glob("run_grammamt*.py")}
    glosses = []
    if cfg["condition"] == "modelgloss":
        from run_grammamt_ModelGloss import load_glosslm_predictions
        glosses = load_glosslm_predictions(cfg["language"])
        if len(glosses) < len(expected) or any(not g for g in glosses[:len(expected)]):
            raise ValueError("Missing GlossLM predictions")
        provenance["glosses"] = glosses[:len(expected)]
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    result = ROOT / cfg["results"]
    if args.preflight:
        result = ROOT / "results/preflight" / (preflight_stem(cfg, args.attention) + ".jsonl")
    result.parent.mkdir(parents=True, exist_ok=True)
    status_path = result.with_suffix(".status.json")
    with result.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = read_records(result)
        # Preflight uses the longest source sentences, not just the first easy example.
        target = expected if not args.preflight else sorted(expected, key=lambda x:len(x["source"]), reverse=True)[:3]
        validate_records(rows, target, fingerprint=fingerprint)
        if len(rows) == len(target):
            if not args.preflight:
                write_basic_metrics(cfg, rows, target, result, fingerprint)
            if any(r[cfg["prediction_key"]].get("error") for r in rows):
                raise RuntimeError("Completed records contain invalid outputs; do not treat this as success")
            print(f"[SKIP] Complete verified output: {result}")
            return
        atomic_json(result.with_suffix(".provenance.json"), provenance)
        started = time.time()
        try:
            backend = Backend(cfg["model_id"], cfg["model"], args.attention)
            print(f"[INFO] model={cfg['model_id']} dtype=float32 gpu_count=1 images={len(images)} attention={args.attention}", flush=True)
            with result.open("a", encoding="utf-8") as f:
                for i in range(len(rows), len(target)):
                    ex = target[i]
                    original_idx = expected.index(ex)
                    gloss = glosses[original_idx] if glosses else ""
                    system, user = prompt_for(cfg, support, ex, text, gloss)
                    attempts, generated_gloss, prediction, error = [], "", "", None
                    for attempt in range(2):
                        raw, truncated, special, tokens = backend.generate(system, user, images)
                        attempts.append({"raw":raw, "raw_with_special_tokens":special, "truncated":truncated, "input_tokens":tokens})
                        try:
                            if truncated:
                                raise ValueError("truncated")
                            if "<think>" in special.lower() or "</think>" in special.lower():
                                raise ValueError("reasoning_violation")
                            if cfg["condition"] == "chain_gloss":
                                generated_gloss, prediction = parse_chain(raw)
                            elif cfg["model"] == "qwen35":
                                final = re.fullmatch(r"\s*FINAL_TRANSLATION:\s*([^\n]+)\s*", raw)
                                if not final:
                                    raise ValueError("invalid_translation_format")
                                prediction = final[1].strip()
                            else:
                                from run_grammamt_Qwen_grammar_images import extract_translation
                                prediction = extract_translation(raw)
                            if not prediction.strip():
                                raise ValueError("empty_output")
                            error = None
                            break
                        except ValueError as exc:
                            error = str(exc)
                            user += "\nFollow the requested output format exactly. Do not add analysis."
                    if error:
                        prediction = ""
                    record = {"idx":i, "source":ex["source"], "reference":ex["reference"], "fingerprint":fingerprint,
                              cfg["prediction_key"]:{"prediction":prediction, "generated_gloss":generated_gloss, "raw":attempts[-1]["raw"], "attempts":attempts, "error":error}}
                    if gloss:
                        record[cfg["prediction_key"]]["glosslm_pred_gloss"] = gloss
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                    rows.append(record)
                    print(f"[PROGRESS] {i+1}/{len(target)} elapsed={time.time()-started:.1f}s", flush=True)
            validate_records(rows, target, complete=True, fingerprint=fingerprint)
            failures = sum(bool(r[cfg["prediction_key"]]["error"]) for r in rows)
            atomic_json(status_path, {"status":"complete" if not failures else "complete_with_errors", "records":len(rows), "expected":len(target), "invalid_outputs":failures,
                                      "peak_gpu_bytes":backend.torch.cuda.max_memory_allocated(), "runtime_seconds":time.time()-started, "fingerprint":fingerprint})
            if not args.preflight:
                write_basic_metrics(cfg, rows, target, result, fingerprint)
            if failures:
                raise RuntimeError(f"{failures} invalid outputs; inspect status before reporting")
        except Exception as exc:
            if not status_path.exists() or len(rows) != len(target):
                atomic_json(status_path, {"status":"failed", "error":str(exc), "records":len(rows), "expected":len(target), "fingerprint":fingerprint})
            raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--attention", choices=[ATTENTION_TAG, "efficient", "math"], default=ATTENTION_TAG)
    run(p.parse_args())
