from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import List

import torch
import sacrebleu
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

BASE_VENV_PYTHON = os.path.join(PROJECT_DIR, "venv", "bin", "python")
XCOMET_SCORER = os.path.join(os.path.dirname(__file__), "score_qwen35_xcomet_from_jsonl.py")

from run_grammamt_ModelGloss import load_glosslm_predictions
from run_grammamt_Qwen import (  # noqa: E402
    DATA_ROOT,
    LANGUAGE_FILES,
    parse_gitdev_igt_file,
)


def load_context_images(image_dir: str) -> tuple[list[str], list[Image.Image]]:
    if not image_dir:
        return [], []
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Grammar image directory not found: {image_dir}")
    paths = []
    for name in sorted(os.listdir(image_dir)):
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            paths.append(os.path.join(image_dir, name))
    if not paths:
        raise FileNotFoundError(f"No grammar images found in: {image_dir}")
    return paths, [Image.open(path).convert("RGB") for path in paths]


def load_context_text(path: str) -> str:
    if not path:
        return ""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Grammar text file not found: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def compute_basic_metrics(refs, hyps):
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
    return {
        "bleu": float(bleu.score),
        "chrf": float(chrf.score),
        "xcomet": None,
    }


DIRECT_SYSTEM_MSG = (
    "You are a direct translation engine. "
    "Return only the requested English translation in the requested output format."
)

FINAL_PREFIX = "FINAL_TRANSLATION:"
DISABLED_THINK_BLOCK = "<think>\n\n</think>\n\n"

NO_THINK_INSTRUCTION = (
    f"Return exactly one line in this format: {FINAL_PREFIX} <English translation>\n"
    "Do not include notes, glosses, bullet points, markdown, or extra text."
)

REASONING_MARKERS = (
    "thinking process",
    "analysis:",
    "analyze the",
    "breakdown:",
    "constraints:",
    "task:",
    "step 1",
    "let's",
    "reasoning",
)


def build_qwen35_example_block(src_lang: str, ex) -> str:
    return (
        f"{src_lang} sentence: {ex.source}\n"
        f"Gloss: {ex.gloss}\n"
        f"English translation: {ex.reference}\n\n"
    )


def strip_thinking_text(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"(?is)^thinking process:\s*", "", cleaned)
    cleaned = re.sub(
        r"(?is)^.*?(?:final translation|translation|answer)\s*:\s*",
        "",
        cleaned,
    )
    return cleaned.strip()


def looks_like_reasoning_output(text: str) -> bool:
    head = text.strip()[:800].lower()
    if not head:
        return True
    if any(marker in head for marker in REASONING_MARKERS):
        return True
    return bool(re.match(r"^\s*(?:\d+\.|\*|-)\s+", text))


def extract_qwen35_translation(text: str) -> str:
    final_matches = re.findall(
        rf"(?im)^\s*{re.escape(FINAL_PREFIX)}\s*(.+?)\s*$",
        text,
    )
    final_matches = [
        m.strip().strip("`")
        for m in final_matches
        if m.strip()
        and "<English translation>" not in m
        and "requested output format" not in m.lower()
    ]
    if final_matches:
        return final_matches[-1]

    # Backward-compatible parsing for older result files. Ignore delimiter
    # mentions copied from instructions such as "starting with `###` and ending..."
    matches = [
        m.strip()
        for m in re.findall(r"###\s*(.*?)\s*###", text, flags=re.DOTALL)
        if m.strip() and "ending with" not in m.lower() and "starting with" not in m.lower()
    ]
    if matches:
        return matches[-1]
    text = strip_thinking_text(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for prefix in (FINAL_PREFIX, "Final translation:", "Translation:", "Answer:", "English translation:"):
        for line in reversed(lines):
            if line.lower().startswith(prefix.lower()):
                candidate = line[len(prefix) :].strip().strip("`")
                if candidate and "<English translation>" not in candidate:
                    return candidate
    analysis_markers = (
        *REASONING_MARKERS,
        "analyze the",
        "breakdown",
        "constraints",
        "input:",
    )
    translation_like = [
        line
        for line in lines
        if not any(marker in line.lower() for marker in analysis_markers)
        and not line.startswith(("*", "-", "1.", "2.", "3."))
    ]
    if translation_like:
        return translation_like[-1].strip(" `")
    return lines[-1] if lines else ""


def rescore_xcomet_from_jsonl(result_jsonl: str, metrics_json: str):
    if not os.path.exists(BASE_VENV_PYTHON):
        raise FileNotFoundError(f"Base venv Python not found: {BASE_VENV_PYTHON}")
    if not os.path.exists(XCOMET_SCORER):
        raise FileNotFoundError(f"XCOMET scorer not found: {XCOMET_SCORER}")
    metrics_dir = os.path.dirname(metrics_json) or "."
    cmd = [
        BASE_VENV_PYTHON,
        XCOMET_SCORER,
        "--result_jsonl",
        result_jsonl,
        "--metrics_dir",
        metrics_dir,
        "--overwrite",
    ]
    print("[INFO] Rescoring with XCOMET using base venv:")
    print("[INFO] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_context_intro(src_lang: str, context_kind: str, grammar_text: str) -> str:
    if context_kind == "text":
        return (
            f"\nHere is a grammar reference summary for {src_lang}. "
            f"Use it as supporting context:\n{grammar_text}\n"
        )
    return "\nThe attached images are pages from a grammar reference for this language.\n"


def build_prompt_shot(support, src_lang, source_sentence, context_kind, grammar_text):
    header = f"Here are some examples of {src_lang} sentences and their corresponding English translations:\n"
    shots = "".join(build_qwen35_example_block(src_lang, e) for e in support)
    context = build_context_intro(src_lang, context_kind, grammar_text)
    tail = (
        f"\nUse the grammar reference as supporting context.\n"
        f"{NO_THINK_INSTRUCTION}\n\n"
        f"{src_lang} sentence: {source_sentence}\n"
        f"{FINAL_PREFIX}"
    )
    return DIRECT_SYSTEM_MSG, header + shots + context + tail


def build_prompt_chain(support, src_lang, source_sentence, context_kind, grammar_text):
    header = f"Here are some examples of {src_lang} sentences and their corresponding English translations:\n"
    shots = "".join(build_qwen35_example_block(src_lang, e) for e in support)
    context = build_context_intro(src_lang, context_kind, grammar_text)
    tail = (
        f"\nUse the grammar reference as supporting context.\n"
        f"{NO_THINK_INSTRUCTION}\n\n"
        f"{src_lang} sentence: {source_sentence}\n"
        f"{FINAL_PREFIX}"
    )
    return DIRECT_SYSTEM_MSG, header + shots + context + tail


def build_prompt_model_gloss(support, src_lang, source_sentence, predicted_gloss, context_kind, grammar_text):
    header = f"Here are some examples of {src_lang} sentences and their corresponding English translations:\n"
    shots = "".join(build_qwen35_example_block(src_lang, e) for e in support)
    context = build_context_intro(src_lang, context_kind, grammar_text)
    tail = (
        f"\nUse the grammar reference and predicted gloss as supporting context.\n"
        f"{NO_THINK_INSTRUCTION}\n\n"
        f"{src_lang} sentence: {source_sentence}\n"
        f"Gloss: {predicted_gloss}\n"
        f"{FINAL_PREFIX}"
    )
    return DIRECT_SYSTEM_MSG, header + shots + context + tail


class Qwen35ContextRunner:
    def __init__(self, model_id: str, use_float32: bool = False):
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.bad_words_ids = self._build_bad_words_ids()
        dtype = torch.float32 if use_float32 else torch.bfloat16
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            dtype=dtype,
            trust_remote_code=True,
        ).eval()

    def _build_bad_words_ids(self) -> list[list[int]] | None:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            return None
        banned = [
            "Thinking",
            "thinking",
            "Analysis",
            "analysis",
            "Breakdown",
            "breakdown",
            "Reasoning",
            "reasoning",
        ]
        bad_words_ids = []
        for word in banned:
            ids = tokenizer.encode(word, add_special_tokens=False)
            if ids:
                bad_words_ids.append(ids)
        return bad_words_ids or None

    def _device(self):
        try:
            return self.model.device
        except AttributeError:
            return next(self.model.parameters()).device

    def apply_chat_template(self, messages):
        errors = []
        attempts = [
            ("enable_thinking", {"enable_thinking": False}),
            ("chat_template_kwargs", {"chat_template_kwargs": {"enable_thinking": False}}),
        ]
        for label, kwargs in attempts:
            try:
                prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **kwargs,
                )
            except TypeError as exc:
                errors.append(f"{label}: {exc}")
                continue

            if DISABLED_THINK_BLOCK in prompt:
                return prompt
            errors.append(f"{label}: rendered prompt did not contain the disabled thinking block")

        raise RuntimeError(
            "Qwen 3.5 thinking was not disabled by the chat template. "
            "Refusing to run because this would produce unusable metrics. "
            + " | ".join(errors)
        )

    def _generate_once(self, system: str, user: str, images: List[Image.Image], max_new_tokens: int):
        user_content = [{"type": "image", "image": image} for image in images]
        user_content.append({"type": "text", "text": user})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": user_content},
        ]
        prompt = self.apply_chat_template(messages)
        kwargs = {"text": [prompt], "return_tensors": "pt"}
        if images:
            kwargs["images"] = images
        inputs = self.processor(**kwargs).to(self._device())
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        if self.bad_words_ids:
            generate_kwargs["bad_words_ids"] = self.bad_words_ids
        with torch.no_grad():
            out = self.model.generate(**inputs, **generate_kwargs)
        generated_ids = [out[i][len(inputs.input_ids[i]) :] for i in range(len(out))]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    def generate(self, system: str, user: str, images: List[Image.Image], max_new_tokens: int):
        raw = self._generate_once(system, user, images, max_new_tokens)
        pred = extract_qwen35_translation(raw)
        if pred and not looks_like_reasoning_output(raw):
            return raw

        print("[WARN] Qwen 3.5 produced reasoning-like output; retrying with stricter final-only prompt.")
        retry_user = (
            user
            + "\n\n"
            f"Return one line only. No headings. No explanation.\n{FINAL_PREFIX}"
        )
        return self._generate_once(system, retry_user, images, max_new_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True, choices=["Gitksan", "Natugu", "Lezgi", "Tsez"])
    ap.add_argument("--model_id", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--support_n", type=int, default=21)
    ap.add_argument("--test_n", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--use_float32", action="store_true")
    ap.add_argument("--src_lang", default="Gitksan")
    ap.add_argument("--grammar_image_dir", default="")
    ap.add_argument("--grammar_text_file", default="")
    ap.add_argument("--model_gloss", action="store_true")
    ap.add_argument("--glosslm_preds_csv", default=None)
    ap.add_argument("--out_jsonl", default="results.jsonl")
    ap.add_argument("--out_metrics", default="metrics.json")
    ap.add_argument("--score_xcomet_after_generation", action="store_true")
    args = ap.parse_args()

    if bool(args.grammar_image_dir) == bool(args.grammar_text_file):
        raise ValueError("Set exactly one of --grammar_image_dir or --grammar_text_file")
    if args.src_lang == "Gitksan" and args.language != "Gitksan":
        args.src_lang = args.language

    context_kind = "image" if args.grammar_image_dir else "text"
    image_paths, images = load_context_images(args.grammar_image_dir)
    grammar_text = load_context_text(args.grammar_text_file)

    train_folder, train_file = LANGUAGE_FILES[args.language]["train"]
    test_folder, test_file = LANGUAGE_FILES[args.language]["test"]
    covered = parse_gitdev_igt_file(os.path.join(DATA_ROOT, train_folder, train_file))
    uncovered = parse_gitdev_igt_file(os.path.join(DATA_ROOT, test_folder, test_file))
    support = covered[: args.support_n]
    test = uncovered[: args.test_n]

    glosslm_predictions = []
    if args.model_gloss:
        glosslm_predictions = load_glosslm_predictions(args.language, args.glosslm_preds_csv)
        if len(glosslm_predictions) < len(test):
            raise ValueError(f"Only {len(glosslm_predictions)} GlossLM predictions available, but test_n={len(test)}")

    print(f"[INFO] Language: {args.language}")
    print(f"[INFO] Using model: {args.model_id}")
    print(f"[INFO] Context kind: {context_kind}")
    if context_kind == "image":
        print(f"[INFO] Loaded {len(images)} grammar images")
    else:
        print(f"[INFO] Loaded grammar text: {len(grammar_text)} characters")

    start_time = time.time()
    runner = Qwen35ContextRunner(args.model_id, use_float32=args.use_float32)
    comet_name = None
    print("[WARN] Skipping XCOMET inside the Qwen3.5 environment; rescore JSONL outputs separately if XCOMET is needed.")

    refs, sources = [], []
    hyps_a, hyps_b = [], []

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for i, ex in enumerate(test):
            print(f"[DEBUG] Running example {i}")
            refs.append(ex.reference)
            sources.append(ex.source)
            if args.model_gloss:
                system, user = build_prompt_model_gloss(
                    support, args.src_lang, ex.source, glosslm_predictions[i], context_kind, grammar_text
                )
                raw = runner.generate(system, user, images, args.max_new_tokens)
                pred = extract_qwen35_translation(raw)
                hyps_a.append(pred)
                rec = {
                    "idx": i,
                    "source": ex.source,
                    "reference": ex.reference,
                    "context_kind": context_kind,
                    "grammar_image_paths": image_paths,
                    "grammar_text_file": args.grammar_text_file,
                    "qwen35_context_model_gloss": {
                        "prediction": pred,
                        "raw": raw,
                        "glosslm_pred_gloss": glosslm_predictions[i],
                    },
                }
            else:
                sys1, usr1 = build_prompt_shot(support, args.src_lang, ex.source, context_kind, grammar_text)
                raw1 = runner.generate(sys1, usr1, images, args.max_new_tokens)
                pred1 = extract_qwen35_translation(raw1)
                hyps_a.append(pred1)
                sys2, usr2 = build_prompt_chain(support, args.src_lang, ex.source, context_kind, grammar_text)
                raw2 = runner.generate(sys2, usr2, images, args.max_new_tokens)
                pred2 = extract_qwen35_translation(raw2)
                hyps_b.append(pred2)
                rec = {
                    "idx": i,
                    "source": ex.source,
                    "reference": ex.reference,
                    "context_kind": context_kind,
                    "grammar_image_paths": image_paths,
                    "grammar_text_file": args.grammar_text_file,
                    "qwen35_context_shot": {"prediction": pred1, "raw": raw1},
                    "qwen35_context_chain": {"prediction": pred2, "raw": raw2},
                }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.model_gloss:
        metrics = {
            "qwen35_context_model_gloss": compute_basic_metrics(refs, hyps_a),
            "comet_model_used": comet_name,
        }
    else:
        metrics = {
            "qwen35_context_shot": compute_basic_metrics(refs, hyps_a),
            "qwen35_context_chain": compute_basic_metrics(refs, hyps_b),
            "comet_model_used": comet_name,
        }

    elapsed_time = time.time() - start_time
    metrics["runtime_seconds"] = elapsed_time
    metrics["runtime_hms"] = {
        "hours": int(elapsed_time // 3600),
        "minutes": int((elapsed_time % 3600) // 60),
        "seconds": int(elapsed_time % 60),
    }
    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.score_xcomet_after_generation:
        rescore_xcomet_from_jsonl(args.out_jsonl, args.out_metrics)


if __name__ == "__main__":
    main()
