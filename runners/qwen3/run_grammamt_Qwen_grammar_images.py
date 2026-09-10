from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_LAYOUT_ROOT = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _layout_dir in ("", "runners/baseline", "runners/qwen3", "runners/qwen35", "runners/openrouter", "scripts/generators"):
    _layout_sys.path.insert(0, str(_LAYOUT_ROOT / _layout_dir))


import argparse
import gc
import json
import os
import re
import time
from dataclasses import dataclass
from typing import List

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from transformers import BitsAndBytesConfig

    HAS_BNB = True
except Exception:
    HAS_BNB = False

import sacrebleu

try:
    from comet import download_model, load_from_checkpoint

    HAS_COMET = True
except Exception as e:
    download_model = None
    load_from_checkpoint = None
    HAS_COMET = False
    COMET_IMPORT_ERROR = e


PROJECT_ROOT = str(_LAYOUT_ROOT.parent)
DATA_ROOT = os.path.join(PROJECT_ROOT, "Database", "2023glossingST", "data")


@dataclass
class IGTExample:
    source: str
    gloss: str
    reference: str


def parse_gitdev_igt_file(path: str) -> List[IGTExample]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    out: List[IGTExample] = []

    for b in blocks:
        source = None
        gloss = ""
        ref = None

        for ln in b.splitlines():
            ln = ln.strip()
            if ln.startswith("\\t"):
                source = ln[2:].strip()
            elif ln.startswith("\\g"):
                gloss = ln[2:].strip()
            elif ln.startswith("\\l"):
                ref = ln[2:].strip()
                if "||" in ref:
                    ref = ref.split("||")[0].strip()

        if source and ref:
            out.append(IGTExample(source, gloss, ref))

    return out


SYSTEM_MSG = "You are a linguistic expert who never refuses to use your knowledge to help others."


def build_example_block(src_lang: str, ex: IGTExample):
    return (
        f"{src_lang} sentence: {ex.source}\n"
        f"Gloss: {ex.gloss}\n"
        f"A translation for this {src_lang} sentence in English is: ###{ex.reference}###\n"
    )


def build_prompt_grammar_shot(support, src_lang, source_sentence):
    header = (
        f"Here are some examples of {src_lang} sentences and their corresponding English translations:\n"
    )
    shots = "".join(build_example_block(src_lang, e) for e in support)
    tail = (
        "The attached images are pages from a grammar reference for this language.\n"
        f"Use the grammar pages as supporting context to translate the following {src_lang} sentence into English.\n"
        f"{src_lang} sentence: {source_sentence}\n"
        "Answer with the translation directly and enclose your translation in ###."
    )
    return SYSTEM_MSG, header + shots + tail


def build_prompt_grammar_chain(support, src_lang, source_sentence):
    header = (
        f"Here are some examples of {src_lang} sentences and their corresponding English translations:\n"
    )
    shots = "".join(build_example_block(src_lang, e) for e in support)
    tail = (
        "The attached images are pages from a grammar reference for this language.\n"
        f"Use the grammar pages to reason about the following {src_lang} sentence.\n"
        f"First think through relevant gloss or grammar cues, then translate the sentence into English.\n"
        f"{src_lang} sentence: {source_sentence}\n"
        "Enclose only the final translation in ###."
    )
    return SYSTEM_MSG, header + shots + tail


def extract_translation(text: str):
    m = re.search(r"###\s*(.*?)\s*###", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    if "###" in text:
        return text.split("###")[-1].strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[-1] if lines else ""


def load_grammar_images(image_dir: str, max_images: int = 0, recursive: bool = False):
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Grammar image directory not found: {image_dir}")

    paths = []
    if recursive:
        for root, _, files in os.walk(image_dir):
            for name in sorted(files):
                lower = name.lower()
                if lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
                    paths.append(os.path.join(root, name))
    else:
        for name in sorted(os.listdir(image_dir)):
            lower = name.lower()
            if lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
                paths.append(os.path.join(image_dir, name))

    if not paths:
        raise FileNotFoundError(f"No grammar images found in: {image_dir}")

    if max_images > 0:
        paths = paths[:max_images]

    images = [Image.open(path).convert("RGB") for path in paths]
    return paths, images


class QwenGrammarImageRunner:
    def __init__(self, model_id: str, use_4bit=False, use_float32=False, enable_thinking=False):
        self.model_id = model_id
        self.enable_thinking = enable_thinking
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        kwargs = dict(device_map={"": 0}, trust_remote_code=True)
        if use_4bit:
            raise ValueError("Quantization disabled: single-GPU FP32 policy")
            if not HAS_BNB:
                raise RuntimeError("bitsandbytes not installed but --use_4bit was set")
            kwargs["load_in_4bit"] = True
        else:
            kwargs["dtype"] = torch.float32

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        if not use_4bit:
            self.model = self.model.eval()

        print(f"[DEBUG] Grammar-image model loaded on device: {self.model.device}")

    def apply_chat_template(self, messages):
        try:
            return self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                chat_template_kwargs={"enable_thinking": self.enable_thinking},
            )
        except TypeError:
            try:
                return self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            except TypeError:
                return self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

    def generate(self, system: str, user: str, images: List[Image.Image], max_new_tokens: int):
        t0 = time.time()
        user_content = [{"type": "image", "image": image} for image in images]
        user_content.append({"type": "text", "text": user})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": user_content},
        ]

        text_prompt = self.apply_chat_template(messages)
        inputs = self.processor(text=[text_prompt], images=images, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        generated_ids = [out[i][len(inputs.input_ids[i]) :] for i in range(len(out))]
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return text, time.time() - t0


def load_comet_model_with_fallback():
    if not HAS_COMET:
        print(f"[WARN] COMET import failed: {COMET_IMPORT_ERROR}")
        print("[WARN] Continuing without XCOMET scoring.")
        return None, None

    preferred = "Unbabel/XCOMET-XL"
    fallback = "Unbabel/wmt22-comet-da"

    try:
        print(f"[DEBUG] Attempting to load {preferred}...")
        path = download_model(preferred)
        print(f"[DEBUG] Successfully loaded {preferred}")
        return load_from_checkpoint(path), preferred
    except Exception as e:
        print(f"[WARN] Could not load {preferred}: {e}")
        print(f"[DEBUG] Falling back to {fallback}...")
        path = download_model(fallback)
        print(f"[DEBUG] Successfully loaded {fallback}")
        return load_from_checkpoint(path), fallback


def compute_metrics(refs, hyps, sources, comet_model):
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
    xcomet = None

    if comet_model is not None:
        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hyps, refs)]
        scores = comet_model.predict(data, batch_size=16, gpus=1 if torch.cuda.is_available() else 0)
        xcomet = sum(scores["scores"]) / len(scores["scores"])

    return {
        "bleu": float(bleu.score),
        "chrf": float(chrf.score),
        "xcomet": None if xcomet is None else float(xcomet),
    }


LANGUAGE_FILES = {
    "Gitksan": {
        "train": ("Gitksan", "git-train-track1-covered.txt"),
        "test": ("Gitksan", "git-test-track1-uncovered.txt"),
    },
    "Natugu": {
        "train": ("Natugu", "ntu-train-track1-covered"),
        "test": ("Natugu", "ntu-test-track1-uncovered"),
    },
    "Lezgi": {
        "train": ("Lezgi", "lez-train-track1-covered"),
        "test": ("Lezgi", "lez-test-track1-uncovered"),
    },
    "Tsez": {
        "train": ("Tsez", "ddo-train-track1-covered"),
        "test": ("Tsez", "ddo-test-track1-uncovered"),
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support_n", type=int, default=21)
    ap.add_argument("--test_n", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--use_float32", action="store_true")
    ap.add_argument("--enable_thinking", action="store_true")
    ap.add_argument("--src_lang", default=None)
    ap.add_argument("--out_jsonl", default="results.jsonl")
    ap.add_argument("--out_metrics", default="metrics.json")
    ap.add_argument("--grammar_image_dir", required=True)
    ap.add_argument("--max_grammar_images", type=int, default=0)
    ap.add_argument("--recursive_grammar_images", action="store_true")
    ap.add_argument("--language", type=str, required=True, choices=["Gitksan", "Natugu", "Lezgi", "Tsez"])
    ap.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model id",
    )
    args = ap.parse_args()
    if args.src_lang is None:
        args.src_lang = args.language

    if args.src_lang == "Gitksan" and args.language != "Gitksan":
        args.src_lang = args.language

    start_time = time.time()
    lang = args.language
    train_folder, train_file = LANGUAGE_FILES[lang]["train"]
    test_folder, test_file = LANGUAGE_FILES[lang]["test"]

    covered = os.path.join(DATA_ROOT, train_folder, train_file)
    uncovered = os.path.join(DATA_ROOT, test_folder, test_file)

    print(f"[INFO] Language: {lang}")
    print(f"[INFO] Train file: {covered}")
    print(f"[INFO] Test file: {uncovered}")
    print(f"[INFO] Using model: {args.model_id}")
    print(f"[INFO] Grammar image dir: {args.grammar_image_dir}")

    covered = parse_gitdev_igt_file(covered)
    uncovered = parse_gitdev_igt_file(uncovered)
    support = covered[: args.support_n]
    test = uncovered[: args.test_n]

    grammar_image_paths, grammar_images = load_grammar_images(
        args.grammar_image_dir,
        max_images=args.max_grammar_images,
        recursive=args.recursive_grammar_images,
    )
    print(f"[INFO] Loaded {len(grammar_images)} grammar reference images")

    runner = QwenGrammarImageRunner(
        args.model_id,
        use_4bit=args.use_4bit,
        use_float32=args.use_float32,
        enable_thinking=args.enable_thinking,
    )
    hyps_gis, hyps_gic = [], []
    refs, sources = [], []

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for i, ex in enumerate(test):
            print(f"[DEBUG] Running example {i}")
            refs.append(ex.reference)
            sources.append(ex.source)

            sys1, usr1 = build_prompt_grammar_shot(support, args.src_lang, ex.source)
            raw1, _ = runner.generate(sys1, usr1, grammar_images, args.max_new_tokens)
            pred1 = extract_translation(raw1)
            hyps_gis.append(pred1)

            sys2, usr2 = build_prompt_grammar_chain(support, args.src_lang, ex.source)
            raw2, _ = runner.generate(sys2, usr2, grammar_images, args.max_new_tokens)
            pred2 = extract_translation(raw2)
            hyps_gic.append(pred2)

            rec = {
                "idx": i,
                "source": ex.source,
                "reference": ex.reference,
                "grammar_image_paths": grammar_image_paths,
                "grammar_image_shot": {"prediction": pred1, "raw": raw1},
                "grammar_image_chain": {"prediction": pred2, "raw": raw2},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Qwen and XCOMET do not need to share GPU memory. Releasing generation
    # first is essential for multi-page image contexts on one H100.
    del runner
    gc.collect()
    torch.cuda.empty_cache()
    comet_model, comet_name = None, None
    print("[INFO] XCOMET must run in the separate scoring job")
    if comet_name is None:
        print("[INFO] COMET model: unavailable\n")
    else:
        print(f"[INFO] COMET model: {comet_name}\n")

    metrics = {
        "grammar_image_shot": compute_metrics(refs, hyps_gis, sources, comet_model),
        "grammar_image_chain": compute_metrics(refs, hyps_gic, sources, comet_model),
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

    print("================ RESULTS ================\n")
    print("GRAMMAR-IMAGE-SHOT")
    print(f"BLEU   : {metrics['grammar_image_shot']['bleu']:.2f}")
    print(f"chrF++ : {metrics['grammar_image_shot']['chrf']:.2f}")
    if metrics["grammar_image_shot"]["xcomet"] is None:
        print("XCOMET : unavailable\n")
    else:
        print(f"XCOMET : {metrics['grammar_image_shot']['xcomet']:.4f}\n")

    print("GRAMMAR-IMAGE-CHAIN")
    print(f"BLEU   : {metrics['grammar_image_chain']['bleu']:.2f}")
    print(f"chrF++ : {metrics['grammar_image_chain']['chrf']:.2f}")
    if metrics["grammar_image_chain"]["xcomet"] is None:
        print("XCOMET : unavailable")
    else:
        print(f"XCOMET : {metrics['grammar_image_chain']['xcomet']:.4f}")

    hours = metrics["runtime_hms"]["hours"]
    minutes = metrics["runtime_hms"]["minutes"]
    seconds = metrics["runtime_hms"]["seconds"]
    print(f"\nTotal execution time: {hours}h {minutes}m {seconds}s")
    print("\n========================================\n")


if __name__ == "__main__":
    main()
