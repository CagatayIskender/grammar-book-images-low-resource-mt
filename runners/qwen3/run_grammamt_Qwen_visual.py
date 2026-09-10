from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_LAYOUT_ROOT = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _layout_dir in ("", "runners/baseline", "runners/qwen3", "runners/qwen35", "runners/openrouter", "scripts/generators"):
    _layout_sys.path.insert(0, str(_LAYOUT_ROOT / _layout_dir))


import argparse
import json
import os
import re
import textwrap
import time
from dataclasses import dataclass
from typing import List

import torch
from PIL import Image, ImageDraw, ImageFont
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


def build_prompt_visual_gloss_shot(support, src_lang):
    header = (
        f"Here are some examples of {src_lang} sentences and their corresponding English translations:\n"
    )
    shots = "".join(build_example_block(src_lang, e) for e in support)
    tail = (
        f"The attached image contains one {src_lang} sentence.\n"
        f"Please translate the sentence in the image into English.\n"
        f"Answer with the translation directly and enclose your translation in ###."
    )
    return SYSTEM_MSG, header + shots + tail


def build_prompt_visual_chain_gloss(support, src_lang):
    header = (
        f"Here are some examples of {src_lang} sentences and their corresponding English translations:\n"
    )
    shots = "".join(build_example_block(src_lang, e) for e in support)
    tail = (
        f"The attached image contains one {src_lang} sentence.\n"
        f"First infer an approximate gloss for the sentence in the image, then give the English translation.\n"
        f"Enclose only the final translation in ###."
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


def get_font(font_size: int):
    for candidate in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(candidate, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def render_sentence_image(src_lang: str, sentence: str, out_path: str, width: int = 1200, font_size: int = 32):
    font = get_font(font_size)
    margin = 50
    line_gap = 12
    title = f"{src_lang} sentence"
    wrapped_lines = textwrap.wrap(sentence, width=42, break_long_words=False, break_on_hyphens=False) or [sentence]

    probe = Image.new("RGB", (width, 1000), "white")
    draw = ImageDraw.Draw(probe)
    title_box = draw.textbbox((0, 0), title, font=font)
    line_heights = []
    for line in wrapped_lines:
        box = draw.textbbox((0, 0), line, font=font)
        line_heights.append(box[3] - box[1])

    title_height = title_box[3] - title_box[1]
    content_height = sum(line_heights) + max(0, len(wrapped_lines) - 1) * line_gap
    height = margin * 2 + title_height + 25 + content_height

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    y = margin
    draw.text((margin, y), title, fill="black", font=font)
    y += title_height + 25
    for line, line_height in zip(wrapped_lines, line_heights):
        draw.text((margin, y), line, fill="black", font=font)
        y += line_height + line_gap

    image.save(out_path)


class QwenVisualRunner:
    def __init__(self, model_id: str, use_4bit=False, use_float32=False):
        self.model_id = model_id
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

        print(f"[DEBUG] Visual model loaded on device: {self.model.device}")

    def generate(self, system: str, user: str, image: Image.Image, max_new_tokens: int):
        t0 = time.time()
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": user},
                ],
            },
        ]
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=[text_prompt], images=[image], return_tensors="pt").to(self.model.device)
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
    ap.add_argument("--src_lang", default=None)
    ap.add_argument("--out_jsonl", default="results.jsonl")
    ap.add_argument("--out_metrics", default="metrics.json")
    ap.add_argument("--image_dir", default=None)
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

    covered = parse_gitdev_igt_file(covered)
    uncovered = parse_gitdev_igt_file(uncovered)
    support = covered[: args.support_n]
    test = uncovered[: args.test_n]

    image_dir = args.image_dir or os.path.join(PROJECT_ROOT, "rendered_visual_inputs", lang.lower())
    os.makedirs(image_dir, exist_ok=True)

    runner = QwenVisualRunner(args.model_id, use_4bit=args.use_4bit, use_float32=args.use_float32)
    comet_model, comet_name = None, None
    print("[INFO] XCOMET must run in the separate scoring job")
    if comet_name is None:
        print("[INFO] COMET model: unavailable\n")
    else:
        print(f"[INFO] COMET model: {comet_name}\n")

    hyps_vgs, hyps_vcg = [], []
    refs, sources = [], []

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for i, ex in enumerate(test):
            print(f"[DEBUG] Running example {i}")
            refs.append(ex.reference)
            sources.append(ex.source)

            image_path = os.path.join(image_dir, f"{lang.lower()}_{i:04d}.png")
            render_sentence_image(args.src_lang, ex.source, image_path)
            image = Image.open(image_path).convert("RGB")

            sys1, usr1 = build_prompt_visual_gloss_shot(support, args.src_lang)
            raw1, _ = runner.generate(sys1, usr1, image, args.max_new_tokens)
            pred1 = extract_translation(raw1)
            hyps_vgs.append(pred1)

            sys2, usr2 = build_prompt_visual_chain_gloss(support, args.src_lang)
            raw2, _ = runner.generate(sys2, usr2, image, args.max_new_tokens)
            pred2 = extract_translation(raw2)
            hyps_vcg.append(pred2)

            rec = {
                "idx": i,
                "source": ex.source,
                "reference": ex.reference,
                "image_path": image_path,
                "visual_gloss_shot": {"prediction": pred1, "raw": raw1},
                "visual_chain_gloss": {"prediction": pred2, "raw": raw2},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    metrics = {
        "visual_gloss_shot": compute_metrics(refs, hyps_vgs, sources, comet_model),
        "visual_chain_gloss": compute_metrics(refs, hyps_vcg, sources, comet_model),
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
    print("VISUAL-GLOSS-SHOT")
    print(f"BLEU   : {metrics['visual_gloss_shot']['bleu']:.2f}")
    print(f"chrF++ : {metrics['visual_gloss_shot']['chrf']:.2f}")
    if metrics["visual_gloss_shot"]["xcomet"] is None:
        print("XCOMET : unavailable\n")
    else:
        print(f"XCOMET : {metrics['visual_gloss_shot']['xcomet']:.4f}\n")

    print("VISUAL-CHAIN-GLOSS")
    print(f"BLEU   : {metrics['visual_chain_gloss']['bleu']:.2f}")
    print(f"chrF++ : {metrics['visual_chain_gloss']['chrf']:.2f}")
    if metrics["visual_chain_gloss"]["xcomet"] is None:
        print("XCOMET : unavailable")
    else:
        print(f"XCOMET : {metrics['visual_chain_gloss']['xcomet']:.4f}")

    hours = metrics["runtime_hms"]["hours"]
    minutes = metrics["runtime_hms"]["minutes"]
    seconds = metrics["runtime_hms"]["seconds"]
    print(f"\nTotal execution time: {hours}h {minutes}m {seconds}s")
    print("\n========================================\n")


if __name__ == "__main__":
    main()
