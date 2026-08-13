from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from typing import List, Dict

import torch
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "Database", "2023glossingST", "data")
from transformers import AutoTokenizer, AutoModelForCausalLM

# optional 4bit
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

def build_prompt_gloss_shot(support, src_lang, source_sentence):
    header = f"Here are some examples of {src_lang} sentences and their corresponding English translations:\n"
    shots = "".join(build_example_block(src_lang, e) for e in support)

    tail = (
        f"Please help me translate the following sentence from {src_lang} to English. "
        f"Please answer with the translation directly and enclose your translation in ###.\n"
        f"{src_lang} sentence: {source_sentence}\n"
        f"A translation for this {src_lang} sentence in English is:###"
    )

    return SYSTEM_MSG, header + shots + tail


def build_prompt_chain_gloss(support, src_lang, source_sentence):
    header = f"Here are some examples of {src_lang} sentences and their corresponding English translations:\n"
    shots = "".join(build_example_block(src_lang, e) for e in support)

    tail = (
        f"Please help me translate the following sentence from {src_lang} to English.\n"
        f"Please answer first with the gloss and then the translation directly and enclose your translation in ###.\n"
        f"{src_lang} sentence: {source_sentence}\n"
        f"A translation for this {src_lang} sentence in English is:###"
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


class LlamaRunner:
    def __init__(self, model_id: str, use_4bit=False, use_float32=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = dict(device_map="auto")

        if use_4bit:
            if not HAS_BNB:
                raise RuntimeError("bitsandbytes not installed but --use_4bit was set")
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        elif use_float32:
            kwargs["dtype"] = torch.float32
        else:
            kwargs["dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()

    def format_chat(self, system, user):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template( 
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )###Llama 3.1
        # return system + "\n\n" + user ###Apertus

    @torch.inference_mode()
    def generate(self, system, user, max_new_tokens):
        prompt = self.format_chat(system, user)
        enc = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        t0 = time.time()
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy
            pad_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(
            out[0][enc["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        return text, time.time() - t0


def load_comet_model_with_fallback():
    if not HAS_COMET:
        print(f"[WARN] COMET import failed: {COMET_IMPORT_ERROR}")
        print("[WARN] Continuing without XCOMET scoring.")
        return None, None

    # Try XCOMET-XL first (better quality). If gated/unavailable, use wmt22-comet-da.
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
        try:
            path = download_model(fallback)
            print(f"[DEBUG] Successfully loaded {fallback}")
            return load_from_checkpoint(path), fallback
        except Exception as e2:
            print(f"[ERROR] Failed to load both models: {e2}")
            raise


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
    ap.add_argument("--covered", required=False)
    ap.add_argument("--uncovered", required=False)
    ap.add_argument("--support_n", type=int, default=21)
    ap.add_argument("--test_n", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)  # accepted but not used (greedy)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--use_float32", action="store_true")
    ap.add_argument("--max_memory_gb", type=int, default=0)  # accepted but not used
    ap.add_argument("--src_lang", default="Gitksan")
    ap.add_argument("--tgt_lang", default="English")  # accepted for compatibility
    ap.add_argument("--out_jsonl", default="results.jsonl")
    ap.add_argument("--out_metrics", default="metrics.json")
    ap.add_argument(
    "--language",
    type=str,
    required=True,
    choices=["Gitksan", "Natugu", "Lezgi", "Tsez"]
)
    ap.add_argument(
    "--model_id",
    type=str,
    default="meta-llama/Llama-3.1-8B-Instruct",
    help="HuggingFace model id"
)
    args = ap.parse_args()

    # Default src_lang to the language argument so prompts use the correct language name
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

    # MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_id = args.model_id
    print(f"[INFO] Using model: {model_id}")

    try:
        print("[DEBUG] Parsing training data...")
        covered = parse_gitdev_igt_file(covered)
        print(f"[DEBUG] Parsed {len(covered)} training examples")
        
        print("[DEBUG] Parsing test data...")
        uncovered = parse_gitdev_igt_file(uncovered)
        print(f"[DEBUG] Parsed {len(uncovered)} test examples")

        support = covered[:args.support_n]
        test = uncovered[:args.test_n]
        
        print(f"[DEBUG] Using {len(support)} support examples and {len(test)} test examples")
        print("[DEBUG] Loading model...")
        runner = LlamaRunner(args.model_id, use_4bit=args.use_4bit, use_float32=args.use_float32)
        print("[DEBUG] Model loaded successfully")

        print("[DEBUG] Loading COMET model...")
        comet_model, comet_name = load_comet_model_with_fallback()
        if comet_name is None:
            print("[INFO] COMET model: unavailable\n")
        else:
            print(f"[INFO] COMET model: {comet_name}\n")
    except Exception as e:
        print(f"[ERROR] Failed during initialization: {e}")
        import traceback
        traceback.print_exc()
        raise

    hyps_gs, hyps_cg = [], []
    refs, sources = [], []

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        from tqdm import tqdm
        for i, ex in enumerate(test):
            print(f"[DEBUG] Running example {i}")
            refs.append(ex.reference)
            sources.append(ex.source)

            sys1, usr1 = build_prompt_gloss_shot(support, args.src_lang, ex.source)
            raw1, _ = runner.generate(sys1, usr1, args.max_new_tokens)
            pred1 = extract_translation(raw1)
            hyps_gs.append(pred1)

            sys2, usr2 = build_prompt_chain_gloss(support, args.src_lang, ex.source)
            raw2, _ = runner.generate(sys2, usr2, args.max_new_tokens)
            pred2 = extract_translation(raw2)
            hyps_cg.append(pred2)

            rec = {
                "idx": i,
                "source": ex.source,
                "reference": ex.reference,
                "gloss_shot": {"prediction": pred1, "raw": raw1},
                "chain_gloss": {"prediction": pred2, "raw": raw2},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # compute metrics
    print("[DEBUG] Computing metrics...")
    metrics = {
        "gloss_shot": compute_metrics(refs, hyps_gs, sources, comet_model),
        "chain_gloss": compute_metrics(refs, hyps_cg, sources, comet_model),
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
    print("GLOSS-SHOT")
    print(f"BLEU   : {metrics['gloss_shot']['bleu']:.2f}")
    print(f"chrF++   : {metrics['gloss_shot']['chrf']:.2f}")
    if metrics["gloss_shot"]["xcomet"] is None:
        print("XCOMET  : unavailable\n")
    else:
        print(f"XCOMET  : {metrics['gloss_shot']['xcomet']:.4f}\n")

    print("CHAIN-GLOSS")
    print(f"BLEU   : {metrics['chain_gloss']['bleu']:.2f}")
    print(f"chrF++   : {metrics['chain_gloss']['chrf']:.2f}")
    if metrics["chain_gloss"]["xcomet"] is None:
        print("XCOMET  : unavailable")
    else:
        print(f"XCOMET  : {metrics['chain_gloss']['xcomet']:.4f}")

    hours = metrics["runtime_hms"]["hours"]
    minutes = metrics["runtime_hms"]["minutes"]
    seconds = metrics["runtime_hms"]["seconds"]
    print(f"\nTotal execution time: {hours}h {minutes}m {seconds}s")
    print("\n========================================\n")


if __name__ == "__main__":
    main()
