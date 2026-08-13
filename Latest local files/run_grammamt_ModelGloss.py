from __future__ import annotations

import argparse
import csv
import io
import json
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

import torch
print(torch.cuda.memory_allocated() / 1e9)
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "Database", "2023glossingST", "data")
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except Exception:
    HAS_BNB = False

import sacrebleu
from comet import download_model, load_from_checkpoint


# GlossLM prediction CSVs (frozen/glosslm-v1 branch, no-translation experiments)
GLOSSLM_PREDS_URLS = {
    "Gitksan": "https://raw.githubusercontent.com/lecs-lab/polygloss/frozen/glosslm-v1/preds/glosslm-all-no_trans/git-all-no_trans/test_OOD-preds.csv",
}


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


def load_glosslm_predictions(language: str, local_path: Optional[str] = None) -> List[str]:
    """
    Load GlossLM predicted glosses for the test set (unsegmented variant).
    Returns a list of predicted gloss strings, one per test example.
    """
    if local_path and os.path.exists(local_path):
        print(f"[INFO] Loading GlossLM predictions from local file: {local_path}")
        with open(local_path, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        url = GLOSSLM_PREDS_URLS.get(language)
        if url is None:
            raise ValueError(f"No GlossLM prediction URL configured for language: {language}")
        print(f"[INFO] Downloading GlossLM predictions from: {url}")
        with urllib.request.urlopen(url) as response:
            content = response.read().decode("utf-8")

    reader = csv.DictReader(io.StringIO(content))
    rows_by_idx = {}
    for row in reader:
        if row["is_segmented"] == "no":
            idx = int(row["id"].rsplit("_", 1)[-1])
            rows_by_idx[idx] = row["pred"].replace("\n", " ").strip()

    max_idx = max(rows_by_idx.keys())
    predictions = [rows_by_idx.get(i, "") for i in range(max_idx + 1)]
    print(f"[INFO] Loaded {len(predictions)} GlossLM predictions")
    return predictions


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


def build_prompt_model_gloss(support, src_lang, source_sentence, predicted_gloss: str):
    """
    Like gloss_shot but explicitly provides the GlossLM-predicted gloss for the
    test sentence, giving the model morphological structure information.
    """
    header = f"Here are some examples of {src_lang} sentences and their corresponding English translations:\n"
    shots = "".join(build_example_block(src_lang, e) for e in support)
    tail = (
        f"Please help me translate the following sentence from {src_lang} to English. "
        f"Please answer with the translation directly and enclose your translation in ###.\n"
        f"{src_lang} sentence: {source_sentence}\n"
        f"Gloss: {predicted_gloss}\n"
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


class ModelRunner:
    def __init__(self, model_id: str, use_4bit=False, use_float32=False):
        self.model_id = model_id
        import sys
        self.is_vl_model = "vl" in model_id.lower()

        print(f"[DEBUG] Loading processor/tokenizer for {model_id}...")
        sys.stdout.flush()

        if self.is_vl_model:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            print(f"[DEBUG] VL processor loaded")
            kwargs = dict(device_map="auto", trust_remote_code=True)
            if use_4bit:
                kwargs["load_in_4bit"] = True
            else:
                kwargs["torch_dtype"] = torch.float32 if use_float32 else torch.float16
            print(f"[DEBUG] Loading VL model with kwargs: {list(kwargs.keys())}")
            sys.stdout.flush()
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
            if not use_4bit:
                self.model = self.model.eval()
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"[DEBUG] Tokenizer loaded")
            kwargs = dict(device_map="auto", trust_remote_code=True)
            if use_4bit:
                if not HAS_BNB:
                    raise RuntimeError("bitsandbytes not installed but --use_4bit was set")
                kwargs["load_in_4bit"] = True
            else:
                if use_float32 or "apertus" in model_id.lower():
                    kwargs["torch_dtype"] = torch.float32
                else:
                    kwargs["torch_dtype"] = torch.float16
            print(f"[DEBUG] Loading model with kwargs: {list(kwargs.keys())}")
            sys.stdout.flush()
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
            if not use_4bit:
                self.model = self.model.eval()

        print(f"[DEBUG] Model loaded on device: {self.model.device}")
        sys.stdout.flush()

    def generate(self, system, user, max_new_tokens):
        t0 = time.time()

        if self.is_vl_model:
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user",   "content": [{"type": "text", "text": user}]},
            ]
            text_prompt = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(text=[text_prompt], return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            generated_ids = [out[i][len(inputs.input_ids[i]):] for i in range(len(out))]
            text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            n_tokens = len(generated_ids[0])
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ]
            try:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                print(f"[WARN] Chat template failed: {e}, using fallback")
                prompt = f"{system}\n\n{user}"
            enc = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            generated_token_ids = out[0, enc["input_ids"].shape[1]:]
            text = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip()
            n_tokens = len(generated_token_ids)

        print(f"[DEBUG] Generated {n_tokens} tokens")
        if len(text) > 50:
            print(f"[DEBUG] Output: '{text[:100]}...'")
        else:
            print(f"[DEBUG] Output: '{text}'")
        return text, time.time() - t0


def load_comet_model_with_fallback():
    preferred = "Unbabel/XCOMET-XL"
    fallback = "Unbabel/wmt22-comet-da"
    try:
        print(f"[DEBUG] Attempting to load {preferred}...")
        path = download_model(preferred)
        return load_from_checkpoint(path), preferred
    except Exception as e:
        print(f"[WARN] Could not load {preferred}: {e}")
        path = download_model(fallback)
        return load_from_checkpoint(path), fallback


def compute_metrics(refs, hyps, sources, comet_model):
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(sources, hyps, refs)]
    scores = comet_model.predict(data, batch_size=16, gpus=1 if torch.cuda.is_available() else 0)
    xcomet = sum(scores["scores"]) / len(scores["scores"])
    return {
        "bleu": float(bleu.score),
        "chrf": float(chrf.score),
        "xcomet": float(xcomet),
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
    ap.add_argument("--language", type=str, required=True, choices=["Gitksan", "Natugu", "Lezgi", "Tsez"])
    ap.add_argument("--model_id", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--support_n", type=int, default=21)
    ap.add_argument("--test_n", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--use_float32", action="store_true")
    ap.add_argument("--src_lang", default="Gitksan")
    ap.add_argument("--glosslm_preds_csv", type=str, default=None,
                    help="Path to a local GlossLM predictions CSV. If omitted, downloads from GitHub.")
    ap.add_argument("--out_jsonl", default="results.jsonl")
    ap.add_argument("--out_metrics", default="metrics.json")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_memory_gb", type=int, default=0)
    ap.add_argument("--tgt_lang", default="English")
    args = ap.parse_args()

    start_time = time.time()
    lang = args.language

    train_folder, train_file = LANGUAGE_FILES[lang]["train"]
    test_folder, test_file = LANGUAGE_FILES[lang]["test"]

    covered_path = os.path.join(DATA_ROOT, train_folder, train_file)
    uncovered_path = os.path.join(DATA_ROOT, test_folder, test_file)

    print(f"[INFO] Language: {lang}")
    print(f"[INFO] Model: {args.model_id}")

    try:
        covered = parse_gitdev_igt_file(covered_path)
        uncovered = parse_gitdev_igt_file(uncovered_path)
        print(f"[DEBUG] Train: {len(covered)} | Test: {len(uncovered)}")

        support = covered[:args.support_n]
        test = uncovered[:args.test_n]

        if lang not in GLOSSLM_PREDS_URLS and args.glosslm_preds_csv is None:
            raise ValueError(f"No GlossLM predictions available for language '{lang}'. "
                             f"Provide --glosslm_preds_csv.")
        glosslm_preds = load_glosslm_predictions(lang, local_path=args.glosslm_preds_csv)
        if len(glosslm_preds) < len(test):
            raise ValueError(f"GlossLM predictions ({len(glosslm_preds)}) < test examples ({len(test)})")

        runner = ModelRunner(args.model_id, use_4bit=args.use_4bit, use_float32=args.use_float32)
        comet_model, comet_name = load_comet_model_with_fallback()
        print(f"[INFO] COMET model: {comet_name}\n")
    except Exception as e:
        import traceback
        print(f"[ERROR] Initialization failed: {e}")
        traceback.print_exc()
        raise

    hyps_gs, hyps_cg, hyps_mg = [], [], []
    refs, sources = [], []

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for i, ex in enumerate(test):
            print(f"[DEBUG] Example {i}/{len(test)-1}")
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

            sys3, usr3 = build_prompt_model_gloss(support, args.src_lang, ex.source, glosslm_preds[i])
            raw3, _ = runner.generate(sys3, usr3, args.max_new_tokens)
            pred3 = extract_translation(raw3)
            hyps_mg.append(pred3)

            rec = {
                "idx": i,
                "source": ex.source,
                "reference": ex.reference,
                "glosslm_pred_gloss": glosslm_preds[i],
                "gloss_shot":  {"prediction": pred1, "raw": raw1},
                "chain_gloss": {"prediction": pred2, "raw": raw2},
                "model_gloss": {"prediction": pred3, "raw": raw3},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("[DEBUG] Computing metrics...")
    metrics = {
        "gloss_shot":  compute_metrics(refs, hyps_gs, sources, comet_model),
        "chain_gloss": compute_metrics(refs, hyps_cg, sources, comet_model),
        "model_gloss": compute_metrics(refs, hyps_mg, sources, comet_model),
    }

    with open(args.out_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[DEBUG] Results: {args.out_jsonl} | Metrics: {args.out_metrics}")

    print("================ RESULTS ================\n")
    for strategy, m in metrics.items():
        label = strategy.upper().replace("_", "-")
        print(f"{label}")
        print(f"  BLEU   : {m['bleu']:.2f}")
        print(f"  chrF++ : {m['chrf']:.2f}")
        print(f"  XCOMET : {m['xcomet']:.4f}\n")

    elapsed = time.time() - start_time
    h, rem = divmod(int(elapsed), 3600)
    m_t, s = divmod(rem, 60)
    print(f"Total execution time: {h}h {m_t}m {s}s")
    print("\n========================================\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
