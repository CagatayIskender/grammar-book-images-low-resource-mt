# run_grammamt_llama31_hf.py
# pip install -U transformers accelerate torch
# optional: pip install -U bitsandbytes

from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_LAYOUT_ROOT = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _layout_dir in ("", "runners/baseline", "runners/qwen3", "runners/qwen35", "runners/openrouter", "scripts/generators"):
    _layout_sys.path.insert(0, str(_LAYOUT_ROOT / _layout_dir))

from dataclasses import dataclass
from typing import List, Dict, Optional
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------
# Data structure
# ---------------------------

@dataclass
class IGTExample:
    src_lang: str
    tgt_lang: str
    source: str
    gloss: str
    translation: str


# ---------------------------
# GitDev-style IGT parser
# ---------------------------

def parse_gitdev_igt_file(path: str, src_lang: str, tgt_lang: str) -> List[IGTExample]:
    """
    Expects blocks like:
        <surface sentence>
        \\g <gloss line>
        \\l <translation line>

    Returns IGTExample list.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split roughly by blank lines, keep robustness
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    examples: List[IGTExample] = []

    surface = None
    gloss = None
    trans = None

    def flush():
        nonlocal surface, gloss, trans
        if surface and trans:
            examples.append(
                IGTExample(
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    source=surface.strip(),
                    gloss=(gloss or "").strip(),
                    translation=trans.strip(),
                )
            )
        surface = gloss = trans = None

    for b in blocks:
        lines = [ln.rstrip() for ln in b.splitlines() if ln.strip()]

        # Heuristic: surface is often the first non-tag line
        # gloss line starts with "\g"
        # translation line starts with "\l"
        maybe_surface = None
        maybe_gloss = None
        maybe_trans = None

        for ln in lines:
            if ln.lstrip().startswith("\\g"):
                maybe_gloss = ln.lstrip()[2:].strip()
            elif ln.lstrip().startswith("\\l"):
                maybe_trans = ln.lstrip()[2:].strip()
            elif not ln.lstrip().startswith("\\"):
                if maybe_surface is None:
                    maybe_surface = ln.strip()

        # If we got a complete triple in this block, flush it directly
        if maybe_surface and maybe_trans:
            examples.append(
                IGTExample(
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    source=maybe_surface,
                    gloss=(maybe_gloss or ""),
                    translation=maybe_trans,
                )
            )
        else:
            # Fallback accumulate across weird formatting
            if maybe_surface:
                surface = maybe_surface
            if maybe_gloss:
                gloss = maybe_gloss
            if maybe_trans:
                trans = maybe_trans
            if surface and trans:
                flush()

    flush()
    return examples


# ---------------------------
# GRAMMAMT prompt builders
# ---------------------------

def build_igt_block(ex: IGTExample) -> str:
    return (
        f"{ex.src_lang} sentence: {ex.source}\n"
        f"Gloss: {ex.gloss}\n"
        f"{ex.tgt_lang} sentence: {ex.translation}\n"
    )

def build_prompt_gloss_shot(examples: List[IGTExample], src_lang: str, tgt_lang: str, source_sentence: str) -> str:
    header = (
        f"Here are some examples of {src_lang} sentences\n"
        f"and their corresponding {tgt_lang} translations:\n\n"
    )
    shots = "\n".join(build_igt_block(e) for e in examples).strip()
    tail = (
        f"\n\nPlease help me translate the following sentence\n"
        f"from {src_lang} to {tgt_lang}:\n"
        f"{src_lang} sentence: {source_sentence}\n"
        f"Translation:"
    )
    return header + shots + tail

def build_prompt_chain_gloss(examples: List[IGTExample], src_lang: str, tgt_lang: str, source_sentence: str) -> str:
    header = (
        f"Here are some examples of {src_lang} sentences\n"
        f"and their corresponding {tgt_lang} translations:\n\n"
    )
    shots = "\n".join(build_igt_block(e) for e in examples).strip()
    tail = (
        f"\n\nPlease answer first with the gloss and then\n"
        f"the translation directly:\n"
        f"{src_lang} sentence: {source_sentence}\n"
        f"Gloss:"
    )
    return header + shots + tail

def parse_chain_gloss_output(text: str) -> Dict[str, str]:
    out = {"gloss": "", "translation": ""}
    t = text.strip()
    g = re.search(r"Gloss\s*:\s*(.*)", t, flags=re.IGNORECASE)
    tr = re.search(r"Translation\s*:\s*(.*)", t, flags=re.IGNORECASE)
    if g:
        out["gloss"] = g.group(1).strip()
    if tr:
        out["translation"] = tr.group(1).strip()
    if not out["gloss"] and not out["translation"]:
        out["translation"] = t
    return out


# ---------------------------
# Llama 3.1-8B Instruct via HF Transformers
# ---------------------------

class LlamaHF:
    def __init__(
        self,
        model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        load_in_4bit: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

        if torch_dtype is None:
            if torch.cuda.is_available():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32

        model_kwargs = {"torch_dtype": torch_dtype}

        if load_in_4bit:
            # needs bitsandbytes
            model_kwargs.update({"load_in_4bit": True, "device_map": "auto"})
        else:
            if torch.cuda.is_available():
                model_kwargs.update({"device_map": "auto"})

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    def generate(self, user_prompt: str, max_new_tokens: int = 256, temperature: float = 0.0, top_p: float = 1.0) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant for machine translation."},
            {"role": "user", "content": user_prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if torch.cuda.is_available():
            input_ids = input_ids.to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        if temperature and temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p})
        else:
            gen_kwargs.update({"do_sample": False})

        with torch.no_grad():
            out = self.model.generate(input_ids, **gen_kwargs)

        decoded = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()

        # best-effort: keep only the assistant continuation part
        # (chat template often includes the prompt text too)
        lower = decoded.lower()
        idx = lower.rfind("assistant")
        if idx != -1:
            decoded = decoded[idx + len("assistant"):].strip()

        return decoded


# ---------------------------
# Demo: load files, build support set, run prompts
# ---------------------------

def main():
    # Adjust these paths to your environment
    COVERED = str(_LAYOUT_ROOT.parent / "Database/2023glossingST/data/Gitksan/git-dev-track1-covered.txt")
    UNCOVERED = str(_LAYOUT_ROOT.parent / "Database/2023glossingST/data/Gitksan/git-dev-track1-uncovered.txt")

    # Set languages (edit to your real pair)
    src_lang = "Gitksan"     # example label
    tgt_lang = "English"

    # 1) Build support set from covered (few-shot examples)
    covered_examples = parse_gitdev_igt_file(COVERED, src_lang, tgt_lang)

    # pick N support examples
    N = 12
    support = covered_examples[:N]

    # 2) Pick test sentence from uncovered
    uncovered_examples = parse_gitdev_igt_file(UNCOVERED, src_lang, tgt_lang)
    test_source = uncovered_examples[0].source if uncovered_examples else "Ii hogwin kw'itxw Jacob Brown."

    # 3) Load Llama
    llm = LlamaHF(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        load_in_4bit=False,  # True if VRAM low and you installed bitsandbytes
    )

    # A) gloss-shot
    prompt_gs = build_prompt_gloss_shot(support, src_lang, tgt_lang, test_source)
    out_gs = llm.generate(prompt_gs, max_new_tokens=128, temperature=0.0)
    print("\n[gloss-shot]\n", out_gs)

    # B) chain-gloss
    prompt_cg = build_prompt_chain_gloss(support, src_lang, tgt_lang, test_source)
    out_cg_raw = llm.generate(prompt_cg, max_new_tokens=220, temperature=0.0)
    out_cg = parse_chain_gloss_output(out_cg_raw)
    print("\n[chain-gloss raw]\n", out_cg_raw)
    print("\n[chain-gloss parsed]\n", out_cg)


if __name__ == "__main__":
    main()
