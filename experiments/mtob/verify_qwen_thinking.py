#!/usr/bin/env python3
import argparse

from transformers import AutoProcessor

from mtob_grammar.backends import QwenBackend, _contains_qwen_reasoning
from mtob_grammar.config import model_config, output_path
from mtob_grammar.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify thinking controls with a cached Qwen chat template.")
    parser.add_argument("--model", required=True, choices=("qwen3", "qwen35"))
    args = parser.parse_args()
    config = model_config(args.model)
    backend = QwenBackend.__new__(QwenBackend)
    backend.processor = AutoProcessor.from_pretrained(
        config["model_id"], trust_remote_code=True, local_files_only=True
    )
    rendered, control = backend._render("THINKING-CONTROL-TEST")
    folded = rendered.casefold()
    start = folded.find("<think>")
    end = folded.find("</think>", start) if start >= 0 else -1
    think_content = folded[start + len("<think>") : end].strip() if start >= 0 and end >= 0 else None
    if think_content:
        raise RuntimeError(f"{args.model} rendered a nonempty thinking block")
    tokenizer = getattr(backend.processor, "tokenizer", backend.processor)
    protocol_probe = "<think>\nPROTOCOL-PROBE\n</think>\nFINAL"
    probe_ids = tokenizer.encode(protocol_probe, add_special_tokens=False)
    decoded_probe = tokenizer.decode(probe_ids, skip_special_tokens=True)
    if not _contains_qwen_reasoning(decoded_probe):
        raise RuntimeError(f"{args.model} reasoning tags disappeared during decoding")
    manifest = {
        "model_key": args.model,
        "model_id": config["model_id"],
        "enable_thinking": False,
        "template_control": control,
        "think_tags_present": start >= 0,
        "think_content_empty": think_content in (None, ""),
        "reasoning_tags_survive_clean_decode": True,
        "reasoning_protocol_detector_verified": True,
        "single_user_message": True,
        "rendered_template": rendered,
    }
    write_json(output_path("manifests", "thinking", f"{args.model}.json"), manifest)
    print(f"[OK] {args.model}: enable_thinking=False; nonempty thinking content absent")


if __name__ == "__main__":
    main()
