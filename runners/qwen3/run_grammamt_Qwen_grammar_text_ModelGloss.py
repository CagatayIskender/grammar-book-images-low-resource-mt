from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_LAYOUT_ROOT = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _layout_dir in ("", "runners/baseline", "runners/qwen3", "runners/qwen35", "runners/openrouter", "scripts/generators"):
    _layout_sys.path.insert(0, str(_LAYOUT_ROOT / _layout_dir))


import argparse
import json
import os
import time

from run_grammamt_ModelGloss import load_glosslm_predictions
from run_grammamt_Qwen import (
    DATA_ROOT,
    LANGUAGE_FILES,
    QwenRunner,
    SYSTEM_MSG,
    build_example_block,
    compute_metrics,
    extract_translation,
    load_comet_model_with_fallback,
    parse_gitdev_igt_file,
)
from run_grammamt_Qwen_grammar_text import load_grammar_text


def build_prompt_grammar_text_model_gloss(support, src_lang, source_sentence, predicted_gloss, grammar_text):
    header = f"Here are some examples of {src_lang} sentences and their corresponding English translations:\n"
    shots = "".join(build_example_block(src_lang, e) for e in support)
    grammar = (
        f"\nHere is a grammar reference summary for {src_lang}. Use it as supporting context:\n"
        f"{grammar_text}\n"
    )
    tail = (
        f"\nUse the grammar summary and the predicted gloss below as supporting context to translate the following {src_lang} sentence into English.\n"
        f"{src_lang} sentence: {source_sentence}\n"
        f"Gloss: {predicted_gloss}\n"
        "Answer with the translation directly and enclose your translation in ###."
    )
    return SYSTEM_MSG, header + shots + grammar + tail


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
    ap.add_argument("--grammar_text_file", required=True)
    ap.add_argument("--glosslm_preds_csv", default=None)
    ap.add_argument("--language", type=str, required=True, choices=["Gitksan", "Natugu", "Lezgi", "Tsez"])
    ap.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    args = ap.parse_args()
    if args.src_lang is None:
        args.src_lang = args.language

    if args.src_lang == "Gitksan" and args.language != "Gitksan":
        args.src_lang = args.language

    start_time = time.time()
    train_folder, train_file = LANGUAGE_FILES[args.language]["train"]
    test_folder, test_file = LANGUAGE_FILES[args.language]["test"]

    covered_path = os.path.join(DATA_ROOT, train_folder, train_file)
    uncovered_path = os.path.join(DATA_ROOT, test_folder, test_file)

    print(f"[INFO] Language: {args.language}")
    print(f"[INFO] Train file: {covered_path}")
    print(f"[INFO] Test file: {uncovered_path}")
    print(f"[INFO] Using model: {args.model_id}")
    print(f"[INFO] Grammar text file: {args.grammar_text_file}")

    covered = parse_gitdev_igt_file(covered_path)
    uncovered = parse_gitdev_igt_file(uncovered_path)
    support = covered[: args.support_n]
    test = uncovered[: args.test_n]
    grammar_text = load_grammar_text(args.grammar_text_file)

    glosslm_predictions = load_glosslm_predictions(args.language, args.glosslm_preds_csv)
    if len(glosslm_predictions) < len(test):
        raise ValueError(
            f"Only {len(glosslm_predictions)} GlossLM predictions available, but test_n={len(test)}"
        )

    runner = QwenRunner(args.model_id, use_4bit=args.use_4bit, use_float32=args.use_float32)
    comet_model, comet_name = None, None
    print("[INFO] XCOMET must run in the separate scoring job")
    if comet_name is None:
        print("[INFO] COMET model: unavailable\n")
    else:
        print(f"[INFO] COMET model: {comet_name}\n")

    hyps, refs, sources = [], [], []

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for i, ex in enumerate(test):
            print(f"[DEBUG] Running grammar-text ModelGloss example {i}")
            refs.append(ex.reference)
            sources.append(ex.source)

            system, user = build_prompt_grammar_text_model_gloss(
                support, args.src_lang, ex.source, glosslm_predictions[i], grammar_text
            )
            raw, _ = runner.generate(system, user, args.max_new_tokens)
            pred = extract_translation(raw)
            hyps.append(pred)

            rec = {
                "idx": i,
                "source": ex.source,
                "reference": ex.reference,
                "grammar_text_file": args.grammar_text_file,
                "grammar_text_model_gloss": {
                    "prediction": pred,
                    "raw": raw,
                    "glosslm_pred_gloss": glosslm_predictions[i],
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    metrics = {
        "grammar_text_model_gloss": compute_metrics(refs, hyps, sources, comet_model),
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
    print("GRAMMAR-TEXT-MODELGLOSS")
    print(f"BLEU   : {metrics['grammar_text_model_gloss']['bleu']:.2f}")
    print(f"chrF++ : {metrics['grammar_text_model_gloss']['chrf']:.2f}")
    if metrics["grammar_text_model_gloss"]["xcomet"] is None:
        print("XCOMET : unavailable")
    else:
        print(f"XCOMET : {metrics['grammar_text_model_gloss']['xcomet']:.4f}")
    print("\n========================================\n")


if __name__ == "__main__":
    main()
