from __future__ import annotations

import argparse
import json
import os
import time

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


def load_grammar_text(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Grammar text file not found: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()
    if not text:
        raise ValueError(f"Grammar text file is empty: {path}")
    return text


def build_prompt_grammar_text_shot(support, src_lang, source_sentence, grammar_text: str):
    header = (
        f"Here are some examples of {src_lang} sentences and their corresponding English translations:\n"
    )
    shots = "".join(build_example_block(src_lang, e) for e in support)
    grammar = (
        f"\nHere is a grammar reference summary for {src_lang}. Use it as supporting context:\n"
        f"{grammar_text}\n"
    )
    tail = (
        f"\nUse the grammar summary above to translate the following {src_lang} sentence into English.\n"
        f"{src_lang} sentence: {source_sentence}\n"
        "Answer with the translation directly and enclose your translation in ###."
    )
    return SYSTEM_MSG, header + shots + grammar + tail


def build_prompt_grammar_text_chain(support, src_lang, source_sentence, grammar_text: str):
    header = (
        f"Here are some examples of {src_lang} sentences and their corresponding English translations:\n"
    )
    shots = "".join(build_example_block(src_lang, e) for e in support)
    grammar = (
        f"\nHere is a grammar reference summary for {src_lang}. Use it as supporting context:\n"
        f"{grammar_text}\n"
    )
    tail = (
        f"\nUse the grammar summary above to reason about the following {src_lang} sentence.\n"
        "First think through relevant gloss or grammar cues, then translate the sentence into English.\n"
        f"{src_lang} sentence: {source_sentence}\n"
        "Enclose only the final translation in ###."
    )
    return SYSTEM_MSG, header + shots + grammar + tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--support_n", type=int, default=21)
    ap.add_argument("--test_n", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--use_float32", action="store_true")
    ap.add_argument("--src_lang", default="Gitksan")
    ap.add_argument("--out_jsonl", default="results.jsonl")
    ap.add_argument("--out_metrics", default="metrics.json")
    ap.add_argument("--grammar_text_file", required=True)
    ap.add_argument("--language", type=str, required=True, choices=["Gitksan", "Natugu", "Lezgi", "Tsez"])
    ap.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HuggingFace model id",
    )
    args = ap.parse_args()

    if args.src_lang == "Gitksan" and args.language != "Gitksan":
        args.src_lang = args.language

    start_time = time.time()
    lang = args.language
    train_folder, train_file = LANGUAGE_FILES[lang]["train"]
    test_folder, test_file = LANGUAGE_FILES[lang]["test"]

    covered_path = os.path.join(DATA_ROOT, train_folder, train_file)
    uncovered_path = os.path.join(DATA_ROOT, test_folder, test_file)

    print(f"[INFO] Language: {lang}")
    print(f"[INFO] Train file: {covered_path}")
    print(f"[INFO] Test file: {uncovered_path}")
    print(f"[INFO] Using model: {args.model_id}")
    print(f"[INFO] Grammar text file: {args.grammar_text_file}")

    covered = parse_gitdev_igt_file(covered_path)
    uncovered = parse_gitdev_igt_file(uncovered_path)
    support = covered[: args.support_n]
    test = uncovered[: args.test_n]
    grammar_text = load_grammar_text(args.grammar_text_file)
    print(f"[INFO] Loaded grammar text: {len(grammar_text)} characters")

    runner = QwenRunner(args.model_id, use_4bit=args.use_4bit, use_float32=args.use_float32)
    comet_model, comet_name = load_comet_model_with_fallback()
    if comet_name is None:
        print("[INFO] COMET model: unavailable\n")
    else:
        print(f"[INFO] COMET model: {comet_name}\n")

    hyps_gts, hyps_gtc = [], []
    refs, sources = [], []

    with open(args.out_jsonl, "w", encoding="utf-8") as f:
        for i, ex in enumerate(test):
            print(f"[DEBUG] Running grammar-text example {i}")
            refs.append(ex.reference)
            sources.append(ex.source)

            sys1, usr1 = build_prompt_grammar_text_shot(
                support, args.src_lang, ex.source, grammar_text
            )
            raw1, _ = runner.generate(sys1, usr1, args.max_new_tokens)
            pred1 = extract_translation(raw1)
            hyps_gts.append(pred1)

            sys2, usr2 = build_prompt_grammar_text_chain(
                support, args.src_lang, ex.source, grammar_text
            )
            raw2, _ = runner.generate(sys2, usr2, args.max_new_tokens)
            pred2 = extract_translation(raw2)
            hyps_gtc.append(pred2)

            rec = {
                "idx": i,
                "source": ex.source,
                "reference": ex.reference,
                "grammar_text_file": args.grammar_text_file,
                "grammar_text_shot": {"prediction": pred1, "raw": raw1},
                "grammar_text_chain": {"prediction": pred2, "raw": raw2},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    metrics = {
        "grammar_text_shot": compute_metrics(refs, hyps_gts, sources, comet_model),
        "grammar_text_chain": compute_metrics(refs, hyps_gtc, sources, comet_model),
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
    print("GRAMMAR-TEXT-SHOT")
    print(f"BLEU   : {metrics['grammar_text_shot']['bleu']:.2f}")
    print(f"chrF++ : {metrics['grammar_text_shot']['chrf']:.2f}")
    if metrics["grammar_text_shot"]["xcomet"] is None:
        print("XCOMET : unavailable\n")
    else:
        print(f"XCOMET : {metrics['grammar_text_shot']['xcomet']:.4f}\n")

    print("GRAMMAR-TEXT-CHAIN")
    print(f"BLEU   : {metrics['grammar_text_chain']['bleu']:.2f}")
    print(f"chrF++ : {metrics['grammar_text_chain']['chrf']:.2f}")
    if metrics["grammar_text_chain"]["xcomet"] is None:
        print("XCOMET : unavailable")
    else:
        print(f"XCOMET : {metrics['grammar_text_chain']['xcomet']:.4f}")

    hours = metrics["runtime_hms"]["hours"]
    minutes = metrics["runtime_hms"]["minutes"]
    seconds = metrics["runtime_hms"]["seconds"]
    print(f"\nTotal execution time: {hours}h {minutes}m {seconds}s")
    print("\n========================================\n")


if __name__ == "__main__":
    main()
