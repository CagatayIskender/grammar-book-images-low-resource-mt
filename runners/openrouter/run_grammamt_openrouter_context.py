from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_LAYOUT_ROOT = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _layout_dir in ("", "runners/baseline", "runners/qwen3", "runners/qwen35", "runners/openrouter", "scripts/generators"):
    _layout_sys.path.insert(0, str(_LAYOUT_ROOT / _layout_dir))


import argparse
import base64
import csv
import hashlib
import io
import json
import math
import mimetypes
import os
import random
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests
import sacrebleu


PROJECT_DIR = _LAYOUT_ROOT
DATA_ROOT = PROJECT_DIR.parent / "Database" / "2023glossingST" / "data"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_MODEL_LABEL = "gemini25flashlite"
DEFAULT_REASONING_EFFORT = os.environ.get("OPENROUTER_REASONING_EFFORT", "none")
FINAL_PREFIX = "FINAL_TRANSLATION:"

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

GLOSSLM_PREDS_URLS = {
    "Gitksan": "https://raw.githubusercontent.com/lecs-lab/polygloss/frozen/glosslm-v1/preds/glosslm-all-no_trans/git-all-no_trans/test_OOD-preds.csv",
    "Natugu": "https://raw.githubusercontent.com/lecs-lab/polygloss/frozen/glosslm-v1/preds/glosslm-all-no_trans/ntu-all-no_trans/test_OOD-preds.csv",
    "Lezgi": "https://raw.githubusercontent.com/lecs-lab/polygloss/frozen/glosslm-v1/preds/glosslm-all-no_trans/lez-all-no_trans/test_OOD-preds.csv",
    "Tsez": "https://raw.githubusercontent.com/lecs-lab/polygloss/frozen/glosslm-v1/preds/glosslm-all-no_trans/ddo-all-no_trans/test_ID-preds.csv",
}

DIRECT_SYSTEM_MSG = (
    "You are a direct translation engine. "
    "Return only the requested English translation in the requested output format."
)
OUTPUT_INSTRUCTION = (
    f"Return exactly one line in this format: {FINAL_PREFIX} <English translation>\n"
    "Do not include notes, glosses, reasoning, bullet points, markdown, or extra text."
)


@dataclass(frozen=True)
class IGTExample:
    source: str
    gloss: str
    reference: str


def parse_gitdev_igt_file(path: Path) -> list[IGTExample]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    examples = []
    for block in blocks:
        source = None
        gloss = ""
        reference = None
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("\\t"):
                source = line[2:].strip()
            elif line.startswith("\\g"):
                gloss = line[2:].strip()
            elif line.startswith("\\l"):
                reference = line[2:].strip().split("||", 1)[0].strip()
        if source and reference:
            examples.append(IGTExample(source, gloss, reference))
    return examples


def example_block(language: str, example: IGTExample) -> str:
    return (
        f"{language} sentence: {example.source}\n"
        f"Gloss: {example.gloss}\n"
        f"English translation: {example.reference}\n\n"
    )


def build_prompt(
    support: list[IGTExample],
    language: str,
    source: str,
    context_kind: str,
    grammar_text: str,
    predicted_gloss: str | None,
) -> tuple[str, str]:
    shots = "".join(example_block(language, example) for example in support)
    if context_kind == "text":
        context = (
            f"\nHere is a grammar reference summary for {language}. "
            f"Use it as supporting context:\n{grammar_text}\n"
        )
    elif context_kind == "none":
        context = ""
    else:
        context = "\nThe attached images are pages from a grammar reference for this language.\n"

    gloss_context = ""
    if predicted_gloss is not None:
        gloss_context = f"Gloss: {predicted_gloss}\n"

    user = (
        f"Here are some examples of {language} sentences and their corresponding English translations:\n"
        f"{shots}{context}\n"
        + ("Use the grammar reference as supporting context.\n" if context_kind != "none" else "")
        +
        f"{OUTPUT_INSTRUCTION}\n\n"
        f"{language} sentence: {source}\n"
        f"{gloss_context}{FINAL_PREFIX}"
    )
    return DIRECT_SYSTEM_MSG, user


def extract_translation(text: str) -> str:
    matches = re.findall(
        rf"(?im)^\s*{re.escape(FINAL_PREFIX)}\s*(.+?)\s*$",
        text,
    )
    if matches:
        return matches[-1].strip().strip("`")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for prefix in ("Final translation:", "Translation:", "Answer:", "English translation:"):
        for line in reversed(lines):
            if line.lower().startswith(prefix.lower()):
                return line[len(prefix) :].strip().strip("`")
    return lines[-1].strip(" `") if lines else ""


def load_context_text(path: str) -> str:
    if not path:
        return ""
    text = Path(path).read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        raise ValueError(f"Grammar text file is empty: {path}")
    return text


def load_image_data_urls(image_dir: str) -> tuple[list[str], list[str]]:
    if not image_dir:
        return [], []
    directory = Path(image_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Grammar image directory not found: {image_dir}")
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    if not paths:
        raise FileNotFoundError(f"No grammar images found in: {image_dir}")
    data_urls = []
    for path in paths:
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        data_urls.append(f"data:{mime};base64,{encoded}")
    return [str(path) for path in paths], data_urls


def load_glosslm_predictions(language: str, cache_dir: Path) -> list[str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{language.lower()}_glosslm_predictions.csv"
    if not cache_path.exists():
        url = GLOSSLM_PREDS_URLS[language]
        print(f"[INFO] Caching GlossLM predictions from {url}")
        with urllib.request.urlopen(url, timeout=120) as response:
            cache_path.write_bytes(response.read())
    content = cache_path.read_text(encoding="utf-8")
    rows_by_idx: dict[int, str] = {}
    for row in csv.DictReader(io.StringIO(content)):
        if row["is_segmented"] == "no":
            idx = int(row["id"].rsplit("_", 1)[-1])
            rows_by_idx[idx] = row["pred"].replace("\n", " ").strip()
    return [rows_by_idx.get(idx, "") for idx in range(max(rows_by_idx) + 1)]


def load_existing_records(path: Path, expected_sources: list[str], result_key: str) -> dict[int, dict]:
    if not path.exists():
        return {}
    records: dict[int, dict] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
        idx = record.get("idx")
        if not isinstance(idx, int) or idx >= len(expected_sources):
            raise ValueError(f"Unexpected record index in {path}:{line_number}")
        if record.get("source") != expected_sources[idx] or result_key not in record:
            raise ValueError(f"Existing result does not match this experiment: {path}:{line_number}")
        records[idx] = record
    return records


def request_with_retries(
    api_key: str,
    model: str,
    system: str,
    user: str,
    image_data_urls: list[str],
    max_tokens: int,
    temperature: float | None,
    seed: int,
    timeout: int,
    max_retries: int,
    allow_data_collection: bool,
    reasoning_effort: str,
) -> tuple[str, dict[str, Any]]:
    user_content: list[dict[str, Any]] = [{"type": "text", "text": user}]
    user_content.extend(
        {"type": "image_url", "image_url": {"url": data_url}}
        for data_url in image_data_urls
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "seed": seed,
        "max_tokens": max_tokens,
        "reasoning": {"effort": reasoning_effort},
        "provider": {
            "data_collection": "allow" if allow_data_collection else "deny",
            "require_parameters": True,
        },
    }
    if temperature is not None:
        payload["temperature"] = temperature
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "GrammarMT OpenRouter experiments",
    }
    retryable = {408, 409, 429, 500, 502, 503, 504}
    for attempt in range(max_retries + 1):
        started = time.time()
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            if attempt == max_retries:
                raise RuntimeError(f"OpenRouter request failed after retries: {exc}") from exc
            delay = min(60.0, 2**attempt + random.random())
            print(f"[WARN] Network error; retrying in {delay:.1f}s: {exc}")
            time.sleep(delay)
            continue

        latency = time.time() - started
        if response.ok:
            data = response.json()
            message = data["choices"][0]["message"]
            metadata = {
                "request_id": data.get("id"),
                "model": data.get("model", model),
                "provider": data.get("provider"),
                "usage": data.get("usage", {}),
                "latency_seconds": latency,
            }
            return message.get("content") or "", metadata

        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", response.text)
        except ValueError:
            error_message = response.text
        error_message = error_message[:1000]
        if response.status_code not in retryable or attempt == max_retries:
            raise RuntimeError(f"OpenRouter HTTP {response.status_code}: {error_message}")
        retry_after = response.headers.get("Retry-After")
        try:
            delay = float(retry_after) if retry_after else min(60.0, 2**attempt + random.random())
        except ValueError:
            delay = min(60.0, 2**attempt + random.random())
        print(f"[WARN] OpenRouter HTTP {response.status_code}; retrying in {delay:.1f}s")
        time.sleep(delay)
    raise AssertionError("Retry loop exited unexpectedly")


def compute_basic_metrics(refs: list[str], hyps: list[str]) -> dict[str, float | None]:
    return {
        "bleu": float(sacrebleu.corpus_bleu(hyps, [refs]).score),
        "chrf": float(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score),
        "xcomet": None,
    }


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def progress_line(
    completed: int,
    total: int,
    elapsed: float,
    started_completed: int,
    total_cost: float,
) -> str:
    width = 30
    fraction = completed / total if total else 1.0
    filled = min(width, int(fraction * width))
    bar = "#" * filled + "-" * (width - filled)
    completed_this_run = completed - started_completed
    remaining = total - completed
    eta = None
    if completed_this_run > 0:
        eta = elapsed / completed_this_run * remaining
    return (
        f"[PROGRESS] [{bar}] {completed}/{total} ({fraction * 100:5.1f}%) "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)} "
        f"cost=${total_cost:.6f}"
    )


def experiment_fingerprint(args: argparse.Namespace, image_paths: list[str]) -> str:
    payload = {
        "model": args.model,
        "language": args.language,
        "support_n": args.support_n,
        "test_n": args.test_n,
        "grammar_text_file": args.grammar_text_file,
        "grammar_image_dir": args.grammar_image_dir,
        "grammar_image_paths": image_paths,
        "model_gloss": args.model_gloss,
        "temperature": args.temperature,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
    }
    if getattr(args, "baseline", False):
        payload["baseline"] = True
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", required=True, choices=sorted(LANGUAGE_FILES))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model_label", default=DEFAULT_MODEL_LABEL)
    parser.add_argument("--support_n", type=int, default=21)
    parser.add_argument("--test_n", type=int, required=True)
    parser.add_argument("--grammar_text_file", default="")
    parser.add_argument("--grammar_image_dir", default="")
    parser.add_argument("--baseline", action="store_true", help="No grammar context; retain support examples and optional ModelGloss.")
    parser.add_argument("--model_gloss", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--omit_temperature",
        action="store_true",
        help="Do not send temperature to models that do not support it",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max_retries", type=int, default=8)
    parser.add_argument("--allow_data_collection", action="store_true")
    parser.add_argument(
        "--reasoning_effort",
        default=DEFAULT_REASONING_EFFORT,
        choices=("none", "low", "medium", "high"),
        help="OpenRouter reasoning effort. The default disables thinking.",
    )
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument("--out_metrics", required=True)
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z0-9]+", args.model_label):
        raise ValueError("--model_label must contain only lowercase letters and digits")
    if args.omit_temperature:
        args.temperature = None

    if sum(map(bool, (args.baseline, args.grammar_text_file, args.grammar_image_dir))) != 1:
        raise ValueError("Set exactly one of --baseline, --grammar_text_file or --grammar_image_dir")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    train_folder, train_file = LANGUAGE_FILES[args.language]["train"]
    test_folder, test_file = LANGUAGE_FILES[args.language]["test"]
    covered = parse_gitdev_igt_file(DATA_ROOT / train_folder / train_file)
    uncovered = parse_gitdev_igt_file(DATA_ROOT / test_folder / test_file)
    support = covered[: args.support_n]
    test = uncovered[: args.test_n]
    if len(test) != args.test_n:
        raise ValueError(f"Requested {args.test_n} tests but found {len(test)}")

    context_kind = "none" if args.baseline else ("text" if args.grammar_text_file else "image")
    grammar_text = load_context_text(args.grammar_text_file)
    image_paths, image_data_urls = load_image_data_urls(args.grammar_image_dir)
    predicted_glosses = None
    if args.model_gloss:
        predicted_glosses = load_glosslm_predictions(args.language, PROJECT_DIR / "runners/openrouter" / "cache")
        if len(predicted_glosses) < len(test):
            raise ValueError("Not enough GlossLM predictions for the selected test set")

    result_key = (
        f"{args.model_label}_modelgloss"
        if args.model_gloss
        else args.model_label
    )
    out_jsonl = Path(args.out_jsonl)
    out_metrics = Path(args.out_metrics)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    expected_sources = [example.source for example in test]
    records = load_existing_records(out_jsonl, expected_sources, result_key)
    fingerprint = experiment_fingerprint(args, image_paths)
    for record in records.values():
        existing_fingerprint = record.get("experiment_fingerprint")
        if existing_fingerprint and existing_fingerprint != fingerprint:
            raise ValueError(f"Existing output belongs to a different experiment: {out_jsonl}")

    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Reasoning effort: {args.reasoning_effort}")
    print(f"[INFO] Language: {args.language}")
    print(f"[INFO] Context: {context_kind}")
    print(f"[INFO] ModelGloss: {args.model_gloss}")
    print(f"[INFO] Resuming with {len(records)}/{len(test)} completed records")
    if image_paths:
        print(f"[INFO] Grammar images: {len(image_paths)}")

    started = time.time()
    started_completed = len(records)
    existing_cost = sum(
        (record[result_key].get("api", {}).get("usage", {}).get("cost") or 0)
        for record in records.values()
    )
    print(progress_line(len(records), len(test), 0.0, started_completed, existing_cost), flush=True)
    with out_jsonl.open("a", encoding="utf-8") as output:
        for idx, example in enumerate(test):
            if idx in records:
                continue
            predicted_gloss = predicted_glosses[idx] if predicted_glosses is not None else None
            system, user = build_prompt(
                support,
                args.language,
                example.source,
                context_kind,
                grammar_text,
                predicted_gloss,
            )
            print(f"[INFO] Request {idx + 1}/{len(test)}")
            raw, api_metadata = request_with_retries(
                api_key,
                args.model,
                system,
                user,
                image_data_urls,
                args.max_tokens,
                args.temperature,
                args.seed,
                args.timeout,
                args.max_retries,
                args.allow_data_collection,
                args.reasoning_effort,
            )
            prediction = extract_translation(raw)
            if not prediction:
                raise RuntimeError(f"Empty translation for test example {idx}")
            result = {
                "prediction": prediction,
                "raw": raw,
                "api": api_metadata,
            }
            if predicted_gloss is not None:
                result["glosslm_pred_gloss"] = predicted_gloss
            record = {
                "idx": idx,
                "source": example.source,
                "reference": example.reference,
                "context_kind": context_kind,
                "grammar_text_file": args.grammar_text_file,
                "grammar_image_paths": image_paths,
                "experiment_fingerprint": fingerprint,
                result_key: result,
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            records[idx] = record
            current_cost = sum(
                (saved[result_key].get("api", {}).get("usage", {}).get("cost") or 0)
                for saved in records.values()
            )
            print(
                progress_line(
                    len(records),
                    len(test),
                    time.time() - started,
                    started_completed,
                    current_cost,
                ),
                flush=True,
            )

    ordered_records = [records[idx] for idx in range(len(test))]
    refs = [record["reference"] for record in ordered_records]
    hyps = [record[result_key]["prediction"] for record in ordered_records]
    usage_fields = ("prompt_tokens", "completion_tokens", "total_tokens", "cost")
    usage = {
        field: sum(
            (record[result_key].get("api", {}).get("usage", {}).get(field) or 0)
            for record in ordered_records
        )
        for field in usage_fields
    }
    metrics = {
        result_key: compute_basic_metrics(refs, hyps),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "seed": args.seed,
        "support_n": args.support_n,
        "test_n": args.test_n,
        "usage": usage,
        "comet_model_used": None,
        "result_jsonl": str(out_jsonl),
        "runtime_seconds_this_invocation": time.time() - started,
    }
    out_metrics.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[WARN] Interrupted; completed JSONL records are safe and the run can resume", file=sys.stderr)
        raise
