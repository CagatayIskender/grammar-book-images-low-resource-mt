from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .backends import ReasoningViolation, create_backend
from .chunking import gpt2_encoding
from .config import ROOT, ensure_output_parent, input_path, model_config, output_path, source_config
from .data import parse_igt
from .evaluation import evaluate, paper_clean
from .io import read_jsonl, write_json
from .prompts import appendix_c_prompt, extract_translation, is_refusal, long_context, passage_context


CONDITIONS = ("ge", "gs", "gl")


def output_paths(model_key: str, source_id: str, condition: str) -> tuple[Path, Path]:
    directory = (model_key, source_id)
    return (
        output_path("results", *directory, f"results_{condition}.jsonl"),
        output_path("metrics", *directory, f"metrics_{condition}.json"),
    )


def _load_existing(path: Path, examples) -> dict[int, dict[str, Any]]:
    if not path.is_file():
        return {}
    records = read_jsonl(path)
    by_index: dict[int, dict[str, Any]] = {}
    for record in records:
        index = int(record["index"])
        if index >= len(examples) or record["source"] != examples[index].source:
            raise ValueError(f"Existing result does not match experiment: {path}")
        by_index[index] = record
    return by_index


def run(model_key: str, source_id: str, condition: str, smoke: bool = False) -> dict[str, Any]:
    condition = condition.casefold()
    if condition not in CONDITIONS:
        raise ValueError(f"Condition must be one of {', '.join(CONDITIONS)}")
    source = source_config(source_id)
    model = model_config(model_key)
    examples = parse_igt(input_path(source["test_file"]), limit=2 if smoke else int(source["test_n"]))
    result_path, metrics_path = output_paths(model_key, source_id, condition)
    if smoke:
        result_path = output_path("results", "smoke", model_key, source_id, f"results_{condition}.jsonl")
        metrics_path = output_path("metrics", "smoke", model_key, source_id, f"metrics_{condition}.json")
    ensure_output_parent(result_path)
    existing = _load_existing(result_path, examples)
    retrieval_manifest = {}
    if condition in {"ge", "gs"}:
        manifest_path = output_path("manifests", source_id, f"retrieval_{condition}.jsonl")
        for record in read_jsonl(manifest_path):
            retrieval_manifest[int(record["index"])] = record
    gl_text = (
        output_path("gl_contexts", source_id, "grammar_long.txt").read_text(encoding="utf-8")
        if condition == "gl"
        else ""
    )
    backend = create_backend(model)
    started = time.time()

    with result_path.open("a", encoding="utf-8") as handle:
        for example in examples:
            if example.index in existing and existing[example.index].get("status") == "ok":
                continue
            if condition == "ge":
                retrieval_record = retrieval_manifest[example.index]
                if retrieval_record["source"] != example.source:
                    raise ValueError(f"Frozen Ge retrieval mismatch at example {example.index}")
                retrieved = retrieval_record["passages"]
                context = passage_context(source["language"], [item["text"] for item in retrieved])
            elif condition == "gs":
                retrieval_record = retrieval_manifest[example.index]
                if retrieval_record["source"] != example.source:
                    raise ValueError(f"Frozen Gs retrieval mismatch at example {example.index}")
                retrieved = retrieval_record["passages"]
                context = passage_context(source["language"], [item["text"] for item in retrieved])
            else:
                retrieved = []
                context = long_context(source["language"], gl_text)
            prompt = appendix_c_prompt(source["language"], source["location"], example.source, context)
            try:
                initial = backend.generate(prompt, seed=2024 + example.index)
            except ReasoningViolation as exc:
                violation = {
                    "index": example.index,
                    "source": example.source,
                    "reference": example.reference,
                    "prediction": "",
                    "cleaned_prediction": "",
                    "cleaned_reference": paper_clean(example.reference),
                    "status": "reasoning_violation",
                    "source_id": source_id,
                    "condition": condition,
                    "model_key": model_key,
                    "model_id": model["model_id"],
                    "prompt": prompt,
                    "violation": str(exc),
                }
                handle.write(json.dumps(violation, ensure_ascii=False) + "\n")
                handle.flush()
                raise
            final = initial
            retry_prompt = None
            if is_refusal(initial.text):
                retry_prompt = appendix_c_prompt(
                    source["language"], source["location"], example.source, context, refusal=True
                )
                final = backend.generate(retry_prompt, seed=3024 + example.index)
            prediction = extract_translation(final.text)
            final_refusal = is_refusal(final.text)
            record = {
                "index": example.index,
                "source": example.source,
                "reference": example.reference,
                "prediction": prediction,
                "cleaned_prediction": paper_clean(prediction),
                "cleaned_reference": paper_clean(example.reference),
                "status": "refusal" if final_refusal else ("ok" if prediction else "empty"),
                "source_id": source_id,
                "language": source["language"],
                "condition": condition,
                "model_key": model_key,
                "model_id": model["model_id"],
                "prompt": prompt,
                "prompt_gpt2_tokens": len(gpt2_encoding().encode(prompt)),
                "retrieved_passages": [
                    {
                        key: item[key]
                        for key in (
                            "chunk_id",
                            "chunk_index",
                            "token_start",
                            "token_end",
                            "pdf_pages",
                            "retrieval_score",
                            "retrieval_method",
                        )
                    }
                    for item in retrieved
                ],
                "initial_response": initial.text,
                "initial_metadata": initial.metadata,
                "refusal_retry_prompt": retry_prompt,
                "retry_response": final.text if retry_prompt else None,
                "retry_metadata": final.metadata if retry_prompt else None,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            completed = example.index + 1
            elapsed = time.time() - started
            rate = completed / elapsed if elapsed else 0.0
            remaining = (len(examples) - completed) / rate if rate else 0.0
            width = 24
            filled = int(width * completed / len(examples))
            bar = "#" * filled + "-" * (width - filled)
            print(
                f"[{bar}] {completed}/{len(examples)} "
                f"elapsed={elapsed / 60:.1f}m eta={remaining / 60:.1f}m",
                flush=True,
            )

    records = sorted(_load_existing(result_path, examples).values(), key=lambda record: record["index"])
    valid = [record for record in records if record.get("status") == "ok"]
    statuses: dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1
    metrics = evaluate(
        [record["prediction"] for record in valid],
        [record["reference"] for record in valid],
    )
    metrics.update(
        {
            "model_key": model_key,
            "model_id": model["model_id"],
            "source_id": source_id,
            "condition": condition,
            "expected_examples": len(examples),
            "valid_examples": len(valid),
            "empty_or_invalid_examples": len(examples) - len(valid),
            "status_counts": statuses,
            "initial_refusals": sum(bool(record.get("refusal_retry_prompt")) for record in records),
            "runtime_seconds_this_invocation": time.time() - started,
            "result_file": str(result_path.relative_to(ROOT)),
        }
    )
    write_json(metrics_path, metrics)
    return metrics
