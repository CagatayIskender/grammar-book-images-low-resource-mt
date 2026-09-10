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
REASONING_RETRY_LIMIT = 3
REFUSAL_RETRY_LIMIT = 3


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


def _generate_without_reasoning(backend, prompt: str, seed: int):
    violations: list[dict[str, Any]] = []
    last_violation: ReasoningViolation | None = None
    for attempt in range(REASONING_RETRY_LIMIT):
        try:
            generation = backend.generate(prompt, seed=seed + attempt * 10_000)
            return generation, violations
        except ReasoningViolation as exc:
            last_violation = exc
            violations.append(
                {
                    "attempt": attempt + 1,
                    "seed": seed + attempt * 10_000,
                    "response": exc.text,
                    "metadata": exc.metadata,
                    "violation": str(exc),
                }
            )
    assert last_violation is not None
    metadata = dict(last_violation.metadata)
    metadata["reasoning_attempts"] = violations
    raise ReasoningViolation(
        str(last_violation),
        text=last_violation.text,
        metadata=metadata,
    )


def run(
    model_key: str,
    source_id: str,
    condition: str,
    smoke: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    condition = condition.casefold()
    if condition not in CONDITIONS:
        raise ValueError(f"Condition must be one of {', '.join(CONDITIONS)}")
    source = source_config(source_id)
    model = model_config(model_key)
    configured_examples = int(source["test_n"])
    if limit is not None and (limit < 1 or limit > configured_examples):
        raise ValueError(f"limit must be between 1 and {configured_examples}")
    example_limit = 2 if smoke else (limit if limit is not None else configured_examples)
    examples = parse_igt(input_path(source["test_file"]), limit=example_limit)
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
            previous = existing.get(example.index)
            rerun_round = int(previous.get("rerun_round", 0)) + 1 if previous else 0
            seed_offset = rerun_round * 100_000
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
                initial, initial_reasoning_retries = _generate_without_reasoning(
                    backend,
                    prompt,
                    seed=2024 + example.index + seed_offset,
                )
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
                    "initial_response": exc.text,
                    "initial_metadata": exc.metadata,
                    "rerun_round": rerun_round,
                }
                handle.write(json.dumps(violation, ensure_ascii=False) + "\n")
                handle.flush()
                continue
            final = initial
            retry_prompt = None
            retry_reasoning_retries: list[dict[str, Any]] = []
            refusal_attempts: list[dict[str, Any]] = []
            retry_reasoning_failed = False
            if is_refusal(initial.text):
                retry_prompt = appendix_c_prompt(
                    source["language"], source["location"], example.source, context, refusal=True
                )
                for refusal_attempt in range(REFUSAL_RETRY_LIMIT):
                    try:
                        candidate, reasoning_retries = _generate_without_reasoning(
                            backend,
                            retry_prompt,
                            seed=(
                                3024
                                + example.index
                                + seed_offset
                                + refusal_attempt * 100_000
                            ),
                        )
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
                            "initial_response": initial.text,
                            "initial_metadata": initial.metadata,
                            "initial_reasoning_retries": initial_reasoning_retries,
                            "refusal_retry_prompt": retry_prompt,
                            "retry_response": exc.text,
                            "retry_metadata": exc.metadata,
                            "refusal_attempts": refusal_attempts,
                            "rerun_round": rerun_round,
                        }
                        handle.write(json.dumps(violation, ensure_ascii=False) + "\n")
                        handle.flush()
                        retry_reasoning_failed = True
                        break
                    retry_reasoning_retries.extend(reasoning_retries)
                    final = candidate
                    refusal_attempts.append(
                        {
                            "attempt": refusal_attempt + 1,
                            "response": candidate.text,
                            "metadata": candidate.metadata,
                        }
                    )
                    if not is_refusal(candidate.text):
                        break
                if retry_reasoning_failed:
                    continue
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
                "initial_reasoning_retries": initial_reasoning_retries,
                "refusal_retry_prompt": retry_prompt,
                "retry_response": final.text if retry_prompt else None,
                "retry_metadata": final.metadata if retry_prompt else None,
                "retry_reasoning_retries": retry_reasoning_retries,
                "refusal_attempts": refusal_attempts,
                "rerun_round": rerun_round,
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
    with result_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
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
            "configured_examples": configured_examples,
            "requested_limit": example_limit,
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
