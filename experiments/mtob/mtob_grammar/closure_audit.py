"""Offline validation and immutable provenance for the 30 Qwen conditions."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from collections import Counter
from pathlib import Path

from .config import ROOT, input_path, models, sources
from .data import parse_igt
from .io import read_jsonl, sha256
from .prompts import appendix_c_prompt, extract_translation, long_context, passage_context
from .response_views import PARSER_VERSION, response_views

OUT = ROOT / "evaluation_v1"
MODELS = ("qwen3", "qwen35")
CONDITIONS = ("ge", "gs", "gl")
SETTINGS = {
    "version": "mtob-evaluation-1.0.0", "parser_version": PARSER_VERSION,
    "denominator": "all expected test records; non-ok predictions blank",
    "punctuation": "remove ' =' then all ASCII punctuation",
    "bleu": {"tokenize": "13a", "smooth_method": "none", "effective_order": False, "scale": "0-1 (bootstrap 0-100)"},
    "chrf": {"char_order": 6, "word_order": 0, "beta": 2, "scale": "0-100"},
    "rouge": "local macro F1; ASCII alphanumeric tokens; rougeLsum equals rougeL, not official summary-level ROUGE",
    "character": "cer package CharacTER: word shifts plus character edits; hypothesis-length normalization, capped at 1; empty=1; not standard CER",
    "bootstrap": {"samples": 100000, "seed": 20260917, "family_size": 120, "correction": "Holm", "alpha": 0.05,
                  "zero_effect_policy": "absolute observed difference <= 1e-12: p_raw=1; preserve unmodified package p as p_sacrebleu"},
}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save(path: Path, value, *, text: bool = False) -> None:
    path = path.resolve()
    if OUT.resolve() not in path.parents:
        raise ValueError(f"Write outside evaluation_v1 rejected: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(value if text else json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def matrix():
    return [(m, s, c) for m in MODELS for s in sources() for c in CONDITIONS]


def result_path(model, source, condition):
    return ROOT / "results" / model / source / f"results_{condition}.jsonl"


def artifact(kind, model, source, condition):
    return OUT / kind / model / source / f"{condition}.json"


def input_hashes() -> dict:
    paths = set()
    for folder in ("configs", "grammar_text", "chunks", "indexes", "gl_contexts"):
        paths.update(p for p in (ROOT / folder).rglob("*") if p.is_file())
    for source, config in sources().items():
        paths.update(p for p in (ROOT / "manifests" / source).rglob("*") if p.is_file())
        paths.update([input_path(config["pdf"]), input_path(config["test_file"])])
    paths.update((ROOT / "manifests" / "thinking").glob("*.json"))
    for model, source, condition in matrix():
        paths.add(result_path(model, source, condition))
        paths.add(ROOT / "metrics" / model / source / f"metrics_{condition}.json")
    return {os.path.relpath(p, ROOT): sha256(p) for p in sorted(paths)}


def implementation_hashes() -> dict:
    names = ("closure_audit.py", "closure_evaluation.py", "response_views.py", "evaluation.py", "prompts.py", "data.py", "chunking.py")
    hashes = {name: sha256(ROOT / "mtob_grammar" / name) for name in names}
    hashes["finalize_results.py"] = sha256(ROOT / "finalize_results.py")
    for name in ("vocab.bpe", "encoder.json"):
        hashes[f"tokenizer/{name}"] = sha256(OUT / "tokenizer" / name)
    from importlib.util import find_spec
    for name in ("cer", "sacrebleu"):
        package = Path(find_spec(name).origin).parent
        hashes.update({f"{name}/{p.relative_to(package)}": sha256(p) for p in sorted(package.rglob("*.py"))})
    return hashes


def provenance() -> dict:
    value = {"inputs": input_hashes(), "implementation": implementation_hashes(), "settings": SETTINGS,
             "versions": {p: importlib.metadata.version(p) for p in ("sacrebleu", "numpy", "tiktoken")}}
    return {**value, "fingerprint": digest(value)}


def validate_rows(rows, examples, model, source, condition) -> tuple[str, list[str]]:
    errors = []
    seen = set()
    expected = {e.index: e for e in examples}
    for r in rows:
        if not isinstance(r, dict):
            errors.append("Result record is not an object")
            continue
        index = r.get("index")
        if type(index) is not int or index not in expected or index in seen:
            errors.append(f"Invalid/duplicate index: {index!r}")
            continue
        seen.add(index)
        e = expected[index]
        for key, value in (("source", e.source), ("reference", e.reference), ("model_key", model),
                           ("source_id", source), ("condition", condition), ("model_id", models()[model]["model_id"])):
            if r.get(key) != value:
                errors.append(f"index {index}: {key} mismatch")
        if r.get("status") not in {"ok", "refusal", "empty", "reasoning_violation"}:
            errors.append(f"index {index}: unsupported status")
    return ("invalid" if errors else "complete" if seen == set(expected) else "partial"), errors


def offline_encoding():
    import tiktoken
    import tiktoken_ext.openai_public
    from unittest.mock import patch

    def local_read(url, expected_hash=None):
        path = OUT / "tokenizer" / url.rsplit("/", 1)[-1]
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError(f"Tokenizer checksum mismatch: {path.name}")
        return data

    # Use tiktoken's own constructor and checksum constants without its download/cache path.
    with patch("tiktoken.load.read_file_cached", local_read):
        return tiktoken.Encoding(**tiktoken_ext.openai_public.gpt2())


def prepared_source(source, config):
    enc = offline_encoding()
    examples = parse_igt(input_path(config["test_file"]))
    if len(examples) != config["test_n"]:
        raise ValueError("Test count differs from source configuration")
    text = (ROOT / "grammar_text" / source / "grammar.txt").read_text()
    tokens = enc.encode(text)
    chunks = read_jsonl(ROOT / "chunks" / source / "chunks_512_overlap256.jsonl")
    from .extraction import page_numbers_for_span
    decoded, offsets = enc.decode_with_offsets(tokens)
    if decoded != text:
        raise ValueError("Grammar text did not round-trip through GPT-2")
    offsets = [*offsets, len(text)]
    page_records = read_jsonl(ROOT / "grammar_text" / source / "pages.jsonl")
    expected_starts = []
    for start in range(0, len(tokens), 256):
        expected_starts.append(start)
        if start + 512 >= len(tokens):
            break
    if len(chunks) != len(expected_starts):
        raise ValueError("Unexpected chunk count")
    for i, (chunk, start) in enumerate(zip(chunks, expected_starts, strict=True)):
        end = min(start + 512, len(tokens))
        for key, value in {"chunk_index": i, "chunk_id": f"{source}-chunk-{i:05d}", "token_start": start,
                           "token_end": end, "token_count": end - start, "text": enc.decode(tokens[start:end]),
                           "char_start": offsets[start], "char_end": offsets[end],
                           "pdf_pages": page_numbers_for_span(page_records, offsets[start], offsets[end])}.items():
            if chunk[key] != value:
                raise ValueError(f"Chunk {i}: {key} differs from exact 512/256 reconstruction")
    by_id = {c["chunk_id"]: c for c in chunks}
    retrieval = {}
    for condition in ("ge", "gs"):
        rows = read_jsonl(ROOT / "manifests" / source / f"retrieval_{condition}.jsonl")
        if len(rows) != len(examples) or {r["index"] for r in rows} != set(range(len(examples))):
            raise ValueError("Incomplete/duplicate retrieval indices")
        retrieval[condition] = {r["index"]: r for r in rows}
        for e in examples:
            row = retrieval[condition][e.index]
            if (row["source"], row["source_id"], row["condition"], row["k"]) != (e.source, source, condition, 2):
                raise ValueError("Retrieval identity mismatch")
            passages = row["passages"]
            if len(passages) != 2 or len({p["chunk_id"] for p in passages}) != 2:
                raise ValueError("Retrieval must contain exactly two distinct chunks")
            for passage in passages:
                chunk = by_id[passage["chunk_id"]]
                if any(passage[k] != v for k, v in chunk.items()):
                    raise ValueError("Retrieved passage differs from frozen chunk")
                expected_method = "embedding" if condition == "ge" else "lcs"
                if passage["retrieval_method"] != expected_method:
                    raise ValueError("Retrieval method mismatch")
    gl = (ROOT / "gl_contexts" / source / "grammar_long.txt").read_text()
    gl_manifest = load(ROOT / "manifests" / source / "gl_selection.json")
    pages = {r["pdf_page"]: r["text"] for r in read_jsonl(ROOT / "grammar_text" / source / "pages.jsonl")}
    selected = [p for a, b in config["gl_page_ranges"] for p in range(a, b + 1)]
    expected_gl = "\n\n".join(pages[p] for p in selected if pages[p].strip()).strip() + "\n"
    gl_tokens = len(enc.encode(gl))
    if gl != expected_gl or selected != gl_manifest["selected_pdf_pages"] or gl_tokens != gl_manifest["actual_gpt2_tokens"]:
        raise ValueError("Gl selection/text/token count mismatch")
    return examples, retrieval, gl, {"gl_tokens": gl_tokens, "gl_pages": selected, "chunks": len(chunks), "total_tokens": len(tokens)}


def audit() -> dict:
    frozen = provenance()
    existing = OUT / "provenance.json"
    if existing.exists() and load(existing)["inputs"] != frozen["inputs"]:
        raise ValueError("Original inputs changed since evaluation_v1 was initialized; refusing reuse")
    save(existing, frozen)
    report = {"fingerprint": frozen["fingerprint"], "conditions": [], "sources": {}, "errors": [],
              "notes": ["No new generation. Earlier rounds not saved in final JSONL cannot be recovered.",
                        "Extraction and retrieval quality are not re-judged; original preparation artifacts are validated.",
                        "Finish reason/output-token counts absent in old records: truncation unknown.",
                        "Prompt text is reconstructed exactly; actual historic device/runtime settings not recorded cannot be certified."]}
    prepared = {}
    for model in MODELS:
        thinking = load(ROOT / "manifests" / "thinking" / f"{model}.json")
        if thinking.get("enable_thinking") is not False or thinking.get("single_user_message") is not True or thinking.get("think_content_empty") is not True:
            report["errors"].append(f"{model}: thinking/single-user-message preparation check failed")
    for source, config in sources().items():
        try:
            prepared[source] = prepared_source(source, config)
            report["sources"][source] = prepared[source][3]
        except Exception as exc:
            report["errors"].append(f"{source}: {type(exc).__name__}: {exc}")
    for model, source, condition in matrix():
        item = {"model": model, "source": source, "condition": condition, "expected": sources()[source]["test_n"],
                "state": "invalid", "errors": [], "count": 0}
        try:
            rows = read_jsonl(result_path(model, source, condition))
            item["count"] = len(rows)
            examples, retrieval, gl, _ = prepared[source]
            item["state"], item["errors"] = validate_rows(rows, examples, model, source, condition)
            if item["state"] != "complete":
                raise ValueError("Record cohort is not complete and valid")
            config = sources()[source]
            records = []
            for r in sorted(rows, key=lambda r: r["index"]):
                context = long_context(config["language"], gl) if condition == "gl" else passage_context(
                    config["language"], [p["text"] for p in retrieval[condition][r["index"]]["passages"]])
                args = (config["language"], config["location"], r["source"], context)
                if r["prompt"] != appendix_c_prompt(*args):
                    raise ValueError(f"index {r['index']}: Appendix C prompt mismatch")
                if r["prompt"].count(r["source"]) != 2:
                    raise ValueError(f"index {r['index']}: source sentence must occur exactly twice")
                if r.get("refusal_retry_prompt") and r["refusal_retry_prompt"] != appendix_c_prompt(*args, refusal=True):
                    raise ValueError(f"index {r['index']}: refusal prompt mismatch")
                if condition != "gl" and "retrieved_passages" in r:
                    passages = retrieval[condition][r["index"]]["passages"]
                    keys = ("chunk_id", "chunk_index", "token_start", "token_end", "pdf_pages", "retrieval_score", "retrieval_method")
                    if r["retrieved_passages"] != [{k: p[k] for k in keys} for p in passages]:
                        raise ValueError(f"index {r['index']}: saved retrieval mismatch")
                for key in ("initial_metadata", "retry_metadata"):
                    meta = r.get(key)
                    if meta and (meta.get("thinking", {}).get("enable_thinking") is not False or meta.get("temperature") != models()[model]["temperature"] or meta.get("model") != models()[model]["model_id"]):
                        raise ValueError(f"index {r['index']}: recorded thinking/temperature mismatch")
                view = response_views(r)
                if r["status"] == "ok" and r["prediction"] != extract_translation(view["original_response"]):
                    raise ValueError(f"index {r['index']}: saved prediction differs from final response")
                records.append({"index": r["index"], "source": r["source"], "reference": r["reference"],
                                "status": r["status"], **view})
            item.update(status_counts=dict(Counter(r["status"] for r in rows)),
                        ambiguous=sum(r["ambiguous_extraction"] for r in records),
                        changed=sum(r["raw"] != r["extracted"] for r in records),
                        refusal_retry_records=sum(bool(r.get("refusal_retry_prompt")) for r in rows),
                        saved_refusal_attempts=sum(len(r.get("refusal_attempts", [])) for r in rows),
                        saved_reasoning_attempts=sum(len(r.get("initial_reasoning_retries", [])) + len(r.get("retry_reasoning_retries", [])) +
                                                    sum(len((r.get(k) or {}).get("reasoning_attempts", [])) for k in ("initial_metadata", "retry_metadata")) for r in rows),
                        rerun_records=[{"index": r["index"], "round": r["rerun_round"], "previous_round_response": "not retained in this result file"}
                                       for r in rows if r.get("rerun_round", 0)],
                        unknown_truncation=sum(r["truncation"] == "unknown" for r in records))
            save(artifact("records", model, source, condition), {"fingerprint": frozen["fingerprint"], "records": records})
        except Exception as exc:
            item["errors"].append(f"{type(exc).__name__}: {exc}")
            if item["state"] != "partial":
                item["state"] = "invalid"
        report["conditions"].append(item)
    report["total_records"] = sum(c["count"] for c in report["conditions"])
    if report["total_records"] != 4230:
        report["errors"].append("Expected exactly 4230 records")
    report["original_hashes_unchanged"] = frozen["inputs"] == input_hashes()
    report["passed"] = not report["errors"] and report["original_hashes_unchanged"] and all(c["state"] == "complete" for c in report["conditions"])
    save(OUT / "audit.json", report)
    print(json.dumps({"passed": report["passed"], "records": report["total_records"], "errors": report["errors"],
                      "invalid": [c for c in report["conditions"] if c["state"] != "complete"]}, indent=2), flush=True)
    return report


def require_audit() -> dict:
    report = load(OUT / "audit.json")
    current = provenance()
    if not report["passed"] or current["fingerprint"] != report["fingerprint"]:
        raise ValueError("Audit failed or inputs/rules/settings changed; run audit again")
    return current
