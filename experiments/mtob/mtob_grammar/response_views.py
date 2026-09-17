"""Reference-blind, conservative extraction rules frozen before evaluation."""
from __future__ import annotations

import re

PARSER_VERSION = "mtob-response-views-1.0.0"
LABEL = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]+)?(?:\*\*|__)?(?:English translation|Translation)"
    r"(?::(?:\*\*|__)?|(?:\*\*|__):)[ \t]*", re.I | re.M,
)
HEADING = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]+)?(?:\*\*|__)?"
    r"(?:Explanation|Breakdown|Notes|Translation notes)"
    r"(?:(?: of | for | and | on | \()[^:\n]*?)?"
    r"(?::(?:\*\*|__)?|(?:\*\*|__):?)[ \t]*$", re.I | re.M,
)
BARE_HEADING = re.compile(
    r"^[ \t]*(?:#{1,6}[ \t]+)?(?:Explanation|Breakdown|Notes|Translation notes)[ \t]*$",
    re.I | re.M,
)


def trimmed_span(text: str, start: int, end: int) -> tuple[int, int]:
    start += len(text[start:end]) - len(text[start:end].lstrip())
    end = start + len(text[start:end].rstrip())
    return start, end


def response_views(record: dict) -> dict:
    # Generation stores the final refusal-retry response separately from the first response.
    field = next((key for key in ("retry_response", "initial_response", "prediction")
                  if isinstance(record.get(key), str)), None)
    if field is None:
        raise ValueError("No recorded response field")
    text = record[field]
    start, end = trimmed_span(text, 0, len(text))
    if text[start:end].lower().startswith("english translation:"):
        start, end = trimmed_span(text, start + len("english translation:"), end)
    raw = text[start:end]
    result = {
        "parser_version": PARSER_VERSION, "response_field": field,
        "original_response": text, "raw": raw, "extracted": raw,
        "raw_span": [start, end], "extracted_span": [start, end],
        "rule": "preserve", "ambiguous_extraction": False,
        "truncation": "unknown", "explanation_heading_count": 0,
    }
    metadata = record.get("retry_metadata" if field == "retry_response" else "initial_metadata") or {}
    if metadata.get("finish_reason") is not None:
        result["truncation"] = metadata["finish_reason"] == "length"
    if record.get("status") in {"refusal", "empty", "reasoning_violation"}:
        result.update(raw="", extracted="", raw_span=None, extracted_span=None,
                      rule="blank_" + record["status"])
        return result
    labels = list(LABEL.finditer(text))
    headings = sorted([*HEADING.finditer(text), *BARE_HEADING.finditer(text)], key=lambda m: m.start())
    result["explanation_heading_count"] = len(headings)
    if len(labels) > 1:
        result.update(rule="preserve_multiple_translation_labels", ambiguous_extraction=True)
        return result
    lo, hi = start, end
    if labels:
        label = labels[0]
        prefix = text[:label.start()].strip()
        if prefix:
            preceding = [h for h in headings if h.start() < label.start()]
            if not preceding or text[:preceding[0].start()].strip():
                result.update(rule="preserve_unclear_prefix", ambiguous_extraction=True)
                return result
        lo = label.end()
        result["rule"] = "explicit_translation_section" if prefix else "leading_translation_label"
    for heading in headings:
        if heading.start() >= lo:
            hi = heading.start()
            result["rule"] += "+cut_explanation"
            break
    lo, hi = trimmed_span(text, lo, hi)
    if not text[lo:hi] and headings:
        result.update(rule="preserve_explanation_without_translation", ambiguous_extraction=True)
        return result
    result.update(extracted=text[lo:hi], extracted_span=[lo, hi])
    # Unrecognized labelled sections remain untouched, not guessed from sentence length.
    if not headings and re.search(r"(?im)^\s*(?:#{1,6}\s*|\*\*)(?:analysis|translation\b|english translation\b)", text) and not labels:
        result.update(rule="preserve_unrecognized_section", ambiguous_extraction=True)
    return result
