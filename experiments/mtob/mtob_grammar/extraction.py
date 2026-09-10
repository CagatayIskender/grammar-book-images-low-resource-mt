from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import fitz

from .config import input_path, output_path, source_config
from .io import sha256, write_json, write_jsonl


def _normalized_line(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip())).casefold()


def _page_lines(page: fitz.Page) -> list[str]:
    return [re.sub(r"[ \t]+", " ", line).strip() for line in page.get_text("text").splitlines()]


def _repeated_marginal_lines(raw_pages: list[list[str]]) -> set[str]:
    candidates: Counter[str] = Counter()
    for lines in raw_pages:
        nonempty = [line for line in lines if line]
        for line in nonempty[:2] + nonempty[-2:]:
            normalized = _normalized_line(line)
            if len(normalized) >= 8:
                candidates[normalized] += 1
    threshold = max(3, len(raw_pages) // 4)
    return {line for line, count in candidates.items() if count >= threshold}


def extract_source(source_id: str) -> dict[str, Any]:
    config = source_config(source_id)
    pdf_path = input_path(config["pdf"])
    document = fitz.open(pdf_path)
    raw_pages = [_page_lines(page) for page in document]
    repeated = _repeated_marginal_lines(raw_pages)
    page_records = []
    grammar_parts: list[str] = []
    cursor = 0
    warnings: list[str] = []

    for page_number, lines in enumerate(raw_pages, start=1):
        kept: list[str] = []
        removed: list[str] = []
        for index, line in enumerate(lines):
            if not line:
                if kept and kept[-1] != "":
                    kept.append("")
                continue
            normalized = _normalized_line(line)
            is_margin = index < 2 or index >= max(0, len(lines) - 2)
            if is_margin and normalized in repeated:
                removed.append(line)
                continue
            cleaned = line.rstrip()
            if cleaned:
                kept.append(cleaned)
        text = "\n".join(kept).strip()
        if len(text) < 20:
            warnings.append(f"PDF page {page_number} has little or no extracted text ({len(text)} characters).")
        if grammar_parts:
            grammar_parts.append("\n\n")
            cursor += 2
        start = cursor
        grammar_parts.append(text)
        cursor += len(text)
        page_records.append(
            {
                "source_id": source_id,
                "pdf_page": page_number,
                "char_start": start,
                "char_end": cursor,
                "text": text,
                "removed_repeated_margin_lines": removed,
            }
        )

    grammar_text = "".join(grammar_parts).strip() + "\n"
    text_path = output_path("grammar_text", source_id, "grammar.txt")
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(grammar_text, encoding="utf-8")
    write_jsonl(output_path("grammar_text", source_id, "pages.jsonl"), page_records)
    manifest = {
        "source_id": source_id,
        "language": config["language"],
        "location_from_pdf": config["location"],
        "input_pdf": str(pdf_path),
        "input_pdf_sha256": sha256(pdf_path),
        "pdf_pages": len(document),
        "extracted_characters": len(grammar_text),
        "repeated_margin_patterns_removed": sorted(repeated),
        "extraction_policy": "preserve embedded PDF text; remove only repeated top/bottom marginal lines",
        "warnings": warnings,
        "audit_status": "requires_manual_review",
    }
    write_json(output_path("manifests", source_id, "extraction.json"), manifest)
    return manifest


def page_numbers_for_span(pages: list[dict[str, Any]], start: int, end: int) -> list[int]:
    return [
        int(page["pdf_page"])
        for page in pages
        if int(page["char_end"]) > start and int(page["char_start"]) < end
    ]
