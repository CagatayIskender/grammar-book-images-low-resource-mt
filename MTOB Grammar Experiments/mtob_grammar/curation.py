from __future__ import annotations

from typing import Any

from .chunking import gpt2_encoding
from .config import output_path, source_config
from .io import read_jsonl, write_json


def configured_pages(config: dict[str, Any]) -> list[int]:
    pages: list[int] = []
    for start, end in config["gl_page_ranges"]:
        pages.extend(range(int(start), int(end) + 1))
    return pages


def curate_source(source_id: str) -> dict[str, Any]:
    config = source_config(source_id)
    records = read_jsonl(output_path("grammar_text", source_id, "pages.jsonl"))
    by_page = {int(record["pdf_page"]): record for record in records}
    selected_pages = configured_pages(config)
    missing = [page for page in selected_pages if page not in by_page]
    if missing:
        raise ValueError(f"Missing configured PDF pages for {source_id}: {missing}")
    text = "\n\n".join(by_page[page]["text"] for page in selected_pages if by_page[page]["text"].strip()).strip()
    text += "\n"
    target = output_path("gl_contexts", source_id, "grammar_long.txt")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    token_count = len(gpt2_encoding().encode(text))
    manifest = {
        "source_id": source_id,
        "condition": "Gl",
        "selection_method": "manual page ranges",
        "target_tokens": 100000,
        "actual_gpt2_tokens": token_count,
        "selected_pdf_pages": selected_pages,
        "selected_page_ranges": config["gl_page_ranges"],
        "selection_notes": config["gl_notes"],
        "below_target_reason": (
            "The available translation-relevant source material is shorter than 100K tokens."
            if token_count < 100000
            else None
        ),
    }
    write_json(output_path("manifests", source_id, "gl_selection.json"), manifest)
    return manifest

