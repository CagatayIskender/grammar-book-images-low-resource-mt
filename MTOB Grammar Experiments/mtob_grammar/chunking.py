from __future__ import annotations

from functools import lru_cache
from typing import Any

import tiktoken

from .config import output_path
from .extraction import page_numbers_for_span
from .io import read_jsonl, write_json, write_jsonl


CHUNK_SIZE = 512
CHUNK_OVERLAP = 256
CHUNK_STEP = CHUNK_SIZE - CHUNK_OVERLAP


@lru_cache(maxsize=1)
def gpt2_encoding():
    return tiktoken.get_encoding("gpt2")


def token_char_offsets(text: str, token_ids: list[int]) -> list[int]:
    encoding = gpt2_encoding()
    decoded, starts = encoding.decode_with_offsets(token_ids)
    if decoded != text:
        raise ValueError("GPT-2 token round-trip changed extracted grammar text")
    return [*starts, len(text)]


def build_source_chunks(source_id: str) -> dict[str, Any]:
    text_path = output_path("grammar_text", source_id, "grammar.txt")
    pages_path = output_path("grammar_text", source_id, "pages.jsonl")
    if not text_path.is_file() or not pages_path.is_file():
        raise FileNotFoundError(f"Extract {source_id} before building chunks")
    text = text_path.read_text(encoding="utf-8")
    pages = read_jsonl(pages_path)
    encoding = gpt2_encoding()
    token_ids = encoding.encode(text)
    offsets = token_char_offsets(text, token_ids)
    chunks = []
    for index, token_start in enumerate(range(0, len(token_ids), CHUNK_STEP)):
        token_end = min(token_start + CHUNK_SIZE, len(token_ids))
        if token_end <= token_start:
            break
        char_start = offsets[token_start]
        char_end = offsets[token_end]
        chunk_text = encoding.decode(token_ids[token_start:token_end])
        chunks.append(
            {
                "chunk_id": f"{source_id}-chunk-{index:05d}",
                "chunk_index": index,
                "token_start": token_start,
                "token_end": token_end,
                "token_count": token_end - token_start,
                "char_start": char_start,
                "char_end": char_end,
                "pdf_pages": page_numbers_for_span(pages, char_start, char_end),
                "text": chunk_text,
            }
        )
        if token_end == len(token_ids):
            break
    write_jsonl(output_path("chunks", source_id, "chunks_512_overlap256.jsonl"), chunks)
    manifest = {
        "source_id": source_id,
        "tokenizer": "tiktoken:gpt2",
        "total_tokens": len(token_ids),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunk_step": CHUNK_STEP,
        "chunk_count": len(chunks),
    }
    write_json(output_path("manifests", source_id, "chunks.json"), manifest)
    return manifest


def verify_chunks(chunks: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for index, chunk in enumerate(chunks):
        count = int(chunk["token_count"])
        if count > CHUNK_SIZE:
            errors.append(f"Chunk {index} has {count} tokens")
        if index:
            previous = chunks[index - 1]
            overlap = int(previous["token_end"]) - int(chunk["token_start"])
            if count == CHUNK_SIZE and overlap != CHUNK_OVERLAP:
                errors.append(f"Chunks {index - 1}/{index} overlap by {overlap}, expected {CHUNK_OVERLAP}")
    return errors
