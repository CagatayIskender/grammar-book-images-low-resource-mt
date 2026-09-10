from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pylcs
import torch
from transformers import AutoModel, AutoTokenizer

from .config import ROOT, input_path, output_path, source_config
from .data import parse_igt
from .io import read_jsonl, sha256, write_json


EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_MAX_SEQUENCE_LENGTH = 384


def lcs_length(query: str, candidate: str) -> int:
    return int(pylcs.lcs_string_length(query, candidate))


def retrieve_gs(query: str, chunks: list[dict[str, Any]], k: int = 2) -> list[dict[str, Any]]:
    scores = np.asarray(pylcs.lcs_string_of_list(query, [chunk["text"] for chunk in chunks]))
    indices = np.argpartition(scores, -k)[-k:].tolist()
    return [
        {**chunks[index], "retrieval_score": int(scores[index]), "retrieval_method": "lcs"}
        for index in indices
    ]


@lru_cache(maxsize=1)
def embedding_model():
    cache = output_path("cache", "huggingface", "hub")
    cache.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL, cache_dir=str(cache), local_files_only=True
    )
    model = AutoModel.from_pretrained(
        EMBEDDING_MODEL, cache_dir=str(cache), local_files_only=True
    ).eval().to("cpu")
    return tokenizer, model


def encode_texts(texts: list[str], batch_size: int = 16) -> np.ndarray:
    tokenizer, model = embedding_model()
    batches: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=EMBEDDING_MAX_SEQUENCE_LENGTH,
            return_tensors="pt",
        )
        with torch.no_grad():
            hidden = model(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        batches.append(normalized.cpu().numpy().astype(np.float32))
    return np.concatenate(batches, axis=0)


def build_embedding_index(source_id: str) -> dict[str, Any]:
    chunks_path = output_path("chunks", source_id, "chunks_512_overlap256.jsonl")
    chunks = read_jsonl(chunks_path)
    index_path = output_path("indexes", source_id, "all_mpnet_base_v2.npy")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = output_path("manifests", source_id, "embedding_index.json")
    old_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    )
    chunk_sha256 = sha256(chunks_path)
    chunk_index_current = (
        index_path.is_file()
        and old_manifest.get("chunk_sha256") == chunk_sha256
        and old_manifest.get("embedding_max_sequence_length") == EMBEDDING_MAX_SEQUENCE_LENGTH
        and len(np.load(index_path, mmap_mode="r")) == len(chunks)
    )
    if chunk_index_current:
        embeddings = np.load(index_path)
    else:
        embeddings = encode_texts([chunk["text"] for chunk in chunks])
        np.save(index_path, embeddings)
    config = source_config(source_id)
    test_path = input_path(config["test_file"])
    test_sha256 = sha256(test_path)
    examples = parse_igt(test_path, limit=int(config["test_n"]))
    test_index_path = output_path("indexes", source_id, "test_all_mpnet_base_v2.npy")
    test_index_current = (
        test_index_path.is_file()
        and old_manifest.get("test_file_sha256") == test_sha256
        and old_manifest.get("embedding_max_sequence_length") == EMBEDDING_MAX_SEQUENCE_LENGTH
        and len(np.load(test_index_path, mmap_mode="r")) == len(examples)
    )
    if test_index_current:
        test_embeddings = np.load(test_index_path)
    else:
        test_embeddings = encode_texts([example.source for example in examples])
        np.save(test_index_path, test_embeddings)
    write_json(
        output_path("indexes", source_id, "test_sources.json"),
        {"sources": [example.source for example in examples]},
    )
    manifest = {
        "source_id": source_id,
        "model": EMBEDDING_MODEL,
        "similarity": "cosine_on_normalized_embeddings",
        "pooling": "attention-mask mean pooling followed by L2 normalization",
        "embedding_max_sequence_length": EMBEDDING_MAX_SEQUENCE_LENGTH,
        "vectors": int(embeddings.shape[0]),
        "dimensions": int(embeddings.shape[1]),
        "test_vectors": int(test_embeddings.shape[0]),
        "chunk_file": str(chunks_path.relative_to(ROOT)),
        "chunk_sha256": chunk_sha256,
        "test_file_sha256": test_sha256,
    }
    write_json(manifest_path, manifest)
    return manifest


def retrieve_ge(
    query: str,
    chunks: list[dict[str, Any]],
    source_id: str,
    k: int = 2,
    query_index: int | None = None,
) -> list[dict[str, Any]]:
    embeddings = np.load(output_path("indexes", source_id, "all_mpnet_base_v2.npy"))
    if len(chunks) != len(embeddings):
        raise ValueError(f"Embedding/chunk count mismatch for {source_id}")
    if query_index is None:
        query_embedding = encode_texts([query], batch_size=1)[0]
    else:
        sources = json.loads(
            output_path("indexes", source_id, "test_sources.json").read_text(encoding="utf-8")
        )["sources"]
        if query_index >= len(sources) or sources[query_index] != query:
            raise ValueError(f"Precomputed query embedding mismatch for {source_id} index {query_index}")
        query_embedding = np.load(
            output_path("indexes", source_id, "test_all_mpnet_base_v2.npy"), mmap_mode="r"
        )[query_index]
    scores = embeddings @ query_embedding
    order = sorted(range(len(chunks)), key=lambda index: (-float(scores[index]), index))[:k]
    return [
        {**chunks[index], "retrieval_score": float(scores[index]), "retrieval_method": "embedding"}
        for index in order
    ]


def load_chunks(source_id: str) -> list[dict[str, Any]]:
    return read_jsonl(output_path("chunks", source_id, "chunks_512_overlap256.jsonl"))
