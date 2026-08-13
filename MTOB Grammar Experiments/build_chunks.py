#!/usr/bin/env python3
import argparse
import json

from mtob_grammar.chunking import build_source_chunks
from mtob_grammar.config import output_path, sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Create exact GPT-2 512/256 grammar chunks.")
    parser.add_argument("--source", choices=sources())
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all and not args.source:
        parser.error("choose --source or --all")
    selected = sources() if args.all else [args.source]
    for source_id in selected:
        audit = json.loads(output_path("manifests", source_id, "extraction.json").read_text(encoding="utf-8"))
        if audit.get("audit_status") != "manually_reviewed":
            raise RuntimeError(f"Extraction for {source_id} has not been manually reviewed")
        manifest = build_source_chunks(source_id)
        print(f"[OK] {source_id}: {manifest['total_tokens']} tokens, {manifest['chunk_count']} chunks")


if __name__ == "__main__":
    main()
