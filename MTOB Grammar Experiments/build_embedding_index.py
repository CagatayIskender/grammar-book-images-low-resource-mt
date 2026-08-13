#!/usr/bin/env python3
import argparse

from mtob_grammar.config import sources
from mtob_grammar.retrieval import build_embedding_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build all-mpnet-base-v2 cosine indexes.")
    parser.add_argument("--source", choices=sources())
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all and not args.source:
        parser.error("choose --source or --all")
    selected = sources() if args.all else [args.source]
    for source_id in selected:
        manifest = build_embedding_index(source_id)
        print(f"[OK] {source_id}: {manifest['vectors']} x {manifest['dimensions']}")


if __name__ == "__main__":
    main()
