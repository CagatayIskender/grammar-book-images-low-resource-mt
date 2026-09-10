#!/usr/bin/env python3
import argparse

from mtob_grammar.config import sources
from mtob_grammar.curation import curate_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manually selected long grammar contexts.")
    parser.add_argument("--source", choices=sources())
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all and not args.source:
        parser.error("choose --source or --all")
    selected = sources() if args.all else [args.source]
    for source_id in selected:
        manifest = curate_source(source_id)
        print(f"[OK] {source_id}: {manifest['actual_gpt2_tokens']} GPT-2 tokens")


if __name__ == "__main__":
    main()
