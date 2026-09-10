#!/usr/bin/env python3
import argparse

from mtob_grammar.config import sources
from mtob_grammar.extraction import extract_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract page-mapped plaintext from source PDFs.")
    parser.add_argument("--source", choices=sources())
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all and not args.source:
        parser.error("choose --source or --all")
    selected = sources() if args.all else [args.source]
    for source_id in selected:
        manifest = extract_source(source_id)
        print(f"[OK] {source_id}: {manifest['pdf_pages']} pages, {manifest['extracted_characters']} characters")


if __name__ == "__main__":
    main()
