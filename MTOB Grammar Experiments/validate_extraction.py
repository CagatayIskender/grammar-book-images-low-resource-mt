#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone

from mtob_grammar.config import output_path, sources
from mtob_grammar.io import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and optionally approve extracted PDF text.")
    parser.add_argument("--source", choices=sources())
    parser.add_argument("--approve", action="store_true", help="Mark reviewed extractions as manually audited.")
    args = parser.parse_args()
    selected = [args.source] if args.source else list(sources())
    failed = False
    for source_id in selected:
        manifest_path = output_path("manifests", source_id, "extraction.json")
        pages_path = output_path("grammar_text", source_id, "pages.jsonl")
        if not manifest_path.is_file() or not pages_path.is_file():
            print(f"[ERROR] {source_id}: extraction is missing")
            failed = True
            continue
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        pages = read_jsonl(pages_path)
        empty = [p["pdf_page"] for p in pages if len(p["text"].strip()) < 20]
        samples = sorted({1, max(1, len(pages) // 2), len(pages)})
        print(f"\n[{source_id}] pages={len(pages)} empty_or_short={empty or 'none'}")
        for number in samples:
            preview = " ".join(pages[number - 1]["text"].split())[:240]
            print(f"  p.{number}: {preview}")
        for warning in manifest.get("warnings", []):
            print(f"  warning: {warning}")
        if args.approve:
            manifest["audit_status"] = "manually_reviewed"
            manifest["audit_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["audit_scope"] = "page counts, representative page text, extraction warnings, and Gl page ranges"
            write_json(manifest_path, manifest)
            print("  [APPROVED] extraction audit recorded")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
