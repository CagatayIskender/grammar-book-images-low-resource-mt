#!/usr/bin/env python3
import argparse

from mtob_grammar.config import input_path, output_path, source_config, sources
from mtob_grammar.data import parse_igt
from mtob_grammar.io import write_jsonl
from mtob_grammar.retrieval import load_chunks, retrieve_ge, retrieve_gs


def build(source_id: str) -> None:
    config = source_config(source_id)
    examples = parse_igt(input_path(config["test_file"]), limit=int(config["test_n"]))
    chunks = load_chunks(source_id)
    for condition in ("ge", "gs"):
        records = []
        for example in examples:
            passages = (
                retrieve_ge(example.source, chunks, source_id, 2, query_index=example.index)
                if condition == "ge"
                else retrieve_gs(example.source, chunks, 2)
            )
            records.append(
                {
                    "index": example.index,
                    "source": example.source,
                    "source_id": source_id,
                    "condition": condition,
                    "k": 2,
                    "passages": passages,
                }
            )
        write_jsonl(output_path("manifests", source_id, f"retrieval_{condition}.jsonl"), records)
        print(f"[OK] {source_id}/{condition}: {len(records)} queries x 2 passages")


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze complete Ge and Gs retrieval manifests.")
    parser.add_argument("--source", choices=sources())
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all and not args.source:
        parser.error("choose --source or --all")
    for source_id in (sources() if args.all else [args.source]):
        build(source_id)


if __name__ == "__main__":
    main()
