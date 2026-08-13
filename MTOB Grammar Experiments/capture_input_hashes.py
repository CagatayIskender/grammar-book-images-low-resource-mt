#!/usr/bin/env python3
import argparse
from datetime import datetime, timezone

from mtob_grammar.config import input_path, output_path, sources
from mtob_grammar.io import sha256, write_json


def current_hashes() -> dict:
    files = {}
    for config in sources().values():
        for key in ("pdf", "test_file"):
            path = input_path(config[key])
            files[str(path)] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": dict(sorted(files.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture or verify immutable input hashes.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--baseline", action="store_true")
    group.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    target = output_path("manifests", "input_hashes_baseline.json")
    current = current_hashes()
    if args.baseline:
        if target.exists():
            raise RuntimeError(f"Baseline already exists and will not be overwritten: {target}")
        write_json(target, current)
        print(f"[OK] Captured {len(current['files'])} immutable inputs")
        return
    if not target.is_file():
        raise FileNotFoundError("Capture --baseline before verification")
    import json

    baseline = json.loads(target.read_text(encoding="utf-8"))
    if baseline["files"] != current["files"]:
        raise RuntimeError("Immutable input hashes changed")
    print(f"[OK] All {len(current['files'])} input hashes match the baseline")


if __name__ == "__main__":
    main()
