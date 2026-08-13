from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TranslationExample:
    index: int
    source: str
    reference: str


def parse_igt(path: Path, limit: int | None = None) -> list[TranslationExample]:
    text = path.read_text(encoding="utf-8", errors="strict")
    blocks = [block for block in re.split(r"\n\s*\n", text) if block.strip()]
    examples: list[TranslationExample] = []
    for block in blocks:
        source = reference = None
        for line in block.splitlines():
            if line.startswith("\\t"):
                source = line[2:].strip()
            elif line.startswith("\\l"):
                reference = line[2:].strip().split("||", 1)[0].strip()
        if source and reference:
            examples.append(TranslationExample(len(examples), source, reference))
            if limit and len(examples) >= limit:
                break
    return examples

