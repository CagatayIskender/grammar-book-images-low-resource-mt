"""Shared, dependency-free provenance and output validation."""
from pathlib import Path
import hashlib
import json
import os
import re

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "Database/2023glossingST/data"
PREFIX = {"Gitksan": "git", "Natugu": "ntu", "Lezgi": "lez", "Tsez": "ddo"}
COUNTS = {"Gitksan": 37, "Natugu": 99, "Lezgi": 87, "Tsez": 445}


CHAIN_PROMPT_REVISION = "gloss_prompt_v2"


def preflight_stem(cfg, attention):
    suffix = f"_{CHAIN_PROMPT_REVISION}" if cfg["condition"] == "chain_gloss" else ""
    return f"{cfg['id']}_{attention}{suffix}"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def dataset(language, split="test"):
    tail = "test-track1-uncovered" if split == "test" else "train-track1-covered"
    name = f"{PREFIX[language]}-{tail}" + (".txt" if language == "Gitksan" else "")
    text = (DATA / language / name).read_text(encoding="utf-8")
    records = []
    for block in re.split(r"\n\s*\n", text):
        fields = {}
        for line in block.splitlines():
            line = line.strip()
            for marker, key in (("\\t", "source"), ("\\g", "gloss"), ("\\l", "reference")):
                if line.startswith(marker):
                    fields[key] = line[2:].strip()
        if fields.get("source") and fields.get("reference"):
            fields["reference"] = fields["reference"].split("||")[0].strip()
            records.append(fields)
    if split == "test" and len(records) != COUNTS[language]:
        raise ValueError(f"Unexpected test count for {language}: {len(records)}")
    return records


def read_records(path):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def validate_records(records, expected, complete=False, fingerprint=None):
    if len(records) > len(expected) or (complete and len(records) != len(expected)):
        raise ValueError(f"Incomplete or oversized output: {len(records)}/{len(expected)}")
    for i, (row, ex) in enumerate(zip(records, expected)):
        if row.get("idx") != i or row.get("source") != ex["source"] or row.get("reference") != ex["reference"]:
            raise ValueError(f"Dataset/order mismatch at row {i}")
        if fingerprint and row.get("fingerprint") != fingerprint:
            raise ValueError(f"Unverifiable or changed provenance at row {i}; do not append")


def parse_chain(raw):
    if re.search(r"<\s*/?think\b|^\s*(?:analysis|reasoning|thinking)\s*:", raw, re.I | re.M):
        raise ValueError("reasoning_violation")
    match = re.fullmatch(r"\s*Gloss:\s*([^\n]+)\nFINAL_TRANSLATION:\s*([^\n]+)\s*", raw)
    if not match or not all(x.strip() for x in match.groups()):
        raise ValueError("invalid_chain_format")
    return match[1].strip(), match[2].strip()


def build_chain_prompt(support, language, source, text="", images=False):
    examples = []
    for e in support:
        block = f"{language} sentence: {e['source']}\n"
        gloss = e.get("gloss", "").strip()
        if gloss:
            block += f"Gloss: {gloss}\n"
        block += f"FINAL_TRANSLATION: {e['reference']}\n\n"
        examples.append(block)
    context = "The attached images are pages from a grammar reference for this language.\n" if images else f"Here is a grammar reference summary for {language}:\n{text}\n"
    user = (f"Here are some examples of {language} sentences and their corresponding English translations:\n"
            "An example without a gloss provides translation only; it does not demonstrate an empty gloss.\n"
            + "".join(examples) + context + f"\n{language} sentence: {source}\n"
            "Use the grammar reference to produce the gloss first and then the English translation.\n"
            "The gloss must give English lexical meanings and grammatical abbreviations for the source morphemes, in source-word order.\n"
            "Preserve identifiable morpheme boundaries. Use ? for an uncertain morpheme rather than inventing a meaning.\n"
            "Do not copy the source sentence as the gloss. Then translate the complete sentence into natural English.\n"
            "Return exactly two lines:\nGloss: <generated gloss>\nFINAL_TRANSLATION: <English translation>\n"
            "Do not add explanations or analysis.")
    return "You are a linguistic translation engine. Produce the requested gloss and translation.", user
