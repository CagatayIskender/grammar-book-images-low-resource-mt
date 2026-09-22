"""Gold-support correction; preserve the previous generation/prompt policy."""
import importlib.util
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("matched_v1_protocol", ROOT / "runners/matched/protocol.py")
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
digest = previous.digest
write_rows = previous.write_rows
REVISION = "matched_gold_v2"


def validate_support(cfg):
    support = cfg["support"]
    if len(support) != 21 or any(not isinstance(e.get("gloss"), str) or not e["gloss"].strip() for e in support):
        raise ValueError("Exactly 21 nonempty GOLD support glosses are required")
    record = json.loads((ROOT / cfg["gold_support_manifest"]).read_text())
    if support != record["support"] or digest(support) != cfg["gold_support_sha256"]:
        raise ValueError("Support differs from the independently aligned gold training source")
    if record["language"] != cfg["language"]:
        raise ValueError("Gold support language mismatch")
    if {e['source'] for e in support} & {e['source'] for e in cfg['test']}:
        raise ValueError("Support source overlaps the evaluation cohort; gold test leakage rejected")


def verify_inputs(cfg):
    previous.verify_inputs(cfg)
    if cfg.get("family") != REVISION:
        raise ValueError("Wrong experiment family")
    for key in ("results", "metrics"):
        if not (ROOT / cfg[key]).resolve().is_relative_to(ROOT / key / REVISION):
            raise ValueError("Historical output overwrite rejected")
    validate_support(cfg)


def prompt(cfg, source):
    validate_support(cfg)
    system, user = previous.prompt(cfg, source)
    required = Counter(f"Gloss: {e['gloss']}\n" for e in cfg["support"])
    if any(user.count(block) < count for block, count in required.items()):
        raise ValueError("Rendered prompt omits a gold support gloss")
    return system, user


def result_row(cfg, index, attempt, origin):
    row = previous.result_row(cfg, index, attempt, origin)
    row["gold_support_sha256"] = cfg["gold_support_sha256"]
    return row


def validate_rows(rows, cfg, complete=False):
    previous.validate_rows(rows, cfg, complete)
    for row in rows:
        if row.get("gold_support_sha256") != cfg["gold_support_sha256"]:
            raise ValueError("Result has no verified gold-support identity")
        index = row["idx"]
        active = dict(cfg, current_gloss=cfg.get("predicted_glosses", [""] * len(cfg["test"]))[index])
        system, user = prompt(active, cfg["test"][index]["source"])
        attempt = row["translation"]["attempt"]
        if attempt.get("user_prompt_sha256") != digest([system, user]):
            raise ValueError("Actual generation prompt hash differs from the expected gold prompt")
        if attempt.get("prompt_evidence") != {"system": system, "user": user, "support_count": 21,
                                                "nonempty_gold_glosses": 21, "support_sha256": cfg["gold_support_sha256"]}:
            raise ValueError("Recorded prompt evidence differs from the reconstructed prompt")
