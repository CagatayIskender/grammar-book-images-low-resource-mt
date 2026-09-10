"""OpenRouter XCOMET entry point for the organized output tree."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "runners"))
from score_verified import score_targets
from experiment_io import COUNTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_label", default="gemini25flashlite")
    parser.add_argument("--results_root", default=str(ROOT / "results/curated_v1"))
    parser.add_argument("--metrics_root", default=str(ROOT / "metrics"))
    parser.add_argument("--group", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    targets = []
    for path in Path(args.results_root).rglob(f"results_{args.model_label}_*.jsonl"):
        rel = path.resolve().relative_to(ROOT / "results")
        if rel.parts[0] == "smoke":
            continue
        lang = next(x for x in ("gitksan", "lezgi", "natugu", "tsez") if x in rel.parts)
        source = next((x for x in rel.parts if x.startswith("pdf")), "grammar")
        group = lang + "_" + source if lang == "gitksan" else lang
        if args.group and group not in args.group:
            continue
        metric = Path(args.metrics_root) / rel.parent / path.name.replace("results_", "metrics_", 1).replace(".jsonl", ".json")
        targets.append({"language":lang.title(), "results":str(path.resolve()), "metrics":str(metric.resolve()),
                        "expected_records":99 if lang == "tsez" else COUNTS[lang.title()]})
    if not targets:
        raise FileNotFoundError("No matching complete-experiment candidates")
    score_targets(targets, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
