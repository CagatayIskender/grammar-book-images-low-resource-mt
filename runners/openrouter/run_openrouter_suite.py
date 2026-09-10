from __future__ import annotations

from pathlib import Path as _LayoutPath
import sys as _layout_sys
_LAYOUT_ROOT = next(p for p in _LayoutPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
for _layout_dir in ("", "runners/baseline", "runners/qwen3", "runners/qwen35", "runners/openrouter", "scripts/generators"):
    _layout_sys.path.insert(0, str(_LAYOUT_ROOT / _layout_dir))


import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_DIR = _LAYOUT_ROOT
RUNNER = PROJECT_DIR / "runners/openrouter" / "run_grammamt_openrouter_context.py"
CONFIG_PATH = PROJECT_DIR / "runners/openrouter" / "experiment_sources.json"
DEFAULT_MODEL = "google/gemini-2.5-flash-lite"
DEFAULT_MODEL_LABEL = "gemini25flashlite"
DEFAULT_REASONING_EFFORT = os.environ.get("OPENROUTER_REASONING_EFFORT", "none")

CONDITIONS = {
    "cheatsheet_txt": ("text", "cheatsheet/compact_cheatsheet.txt"),
    "cheatsheet_jpg": ("image", "cheatsheet"),
    "summary_tables_txt": ("text", "summary_tables/summary_tables.txt"),
    "summary_tables_jpg": ("image", "summary_tables"),
    "summary_text_txt": ("text", "summary_text/summary_model_readable.txt"),
    "pdfpages_impactful_jpg": ("image", "selected_pages"),
}


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    rounded = int(round(seconds))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def print_suite_progress(completed: int, total: int, elapsed: float) -> None:
    width = 30
    fraction = completed / total if total else 1.0
    filled = min(width, int(fraction * width))
    bar = "#" * filled + "-" * (width - filled)
    eta = elapsed / completed * (total - completed) if completed else None
    print(
        f"[SUITE]    [{bar}] {completed}/{total} ({fraction * 100:5.1f}%) "
        f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model_label", default=DEFAULT_MODEL_LABEL)
    parser.add_argument("--test_n", type=int, default=0, help="Override configured test size")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--normal_only", action="store_true")
    parser.add_argument("--omit_temperature", action="store_true")
    parser.add_argument(
        "--reasoning_effort",
        default=DEFAULT_REASONING_EFFORT,
        choices=("none", "low", "medium", "high"),
        help="OpenRouter reasoning effort. Defaults to none to disable thinking.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate_only", action="store_true")
    args = parser.parse_args()

    if not args.model_label.isalnum() or args.model_label.lower() != args.model_label:
        raise ValueError("--model_label must contain only lowercase letters and digits")

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.source not in config:
        raise ValueError(f"Unknown source {args.source!r}; choose from {', '.join(config)}")
    source_config = config[args.source]
    test_n = args.test_n or source_config["test_n"]
    selected_conditions = [name.strip() for name in args.conditions.split(",") if name.strip()]
    unknown = sorted(set(selected_conditions) - set(CONDITIONS))
    if unknown:
        raise ValueError(f"Unknown conditions: {', '.join(unknown)}")

    language = source_config["language"].lower()
    source = "pdf1_brown" if "pdf1" in args.source else "pdf2_rigsby" if "pdf2" in args.source else "grammar"
    output_group = Path("smoke" if args.smoke else "curated_v1") / args.model_label / language / source / "original"
    materials_root = PROJECT_DIR / source_config["materials_root"]
    modes = [False] if args.normal_only else [False, True]
    if args.validate_only:
        for condition in selected_conditions:
            context_kind, relative_path = CONDITIONS[condition]
            context_path = materials_root / relative_path
            if not context_path.exists():
                raise FileNotFoundError(f"Missing context material: {context_path}")
            if context_kind == "image":
                image_count = sum(
                    path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                    for path in context_path.iterdir()
                )
                if not image_count:
                    raise FileNotFoundError(f"No context images found: {context_path}")
                print(f"[OK] {condition}: {image_count} images")
            else:
                if not context_path.read_text(encoding="utf-8", errors="ignore").strip():
                    raise ValueError(f"Empty context text: {context_path}")
                print(f"[OK] {condition}: text")
        experiment_count = len(selected_conditions) * len(modes)
        print(
            f"[OK] {args.source}: {experiment_count} experiments, "
            f"{experiment_count * test_n} API requests"
        )
        return

    total_experiments = len(selected_conditions) * len(modes)
    completed_experiments = 0
    suite_started = time.time()
    print_suite_progress(0, total_experiments, 0.0)
    for condition in selected_conditions:
        context_kind, relative_path = CONDITIONS[condition]
        context_path = materials_root / relative_path
        if not context_path.exists():
            raise FileNotFoundError(f"Missing context material: {context_path}")
        for model_gloss in modes:
            mode_label = "modelgloss" if model_gloss else ""
            mode_segment = f"{mode_label}_" if mode_label else ""
            stem = f"{args.model_label}_{args.source}_curated_v1_{mode_segment}{condition}"
            result_path = PROJECT_DIR / "results" / output_group / f"results_{stem}.jsonl"
            metric_path = PROJECT_DIR / "metrics" / output_group / f"metrics_{stem}.json"
            command = [
                sys.executable,
                str(RUNNER),
                "--language",
                source_config["language"],
                "--model",
                args.model,
                "--model_label",
                args.model_label,
                "--support_n",
                "21",
                "--test_n",
                str(test_n),
                "--out_jsonl",
                str(result_path),
                "--out_metrics",
                str(metric_path),
                "--reasoning_effort",
                args.reasoning_effort,
            ]
            if context_kind == "text":
                command.extend(["--grammar_text_file", str(context_path)])
            else:
                command.extend(["--grammar_image_dir", str(context_path)])
            if model_gloss:
                command.append("--model_gloss")
            if args.omit_temperature:
                command.append("--omit_temperature")

            display_mode = mode_label or "without ModelGloss"
            print(f"[INFO] Running {args.source}: {display_mode} {condition}", flush=True)
            subprocess.run(command, cwd=PROJECT_DIR, check=True, env=os.environ.copy())
            completed_experiments += 1
            print_suite_progress(
                completed_experiments,
                total_experiments,
                time.time() - suite_started,
            )


if __name__ == "__main__":
    main()
