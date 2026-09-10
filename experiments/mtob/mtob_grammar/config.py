from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sources() -> dict[str, Any]:
    return load_json(ROOT / "configs" / "sources.json")


def models() -> dict[str, Any]:
    return load_json(ROOT / "configs" / "models.json")


def input_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def output_path(*parts: str) -> Path:
    path = ROOT.joinpath(*parts).resolve()
    if path != ROOT and ROOT not in path.parents:
        raise ValueError(f"Refusing to write outside isolated suite: {path}")
    return path


def ensure_output_parent(path: Path) -> None:
    resolved = path.resolve()
    if ROOT not in resolved.parents:
        raise ValueError(f"Refusing to write outside isolated suite: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)


def source_config(source_id: str) -> dict[str, Any]:
    config = sources()
    if source_id not in config:
        raise ValueError(f"Unknown source {source_id!r}; choose from {', '.join(config)}")
    return config[source_id]


def model_config(model_id: str) -> dict[str, Any]:
    config = models()
    if model_id not in config:
        raise ValueError(f"Unknown model {model_id!r}; choose from {', '.join(config)}")
    return config[model_id]

