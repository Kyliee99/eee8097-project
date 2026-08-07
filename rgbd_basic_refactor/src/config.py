"""Configuration loading and repository-relative path handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    required = {"project", "data", "preprocessing", "models", "evaluation"}
    missing = sorted(required.difference(config))
    if missing:
        raise ValueError(f"Missing configuration sections: {', '.join(missing)}")
    project_root = config_path.parent.parent
    return config, project_root


def resolve_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def nested(config: dict[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        value = value[key]
    return value

