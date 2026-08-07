#!/usr/bin/env python3
"""Evaluate trained checkpoints using one shared test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.evaluation.runner import evaluate_models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "configs" / "default.yaml", type=Path)
    parser.add_argument("--models", nargs="+", choices=("mlp", "bnn", "bnn_xnor", "tsetlin"))
    parser.add_argument("--regenerate-split", action="store_true")
    arguments = parser.parse_args()
    config, project_root = load_config(arguments.config)
    payload, paths = evaluate_models(config, project_root, arguments.models, arguments.regenerate_split)
    print(f"Evaluated {len(payload['results'])} model(s).")
    for kind, path in paths.items():
        print(f"{kind.upper()}: {path}")


if __name__ == "__main__":
    main()

