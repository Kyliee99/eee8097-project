#!/usr/bin/env python3
"""Evaluate trained checkpoints using one shared test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.evaluation.exporters import acev2_values
from src.evaluation.runner import evaluate_models


PRIMARY_ITEMS = (
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "weighted_f1",
    "acev2_theoretical",
    "acev2_actual",
)


def _primary_value(result: dict, item: str):
    """Read one console item using the same ACEv2 mapping as the workbook."""
    if item == "acev2_theoretical":
        return acev2_values(result)[0]
    if item == "acev2_actual":
        return acev2_values(result)[1]
    return (result.get("metrics") or {}).get(item)


def _print_primary_metrics(payload: dict) -> None:
    """Print the main metrics and ACEv2 costs in a compact comparison table."""
    results = payload.get("results", [])
    if not results:
        return

    model_width = max(12, *(len(str(result.get("model", ""))) for result in results))
    metric_width = max(len("test_item"), *(len(item) for item in PRIMARY_ITEMS))
    header = f"{'test_item':<{metric_width}}" + "".join(
        f"  {str(result.get('model', '')):>{model_width}}" for result in results
    )
    separator = "-" * len(header)

    print("\nMain evaluation metrics")
    print(header)
    print(separator)
    for item in PRIMARY_ITEMS:
        values = []
        for result in results:
            value = _primary_value(result, item)
            displayed = "N/A" if value is None else f"{float(value):.6f}"
            values.append(f"  {displayed:>{model_width}}")
        print(f"{item:<{metric_width}}{''.join(values)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=ROOT / "configs" / "default.yaml", type=Path)
    parser.add_argument("--models", nargs="+", choices=("mlp", "bnn", "bnn_xnor", "tsetlin"))
    parser.add_argument("--regenerate-split", action="store_true")
    arguments = parser.parse_args()
    config, project_root = load_config(arguments.config)
    payload, paths = evaluate_models(config, project_root, arguments.models, arguments.regenerate_split)
    print(f"Evaluated {len(payload['results'])} model(s).")
    _print_primary_metrics(payload)
    for kind, path in paths.items():
        print(f"{kind.upper()}: {path}")


if __name__ == "__main__":
    main()
