#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_path
from src.data.loaders import create_dataloaders, set_seed
from src.data.preprocessing import tensor_batches_to_boolean
from src.data.split import build_or_load_manifest
from src.models.tsetlin import create_tsetlin_model
from src.training.tsetlin_trainer import train_tsetlin_model


def _convert(loader, thresholds):
    return tensor_batches_to_boolean([(images, labels) for images, labels in loader], thresholds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--regenerate-split", action="store_true")
    args = parser.parse_args()
    config, project_root = load_config(args.config)
    seed = int(config["project"]["seed"])
    set_seed(seed)
    data_root = resolve_path(project_root, config["project"]["data_root"])
    manifest = build_or_load_manifest(
        data_root, resolve_path(project_root, config["data"]["manifest_path"]),
        config["data"]["split_ratios"], seed, config["data"]["split_unit"], args.regenerate_split,
    )
    loaders, classes = create_dataloaders(config, data_root, manifest)
    thresholds = config["preprocessing"]["tsetlin_thresholds"]
    tm_config = config["models"]["tsetlin"]
    model = create_tsetlin_model(tm_config, seed)
    training = config["training"]["tsetlin"]
    metadata = {
        "input_mode": config["preprocessing"]["input_mode"],
        "boolean_thresholds": list(thresholds),
        "boolean_channels": len(thresholds),
        "patch_dim": list(tm_config["patch_dim"]),
        "number_of_clauses": int(tm_config["clauses"]),
        "max_included_literals": int(tm_config["max_included_literals"]),
        "weighted_clauses": bool(tm_config.get("weighted_clauses", True)),
        "split_manifest_sha256": manifest["sha256"],
    }
    train_tsetlin_model(
        model, _convert(loaders["train"], thresholds), _convert(loaders["val"], thresholds), classes,
        resolve_path(project_root, config["models"]["tsetlin_checkpoint"]), metadata,
        training["epochs"], training["patience"],
    )


if __name__ == "__main__":
    main()
