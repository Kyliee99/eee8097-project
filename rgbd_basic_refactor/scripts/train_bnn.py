#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_path
from src.data.loaders import create_dataloaders, set_seed
from src.data.split import build_or_load_manifest
from src.models.bnn import RGBDBNN, clamp_binary_weights
from src.training.torch_trainer import train_torch_model


def main() -> None:
    import torch
    import torch.nn as nn

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RGBDBNN(config["data"]["image_size"], len(classes), config["models"]["hidden_sizes"], use_xnor=False).to(device)
    training = config["training"]["bnn"]
    optimizer = torch.optim.Adam(model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6)
    artifacts = resolve_path(project_root, config["project"]["artifacts_root"])
    train_torch_model(
        model, loaders, device, nn.CrossEntropyLoss(), optimizer,
        resolve_path(project_root, config["models"]["bnn_checkpoint"]), artifacts / "results" / "bnn_history.csv",
        training["epochs"], training["patience"], scheduler=scheduler, after_step=clamp_binary_weights,
    )


if __name__ == "__main__":
    main()

