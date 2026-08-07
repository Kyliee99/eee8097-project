"""Construction of one shared set of loaders for every evaluated model."""

from __future__ import annotations

import random

import numpy as np

from .dataset import RGBDObjectDataset
from .preprocessing import build_transform


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloaders(config: dict, dataset_root, manifest):
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as error:
        raise RuntimeError("PyTorch is required to create data loaders") from error
    data_cfg, prep_cfg = config["data"], config["preprocessing"]
    transform = build_transform(
        int(data_cfg["image_size"]),
        str(prep_cfg.get("input_mode", "grayscale")),
        prep_cfg.get("normalize_mean", [0.5]),
        prep_cfg.get("normalize_std", [0.5]),
    )
    datasets = {
        split: RGBDObjectDataset(dataset_root, manifest, split, transform=transform)
        for split in ("train", "val", "test")
    }
    sizes = {split: len(dataset) for split, dataset in datasets.items()}
    print(
        "Dataset split "
        f"(unit={manifest['split_unit']}, seed={manifest['seed']}): "
        f"train={sizes['train']}, val={sizes['val']}, test={sizes['test']}"
    )
    generator = torch.Generator().manual_seed(int(config["project"]["seed"]))
    common = {
        "batch_size": int(data_cfg["batch_size"]),
        "num_workers": int(data_cfg.get("num_workers", 0)),
        "pin_memory": torch.cuda.is_available(),
    }
    loaders = {
        "train": DataLoader(datasets["train"], shuffle=True, generator=generator, **common),
        "val": DataLoader(datasets["val"], shuffle=False, **common),
        "test": DataLoader(datasets["test"], shuffle=False, **common),
    }
    return loaders, list(manifest["class_names"])
