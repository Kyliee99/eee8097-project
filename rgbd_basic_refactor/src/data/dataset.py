"""PyTorch dataset backed by an explicit, reusable split manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PIL import Image

from .split import paths_for_split

try:
    from torch.utils.data import Dataset
except ImportError:  # Allows split/metric tests to run without PyTorch installed.
    class Dataset:  # type: ignore[no-redef]
        pass


class RGBDObjectDataset(Dataset):
    def __init__(self, dataset_root: str | Path, manifest: dict, split: str, transform: Callable | None = None):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.class_names = list(manifest["class_names"])
        self.transform = transform
        self.samples = list(paths_for_split(manifest, self.dataset_root, split))
        if not self.samples:
            raise RuntimeError(f"No samples assigned to split: {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
        except Exception as error:
            raise RuntimeError(f"Unable to read image: {path}") from error
        if self.transform is not None:
            image = self.transform(image)
        return image, label

