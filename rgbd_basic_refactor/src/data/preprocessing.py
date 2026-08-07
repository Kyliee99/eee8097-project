"""Shared image and Boolean-feature preprocessing."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def build_transform(image_size: int, input_mode: str, mean: Sequence[float], std: Sequence[float]):
    try:
        from torchvision import transforms
    except ImportError as error:
        raise RuntimeError("torchvision is required for image preprocessing") from error
    steps = []
    if input_mode == "grayscale":
        steps.append(transforms.Grayscale(num_output_channels=1))
    elif input_mode != "rgb":
        raise ValueError("input_mode must be 'grayscale' or 'rgb'")
    steps.extend([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    if mean and std:
        steps.append(transforms.Normalize(mean=tuple(mean), std=tuple(std)))
    return transforms.Compose(steps)


def boolean_planes_numpy(images: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """Convert [N,C,H,W] arrays to contiguous [N,H,W,C*thresholds] uint32 planes."""
    values = np.asarray(images)
    if values.ndim != 4:
        raise ValueError(f"Expected [N,C,H,W], received {values.shape}")
    planes = [(values[:, channel] > threshold) for channel in range(values.shape[1]) for threshold in thresholds]
    return np.ascontiguousarray(np.stack(planes, axis=-1), dtype=np.uint32)


def tensor_batches_to_boolean(cached_batches, thresholds: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    features, labels = [], []
    for images, batch_labels in cached_batches:
        features.append(boolean_planes_numpy(images.detach().cpu().numpy(), thresholds))
        labels.append(batch_labels.detach().cpu().numpy().astype(np.uint32, copy=False))
    if not features:
        raise RuntimeError("No cached batches were supplied")
    return np.concatenate(features), np.concatenate(labels)

