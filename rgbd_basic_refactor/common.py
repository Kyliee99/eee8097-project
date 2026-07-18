# -*- coding: utf-8 -*-
"""
公共模块：
1. 固定随机种子
2. 图像预处理
3. 创建 train / val / test 数据集和 DataLoader
4. 通用分类指标
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import transforms

from rgbd_dataset import RGBDObjectDataset


# 请把这些 py 文件放到 rgbd_dataset.py 所在目录。
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "rgbd-dataset_eval"

IMAGE_SIZE = 32
BATCH_SIZE = 32
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    """固定随机种子，使多次运行尽量可重复。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """自动选择 CUDA 或 CPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """RGB -> 灰度 -> 32×32 -> Tensor -> [-1, 1]。"""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
    ])


def create_dataloaders(
    dataset_root: Path = DATASET_ROOT,
    batch_size: int = BATCH_SIZE,
    image_size: int = IMAGE_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """创建三种算法共用的 train / val / test DataLoader。"""
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"找不到数据集目录：{dataset_root.resolve()}"
        )

    transform = build_transform(image_size)

    train_dataset = RGBDObjectDataset(
        root_dir=str(dataset_root),
        split="train",
        transform=transform,
    )
    val_dataset = RGBDObjectDataset(
        root_dir=str(dataset_root),
        split="val",
        transform=transform,
    )
    test_dataset = RGBDObjectDataset(
        root_dir=str(dataset_root),
        split="test",
        transform=transform,
    )

    class_names = list(train_dataset.class_names)

    if list(val_dataset.class_names) != class_names:
        raise RuntimeError("验证集与训练集的类别顺序不一致。")

    if list(test_dataset.class_names) != class_names:
        raise RuntimeError("测试集与训练集的类别顺序不一致。")

    generator = torch.Generator()
    generator.manual_seed(RANDOM_SEED)

    trainloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    testloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print("Dataset path:", dataset_root.resolve())
    print("Classes:", class_names)
    print("Train samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))

    return trainloader, valloader, testloader, class_names


def build_classification_metrics(
    labels: list[int] | np.ndarray,
    predictions: list[int] | np.ndarray,
    class_names: list[str],
    total_time: float | None = None,
) -> dict[str, Any]:
    """统一计算 Accuracy、Precision、Recall、F1 和分类报告。"""
    labels_array = np.asarray(labels)
    predictions_array = np.asarray(predictions)

    accuracy = accuracy_score(labels_array, predictions_array)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_array,
        predictions_array,
        average="weighted",
        zero_division=0,
    )

    result: dict[str, Any] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "classification_report": classification_report(
            labels_array,
            predictions_array,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(
            labels_array,
            predictions_array,
            labels=np.arange(len(class_names)),
        ).tolist(),
    }

    if total_time is not None:
        result["total_time"] = float(total_time)
        result["time_per_image"] = (
            float(total_time) / len(labels_array)
            if len(labels_array) > 0
            else 0.0
        )

    return result
