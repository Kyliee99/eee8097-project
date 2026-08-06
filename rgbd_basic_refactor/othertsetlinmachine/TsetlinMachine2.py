# -*- coding: utf-8 -*-
"""
Tsetlin Machine 算法文件。
公共的数据集和灰度预处理来自 common.py；本文件只保留 Tsetlin 特有部分。
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tmu.models.classification.vanilla_classifier import TMClassifier

from common import (
    RANDOM_SEED,
    build_classification_metrics,
    create_dataloaders,
    set_seed,
)


BOOLEAN_THRESHOLDS = (0,5,)

NUM_CLAUSES = 300
T_THRESHOLD = 150
S_SPECIFICITY = 3.0
PATCH_DIM = (4, 4)
MAX_INCLUDED_LITERALS = 32
WEIGHTED_CLAUSES = True

EPOCHS = 25
PATIENCE = 6
PLATFORM = "CPU"

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "tsetlin_output"
MODEL_PATH = OUTPUT_DIR / "best_tsetlin_rgbd_rgb2.pkl"
METRICS_PATH = OUTPUT_DIR / "training_metrics_rgb2.json"


def tensor_to_boolean(images: torch.Tensor) -> np.ndarray:
    """
    RGB Tensor转换为Tsetlin Boolean输入。

    输入：
        [B, 3, H, W]
        像素范围：[0, 1]

    输出：
        [B, H, W, 6]

    六个Boolean通道分别为：
        R > 0.33
        R > 0.66
        G > 0.33
        G > 0.66
        B > 0.33
        B > 0.66
    """

    if images.ndim != 4:
        raise ValueError(
            f"输入应为四维[B,C,H,W]，当前为{tuple(images.shape)}"
        )

    if images.shape[1] != 3:
        raise ValueError(
            f"输入应为RGB三通道，当前通道数为{images.shape[1]}"
        )

    boolean_planes = []

    # 分别处理R、G、B三个通道
    for channel_index in range(3):

        channel = images[
            :,
            channel_index:channel_index + 1,
            :,
            :,
        ]

        for threshold in BOOLEAN_THRESHOLDS:
            boolean_planes.append(
                (channel > threshold).to(torch.uint32)
            )

    # [B,6,H,W]
    boolean_tensor = torch.cat(
        boolean_planes,
        dim=1,
    )

    # TMU需要[B,H,W,C]
    boolean_tensor = boolean_tensor.permute(
        0,
        2,
        3,
        1,
    ).contiguous()

    return boolean_tensor.cpu().numpy().astype(
        np.uint32,
        copy=False,
    )


def dataloader_to_numpy(
    dataloader: DataLoader,
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """将 DataLoader 的所有 batch 合并为 TMU 需要的 NumPy 数组。"""
    feature_batches = []
    label_batches = []

    print(f"\nLoading {dataset_name}...")

    with torch.no_grad():
        for images, labels in dataloader:
            feature_batches.append(tensor_to_boolean(images))
            label_batches.append(
                labels.cpu().numpy().astype(np.uint32, copy=False)
            )

    if not feature_batches:
        raise RuntimeError(f"{dataset_name} 没有读取到任何数据。")

    X = np.concatenate(feature_batches, axis=0)
    y = np.concatenate(label_batches, axis=0)

    return (
        np.ascontiguousarray(X, dtype=np.uint32),
        np.ascontiguousarray(y, dtype=np.uint32),
    )
def print_class_counts(
    labels: np.ndarray,
    class_names: list[str],
    split_name: str,
) -> None:
    counts = np.bincount(
        labels,
        minlength=len(class_names),
    )

    print(f"\n{split_name} class distribution")
    print("-" * 40)

    for class_name, count in zip(
        class_names,
        counts,
    ):
        print(f"{class_name:<15}: {count}")

def save_checkpoint(
    model,
    class_names: list[str],
    epoch: int,
    val_accuracy: float,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model,
        "class_names": class_names,
        "boolean_thresholds": BOOLEAN_THRESHOLDS,
        "patch_dim": PATCH_DIM,
        "epoch": epoch,
        "validation_accuracy": val_accuracy,
    }

    with MODEL_PATH.open("wb") as file:
        pickle.dump(
            checkpoint,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trainloader, valloader, testloader, class_names = (
        create_dataloaders(
            input_mode="rgb"
        )
    )

    X_train, y_train = dataloader_to_numpy(trainloader, "training set")
    X_val, y_val = dataloader_to_numpy(valloader, "validation set")
    X_test, y_test = dataloader_to_numpy(testloader, "test set")

    print_class_counts(
        y_train,
        class_names,
        "Training set",
    )

    print_class_counts(
        y_val,
        class_names,
        "Validation set",
    )

    print_class_counts(
        y_test,
        class_names,
        "Test set",
    )

    print("\nTsetlin input shapes")
    print("-" * 40)
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    print(
        "Unique Boolean values:",
        np.unique(X_train),
    )

    model = TMClassifier(
        number_of_clauses=NUM_CLAUSES,
        T=T_THRESHOLD,
        s=S_SPECIFICITY,
        patch_dim=PATCH_DIM,
        max_included_literals=MAX_INCLUDED_LITERALS,
        platform=PLATFORM,
        weighted_clauses=WEIGHTED_CLAUSES,
        incremental=True,
        seed=RANDOM_SEED,
    )

    best_val_accuracy = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_start = time.perf_counter()
        model.fit(X_train, y_train, shuffle=True)
        train_time = time.perf_counter() - train_start

        val_predictions = model.predict(X_val)
        val_accuracy = float(np.mean(val_predictions == y_val) * 100.0)

        history.append({
            "epoch": epoch,
            "validation_accuracy": val_accuracy,
            "training_seconds": train_time,
        })

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Val Accuracy: {val_accuracy:.2f}% "
            f"Train Time: {train_time:.2f}s"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            save_checkpoint(model, class_names, epoch, val_accuracy)
            print("Best Tsetlin model saved.")
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                print("Early stopping triggered.")
                break

    with MODEL_PATH.open("rb") as file:
        checkpoint = pickle.load(file)

    best_model = checkpoint["model"]

    print("\nEvaluating training set...")

    train_predictions = best_model.predict(X_train)

    train_accuracy = float(
        np.mean(train_predictions == y_train) * 100.0
    )

    final_val_predictions = best_model.predict(X_val)

    final_val_accuracy = float(
        np.mean(final_val_predictions == y_val) * 100.0
    )

    print("\nFinal split accuracy")
    print("-" * 40)
    print(f"Train accuracy: {train_accuracy:.2f}%")
    print(f"Validation accuracy: {final_val_accuracy:.2f}%")

    test_start = time.perf_counter()
    test_predictions = best_model.predict(X_test)
    test_time = time.perf_counter() - test_start

    metrics = build_classification_metrics(
        y_test,
        test_predictions,
        class_names,
        total_time=test_time,
    )
    metrics.update({
        "best_epoch": checkpoint["epoch"],
        "best_validation_accuracy": checkpoint["validation_accuracy"],
        "history": history,
        "boolean_thresholds": list(BOOLEAN_THRESHOLDS),
        "clauses": NUM_CLAUSES,
        "T": T_THRESHOLD,
        "s": S_SPECIFICITY,
        "patch_dim": list(PATCH_DIM),
    })

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print("\nFinal Tsetlin result")
    print("-" * 40)
    print("Best epoch:", checkpoint["epoch"])
    print(
        "Best validation accuracy: "
        f"{checkpoint['validation_accuracy']:.2f}%"
    )
    print(f"Test accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(
        classification_report(
            y_test,
            test_predictions,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0,
        )
    )
    print("Model saved to:", MODEL_PATH.resolve())
    print("Metrics saved to:", METRICS_PATH.resolve())


if __name__ == "__main__":
    main()
