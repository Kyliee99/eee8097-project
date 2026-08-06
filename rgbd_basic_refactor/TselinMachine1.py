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


BOOLEAN_THRESHOLDS = (-0.5, 0.0, 0.5)

NUM_CLAUSES = 200
T_THRESHOLD = 150
S_SPECIFICITY = 3.0
PATCH_DIM = (6, 6)
MAX_INCLUDED_LITERALS = 24
WEIGHTED_CLAUSES = True

EPOCHS = 25
PATIENCE = 5
PLATFORM = "CPU"

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "tsetlin_output"
MODEL_PATH = OUTPUT_DIR / "best_tsetlin_rgbd.pkl"
METRICS_PATH = OUTPUT_DIR / "training_metrics.json"


def tensor_to_boolean(images: torch.Tensor) -> np.ndarray:
    """[B,1,32,32] 浮点 Tensor -> [B,32,32,3] Boolean uint32。"""
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError(
            f"输入应为 [B,1,H,W]，当前为 {tuple(images.shape)}"
        )

    boolean_planes = [
        (images > threshold).to(torch.uint32)
        for threshold in BOOLEAN_THRESHOLDS
    ]

    boolean_tensor = torch.cat(boolean_planes, dim=1)
    boolean_tensor = boolean_tensor.permute(0, 2, 3, 1).contiguous()

    return boolean_tensor.cpu().numpy()


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

    trainloader, valloader, testloader, class_names = create_dataloaders()

    X_train, y_train = dataloader_to_numpy(trainloader, "training set")
    X_val, y_val = dataloader_to_numpy(valloader, "validation set")
    X_test, y_test = dataloader_to_numpy(testloader, "test set")

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