# -*- coding: utf-8 -*-
"""MLP 和 BNN 共用的 PyTorch 训练与测试流程。"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from common import build_classification_metrics


def evaluate_torch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int], float]:
    """在验证集或测试集上计算 Loss、Accuracy 和预测结果。"""
    if len(dataloader) == 0:
        raise RuntimeError("DataLoader 中没有数据。")

    model.eval()
    running_loss = 0.0
    labels_all: list[int] = []
    predictions_all: list[int] = []

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)

            running_loss += loss.item()
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(predictions.cpu().tolist())

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = time.perf_counter() - start_time
    average_loss = running_loss / len(dataloader)

    correct = sum(
        int(prediction == label)
        for prediction, label in zip(predictions_all, labels_all)
    )
    accuracy = 100.0 * correct / len(labels_all)

    return average_loss, accuracy, labels_all, predictions_all, total_time


def train_torch_model(
    model: nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    testloader: DataLoader,
    class_names: list[str],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_path: Path,
    history_path: Path,
    device: torch.device,
    epochs: int,
    patience: int,
    scheduler=None,
    after_optimizer_step: Callable[[nn.Module], None] | None = None,
) -> dict:
    """MLP / BNN 共用的训练、验证、Early Stopping 和测试流程。"""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_accuracy = -1.0
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in trainloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if after_optimizer_step is not None:
                after_optimizer_step(model)

            running_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

        train_loss = running_loss / len(trainloader)
        train_accuracy = 100.0 * correct / total

        val_loss, val_accuracy, _, _, _ = evaluate_torch(
            model,
            valloader,
            criterion,
            device,
        )

        if scheduler is not None:
            scheduler.step(val_accuracy)

        learning_rate = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": learning_rate,
        })

        print(
            f"Epoch [{epoch}/{epochs}] "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Accuracy: {train_accuracy:.2f}%, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.2f}%, "
            f"LR: {learning_rate:.6f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), model_path)
            print(f"Best model saved: {best_val_accuracy:.2f}%")
        else:
            epochs_without_improvement += 1
            print(
                "No improvement for "
                f"{epochs_without_improvement} epoch(s)."
            )

            if epochs_without_improvement >= patience:
                print("Early stopping triggered.")
                break

    if history:
        with history_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=list(history[0].keys()),
            )
            writer.writeheader()
            writer.writerows(history)

    state_dict = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(state_dict)

    test_loss, test_accuracy, labels, predictions, test_time = evaluate_torch(
        model,
        testloader,
        criterion,
        device,
    )

    metrics = build_classification_metrics(
        labels,
        predictions,
        class_names,
        total_time=test_time,
    )
    metrics.update({
        "best_validation_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "history": history,
    })

    print("\nFinal result")
    print("-" * 40)
    print(f"Best Validation Accuracy: {best_val_accuracy:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(
        classification_report(
            labels,
            predictions,
            labels=range(len(class_names)),
            target_names=class_names,
            zero_division=0,
        )
    )
    print("Model saved to:", model_path.resolve())

    return metrics
