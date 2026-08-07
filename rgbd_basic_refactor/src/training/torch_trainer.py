"""Shared PyTorch training, early stopping and checkpointing."""

from __future__ import annotations

import csv
from pathlib import Path


def predict(model, loader, device, criterion=None):
    import torch

    model.eval()
    labels_all, predictions_all, loss_total = [], [], 0.0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            if criterion is not None:
                loss_total += float(criterion(logits, labels).item())
            predictions_all.extend(logits.argmax(dim=1).cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
    mean_loss = loss_total / len(loader) if criterion is not None else None
    return labels_all, predictions_all, mean_loss


def train_torch_model(
    model,
    loaders: dict,
    device,
    criterion,
    optimizer,
    checkpoint_path: str | Path,
    history_path: str | Path,
    epochs: int,
    patience: int,
    scheduler=None,
    after_step=None,
) -> list[dict]:
    import torch

    checkpoint, history_file = Path(checkpoint_path), Path(history_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy, stale_epochs, history = -1.0, 0, []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        loss_sum, correct, sample_count = 0.0, 0, 0
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if after_step is not None:
                after_step(model)
            loss_sum += float(loss.item())
            correct += int((logits.argmax(1) == labels).sum().item())
            sample_count += int(labels.numel())
        val_labels, val_predictions, val_loss = predict(model, loaders["val"], device, criterion)
        val_accuracy = 100.0 * sum(a == b for a, b in zip(val_labels, val_predictions)) / len(val_labels)
        if scheduler is not None:
            scheduler.step(val_accuracy)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / len(loaders["train"]),
            "train_accuracy": 100.0 * correct / sample_count,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print("Epoch {epoch}/{epochs}: train={train_accuracy:.2f}% val={val_accuracy:.2f}%".format(epochs=epochs, **row))
        if val_accuracy > best_accuracy:
            best_accuracy, stale_epochs = val_accuracy, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                break
    with history_file.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    return history

