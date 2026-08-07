"""Tsetlin training with the same manifest and preprocessing as other models."""

from __future__ import annotations

import time

import numpy as np

from src.models.tsetlin import save_tsetlin_checkpoint


def train_tsetlin_model(model, train_data, val_data, class_names, checkpoint_path, metadata, epochs, patience):
    x_train, y_train = train_data
    x_val, y_val = val_data
    best_accuracy, stale_epochs, history = -1.0, 0, []
    for epoch in range(1, int(epochs) + 1):
        started = time.perf_counter()
        model.fit(x_train, y_train, shuffle=True)
        predictions = model.predict(x_val)
        accuracy = float(np.mean(predictions == y_val) * 100.0)
        row = {"epoch": epoch, "validation_accuracy": accuracy, "training_seconds": time.perf_counter() - started}
        history.append(row)
        print(f"Epoch {epoch}/{epochs}: val={accuracy:.2f}%")
        if accuracy > best_accuracy:
            best_accuracy, stale_epochs = accuracy, 0
            save_tsetlin_checkpoint(
                checkpoint_path,
                model,
                {**metadata, "class_names": list(class_names), "epoch": epoch, "validation_accuracy": accuracy},
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                break
    return history

