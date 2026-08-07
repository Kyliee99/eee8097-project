"""Unambiguous classification metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support


def classification_metrics(labels, predictions, class_names: list[str]) -> dict[str, Any]:
    truth, predicted = np.asarray(labels), np.asarray(predictions)
    label_ids = np.arange(len(class_names))
    macro = precision_recall_fscore_support(truth, predicted, labels=label_ids, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(truth, predicted, labels=label_ids, average="weighted", zero_division=0)
    return {
        "sample_count": int(truth.size),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "classification_report": classification_report(
            truth, predicted, labels=label_ids, target_names=class_names, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=label_ids).tolist(),
    }

