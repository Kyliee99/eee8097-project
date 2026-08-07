"""Tsetlin model/checkpoint helpers; TMU is imported only when needed."""

from __future__ import annotations

import pickle
from pathlib import Path


def create_tsetlin_model(config: dict, seed: int):
    try:
        from tmu.models.classification.vanilla_classifier import TMClassifier
    except ImportError as error:
        raise RuntimeError("Install the 'tmu' package to train or evaluate Tsetlin models") from error
    return TMClassifier(
        number_of_clauses=int(config["clauses"]),
        T=int(config["threshold"]),
        s=float(config["specificity"]),
        patch_dim=tuple(config["patch_dim"]),
        max_included_literals=int(config["max_included_literals"]),
        platform=str(config.get("platform", "CPU")),
        weighted_clauses=bool(config.get("weighted_clauses", True)),
        incremental=True,
        seed=int(seed),
    )


def save_tsetlin_checkpoint(path: str | Path, model, metadata: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump({"model": model, **metadata}, stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_tsetlin_checkpoint(path: str | Path) -> tuple[object, dict]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Tsetlin checkpoint not found: {source}")
    # Pickle is unsafe for untrusted files. Only load checkpoints created by this project.
    with source.open("rb") as stream:
        checkpoint = pickle.load(stream)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"], checkpoint
    return checkpoint, {}

