from __future__ import annotations

import json
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from rgbd_dataset import RGBDObjectDataset
except ImportError as error:
    raise ImportError(
        "Cannot import RGBDObjectDataset. Put this file in the same project "
        "environment as rgbd_dataset.py."
    )

# ============================================================
# 1. Fixed random seed
# ============================================================

RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()


# ============================================================
# 2. File paths
# ============================================================

# Current Python file directory:
# .../BNN_Project/RGB-D Object Dataset
SCRIPT_DIR = Path(__file__).resolve().parent

# Project root:
# .../BNN_Project
PROJECT_ROOT = SCRIPT_DIR.parent

# Same dataset path as the BNN program:
# .../BNN_Project/data/rgbd-dataset_eval
DATASET_ROOT = PROJECT_ROOT / "data" / "rgbd-dataset_eval"

OUTPUT_DIR = SCRIPT_DIR / "tsetlin_rgbd_output_dataloader"
MODEL_PATH = OUTPUT_DIR / "best_tsetlin_rgbd.pkl"
METRICS_PATH = OUTPUT_DIR / "training_metrics.json"

if not DATASET_ROOT.exists():
    raise FileNotFoundError(
        "Dataset directory does not exist: {DATASET_ROOT.resolve()}"
    )

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. Image preprocessing -- same as BNN
# ============================================================

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

# The BNN transform produces values in [-1, 1].
# These three thresholds correspond to approximately 25%, 50%, and 75%
# brightness in the original [0, 1] range.
BOOLEAN_THRESHOLDS = (-0.5, 0.0, 0.5)


# ============================================================
# 4. Create train / validation / test datasets
# ============================================================

train_dataset = RGBDObjectDataset(
    root_dir=str(DATASET_ROOT),
    split="train",
    transform=transform,
)

val_dataset = RGBDObjectDataset(
    root_dir=str(DATASET_ROOT),
    split="val",
    transform=transform,
)

test_dataset = RGBDObjectDataset(
    root_dir=str(DATASET_ROOT),
    split="test",
    transform=transform,
)

CLASS_NAMES = list(train_dataset.class_names)
NUMBER_OF_CLASSES = len(CLASS_NAMES)

if list(val_dataset.class_names) != CLASS_NAMES:
    raise RuntimeError("Validation-set class order differs from training set.")

if list(test_dataset.class_names) != CLASS_NAMES:
    raise RuntimeError("Test-set class order differs from training set.")

print("Dataset path:", DATASET_ROOT.resolve())
print("Number of classes:", NUMBER_OF_CLASSES)
print("Class names:", CLASS_NAMES)
print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))
print("Test samples:", len(test_dataset))


# ============================================================
# 5. Create DataLoaders -- same method as BNN
# ============================================================

BATCH_SIZE = 32

# A seeded generator makes the training-loader order reproducible.
loader_generator = torch.Generator()
loader_generator.manual_seed(RANDOM_SEED)

trainloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    generator=loader_generator,
)

valloader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

testloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)


# ============================================================
# 6. DataLoader batches -> Boolean NumPy arrays
# ============================================================

def tensor_batch_to_boolean_planes(images: torch.Tensor) -> np.ndarray:
    """
    Convert a batch from [B, 1, 32, 32] floating-point Tensor to
    [B, 32, 32, number_of_thresholds] uint32 Boolean data.

    For each grayscale pixel x, the features are:
        x > -0.5
        x >  0.0
        x >  0.5
    """
    if images.ndim != 4:
        raise ValueError(
            "Expected images with 4 dimensions [B,C,H,W], got {images.shape}."
        )

    if images.shape[1] != 1:
        raise ValueError(
            "Expected one grayscale channel, got {images.shape[1]} channels."
        )

    boolean_planes = [
        (images > threshold).to(torch.uint32)
        for threshold in BOOLEAN_THRESHOLDS
    ]

    # [B, thresholds, H, W]
    boolean_tensor = torch.cat(boolean_planes, dim=1)

    # TMU convolutional input format: [B, H, W, channels]
    boolean_tensor = boolean_tensor.permute(0, 2, 3, 1).contiguous()

    return boolean_tensor.cpu().numpy().astype(np.uint32, copy=False)


def dataloader_to_numpy(
    dataloader: DataLoader,
    dataset_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read images through RGBDObjectDataset and DataLoader, then combine all
    batches into the NumPy arrays required by TMU.
    """
    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []

    print(f"\nLoading {dataset_name} through DataLoader...")

    loaded_samples = 0
    with torch.no_grad():
        for images, labels in dataloader:
            boolean_images = tensor_batch_to_boolean_planes(images)

            feature_batches.append(boolean_images)
            label_batches.append(
                labels.detach().cpu().numpy().astype(np.uint32, copy=False)
            )

            loaded_samples += labels.size(0)
            print(
                f"\r  Loaded {loaded_samples}/{len(dataloader.dataset)}",
                end="",
                flush=True,
            )

    print()

    if not feature_batches:
        raise RuntimeError(f"No samples were loaded from {dataset_name}.")

    X = np.concatenate(feature_batches, axis=0)
    y = np.concatenate(label_batches, axis=0)

    return (
        np.ascontiguousarray(X, dtype=np.uint32),
        np.ascontiguousarray(y, dtype=np.uint32),
    )


# ============================================================
# 7. Tsetlin Machine settings
# ============================================================

NUM_CLAUSES = 500
T_THRESHOLD = 250
S_SPECIFICITY = 5.0
PATCH_DIM = (4, 4)
MAX_INCLUDED_LITERALS = 48
WEIGHTED_CLAUSES = True
EPOCHS = 30
PATIENCE = 7
PLATFORM = "CPU"


# ============================================================
# 8. Save the best model
# ============================================================

def save_checkpoint(
    model: object,
    epoch: int,
    val_accuracy: float,
) -> None:
    checkpoint = {
        "model": model,
        "class_names": CLASS_NAMES,
        "image_height": IMAGE_HEIGHT,
        "image_width": IMAGE_WIDTH,
        "boolean_thresholds": tuple(BOOLEAN_THRESHOLDS),
        "patch_dim": tuple(PATCH_DIM),
        "epoch": epoch,
        "validation_accuracy": val_accuracy,
        "data_reader": "RGBDObjectDataset + DataLoader",
    }

    with MODEL_PATH.open("wb") as file:
        pickle.dump(checkpoint, file, protocol=pickle.HIGHEST_PROTOCOL)


# ============================================================
# 9. Train using validation set; evaluate once on test set
# ============================================================

def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, object]:
    try:
        from sklearn.metrics import classification_report
        from tmu.models.classification.vanilla_classifier import TMClassifier
    except ImportError as error:
        raise ImportError(
            "TMU or scikit-learn is missing. Install them before running."
        )

    model = TMClassifier(
        number_of_clauses=NUM_CLAUSES,
        T=T_THRESHOLD,
        s=S_SPECIFICITY,
        patch_dim=tuple(PATCH_DIM),
        max_included_literals=MAX_INCLUDED_LITERALS,
        platform=PLATFORM,
        weighted_clauses=WEIGHTED_CLAUSES,
        incremental=True,
        seed=RANDOM_SEED,
    )

    history: list[dict[str, float | int]] = []
    best_val_accuracy = -1.0
    epochs_without_improvement = 0

    print("\nStart Tsetlin Machine training")
    print("=" * 40)
    print("Training input shape:", X_train.shape)
    print("Validation input shape:", X_val.shape)
    print("Test input shape:", X_test.shape)
    print("Clauses:", NUM_CLAUSES)
    print("T:", T_THRESHOLD)
    print("s:", S_SPECIFICITY)
    print("Patch:", PATCH_DIM)
    print("Platform:", PLATFORM)

    for epoch in range(1, EPOCHS + 1):
        train_start = time.perf_counter()
        model.fit(X_train, y_train, shuffle=True)
        train_seconds = time.perf_counter() - train_start

        val_start = time.perf_counter()
        val_predictions = model.predict(X_val)
        val_seconds = time.perf_counter() - val_start

        val_accuracy = float(
            np.mean(val_predictions == y_val) * 100.0
        )

        history.append({
            "epoch": epoch,
            "validation_accuracy": val_accuracy,
            "training_seconds": train_seconds,
            "validation_seconds": val_seconds,
        })

        print(
            "Epoch [{epoch}/{EPOCHS}] "
            "Val Accuracy: {val_accuracy:.2f}% "
            "Train Time: {train_seconds:.2f}s "
            "Val Time: {val_seconds:.2f}s"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            save_checkpoint(model, epoch, val_accuracy)
            print("Best Tsetlin model saved.")
        else:
            epochs_without_improvement += 1
            print(
                "No validation improvement for "
                f"{epochs_without_improvement} epoch(s)."
            )

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    with MODEL_PATH.open("rb") as file:
        best_checkpoint = pickle.load(file)

    best_model = best_checkpoint["model"]
    test_start = time.perf_counter()
    test_predictions = best_model.predict(X_test)
    test_seconds = time.perf_counter() - test_start

    test_accuracy = float(
        np.mean(test_predictions == y_test) * 100.0
    )

    report_text = classification_report(
        y_test,
        test_predictions,
        labels=np.arange(len(class_names)),
        target_names=list(class_names),
        zero_division=0,
        digits=4,
    )

    report_dict = classification_report(
        y_test,
        test_predictions,
        labels=np.arange(len(class_names)),
        target_names=list(class_names),
        zero_division=0,
        output_dict=True,
    )

    print("\nFinal Tsetlin Machine result")
    print("-" * 40)
    print("Best epoch:", best_checkpoint["epoch"])
    print(
        "Best validation accuracy: "
        f"{best_checkpoint['validation_accuracy']:.2f}%"
    )
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print(f"Test time: {test_seconds:.2f}s")
    print("\nClassification report:")
    print(report_text)

    return {
        "history": history,
        "best_epoch": int(best_checkpoint["epoch"]),
        "best_validation_accuracy": float(
            best_checkpoint["validation_accuracy"]
        ),
        "test_accuracy": test_accuracy,
        "test_seconds": test_seconds,
        "classification_report": report_dict,
    }


# ============================================================
# 10. Main program
# ============================================================

def main() -> None:
    X_train, y_train = dataloader_to_numpy(
        trainloader,
        "training set",
    )

    X_val, y_val = dataloader_to_numpy(
        valloader,
        "validation set",
    )

    X_test, y_test = dataloader_to_numpy(
        testloader,
        "test set",
    )

    results = train_and_evaluate(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        CLASS_NAMES,
    )

    metrics = {
        "dataset_path": str(DATASET_ROOT.resolve()),
        "data_reader": "RGBDObjectDataset + DataLoader",
        "batch_size": BATCH_SIZE,
        "image_size": [IMAGE_HEIGHT, IMAGE_WIDTH],
        "boolean_thresholds": list(BOOLEAN_THRESHOLDS),
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        "class_names": CLASS_NAMES,
        **results,
    }

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    print("\nModel saved to:", MODEL_PATH.resolve())
    print("Metrics saved to:", METRICS_PATH.resolve())


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nError: {error}", file=sys.stderr)
        raise