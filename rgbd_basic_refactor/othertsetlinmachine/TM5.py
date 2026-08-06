# -*- coding: utf-8 -*-
"""
Tsetlin Machine 算法文件。
公共的数据集和灰度预处理来自 common.py；本文件只保留 Tsetlin 特有部分。
"""

from __future__ import annotations

import json
import pickle
import time
import logging
from pathlib import Path

import numpy as np
import torch

# 关闭Pillow的DEBUG日志
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tmu.models.classification.vanilla_classifier import TMClassifier

from common import (
    RANDOM_SEED,
    build_classification_metrics,
    create_dataloaders,
    set_seed,
)


# 是否忽略接近纯黑的背景像素
IGNORE_NEAR_BLACK_BACKGROUND = False

# 每个RGB通道使用三个分位数阈值
RGB_QUANTILES = (
    0.125,
    0.25,
    0.375,
    0.50,
    0.625,
    0.75,
    0.875,
)

NUM_CLAUSES = 250
T_THRESHOLD = 188
S_SPECIFICITY = 2.5
PATCH_DIM = (4, 4)
MAX_INCLUDED_LITERALS = 24
WEIGHTED_CLAUSES = True

CLAUSE_DROP_P = 0.00
LITERAL_DROP_P = 0.00

EPOCHS = 15
PATIENCE = 3
PLATFORM = "CPU"

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "tsetlin5_output"
MODEL_PATH = OUTPUT_DIR / "best_tsetlin_rgbd_rgb5_2.pkl"
METRICS_PATH = OUTPUT_DIR / "training_metrics_rgb5_2.json"


def calculate_rgb_thresholds(
    trainloader: DataLoader,
) -> tuple[tuple[float, ...], ...]:
    """
    只根据训练集，为R、G、B三个通道分别计算
    25%、50%、75%三个分位数阈值。

    返回格式：
        (
            (R25, R50, R75),
            (G25, G50, G75),
            (B25, B50, B75),
        )
    """

    # 三个RGB通道，每个通道统计0～255像素直方图
    histograms = torch.zeros(
        (3, 256),
        dtype=torch.long,
    )

    print(
        "\nCalculating RGB quantile thresholds "
        "from training set..."
    )

    with torch.no_grad():
        for images, _ in trainloader:

            if images.ndim != 4 or images.shape[1] != 3:
                raise ValueError(
                    "自动阈值计算需要RGB输入[B,3,H,W]，"
                    f"当前形状为{tuple(images.shape)}"
                )

            # [0,1]转换为0～255整数
            pixel_values = (
                images
                .clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .to(torch.int64)
            )

            # [B,3,H,W] -> [所有像素,3]
            pixel_values = (
                pixel_values
                .permute(0, 2, 3, 1)
                .reshape(-1, 3)
            )

            # 可选：计算阈值时忽略接近纯黑的背景
            if IGNORE_NEAR_BLACK_BACKGROUND:
                valid_mask = (
                    pixel_values.max(dim=1).values > 5
                )

                pixel_values = pixel_values[
                    valid_mask
                ]

            # 如果当前batch全部是无效背景，则跳过
            if pixel_values.numel() == 0:
                continue

            for channel_index in range(3):
                channel_histogram = torch.bincount(
                    pixel_values[:, channel_index],
                    minlength=256,
                )

                histograms[channel_index] += (
                    channel_histogram
                )

    all_thresholds = []
    channel_names = ("R", "G", "B")

    for channel_index, channel_name in enumerate(
        channel_names
    ):
        histogram = histograms[channel_index]

        total_pixels = int(
            histogram.sum().item()
        )

        if total_pixels == 0:
            raise RuntimeError(
                f"{channel_name}通道没有有效像素。"
            )

        cumulative = histogram.cumsum(dim=0)
        channel_thresholds = []

        for quantile in RGB_QUANTILES:

            target_count = max(
                1,
                int(
                    np.ceil(
                        total_pixels * quantile
                    )
                ),
            )

            candidates = torch.nonzero(
                cumulative >= target_count,
                as_tuple=False,
            )

            pixel_value = int(
                candidates[0].item()
            )

            threshold = (
                pixel_value / 255.0
            )

            channel_thresholds.append(
                float(threshold)
            )

        all_thresholds.append(
            tuple(channel_thresholds)
        )

        threshold_text = ", ".join(
            f"{threshold:.4f}"
            for threshold in channel_thresholds
        )

        print(
            f"{channel_name} thresholds: "
            f"{threshold_text}"
        )

    return tuple(all_thresholds)


def tensor_to_boolean(
    images: torch.Tensor,
    rgb_thresholds: tuple[
        tuple[float, ...],
        ...
    ],
) -> np.ndarray:
    """
    使用每个RGB通道的三个阈值生成Boolean输入。

    输入：
        images：[B,3,H,W]

    阈值：
        R25、R50、R75
        G25、G50、G75
        B25、B50、B75

    输出：
        [B,H,W,9]
    """

    if images.ndim != 4:
        raise ValueError(
            f"输入应为[B,C,H,W]，"
            f"当前为{tuple(images.shape)}"
        )

    if images.shape[1] != 3:
        raise ValueError(
            f"需要RGB三通道，"
            f"当前为{images.shape[1]}通道"
        )

    if len(rgb_thresholds) != 3:
        raise ValueError(
            "rgb_thresholds必须包含R、G、B三个通道。"
        )

    number_of_levels = len(
        rgb_thresholds[0]
    )

    # if number_of_levels != 3:
    #     raise ValueError(
    #         "每个RGB通道必须包含三个阈值。"
    #     )

    if any(
        len(channel_thresholds)
        != number_of_levels
        for channel_thresholds
        in rgb_thresholds
    ):
        raise ValueError(
            "三个RGB通道的阈值数量必须相同。"
        )

    # rgb_thresholds形状：[3,3]
    # 转换后：[1,3,3,1,1]
    threshold_tensor = torch.tensor(
        rgb_thresholds,
        dtype=images.dtype,
        device=images.device,
    ).view(
        1,
        3,
        number_of_levels,
        1,
        1,
    )

    # images.unsqueeze(2)：[B,3,1,H,W]
    # 比较后：[B,3,3,H,W]
    boolean_tensor = (
        images.unsqueeze(2)
        > threshold_tensor
    ).to(torch.uint32)

    batch_size = boolean_tensor.shape[0]
    height = boolean_tensor.shape[3]
    width = boolean_tensor.shape[4]

    # [B,3,3,H,W]
    # 变成[B,H,W,3,3]
    # 再展平为[B,H,W,9]
    boolean_tensor = (
        boolean_tensor
        .permute(0, 3, 4, 1, 2)
        .contiguous()
        .view(
            batch_size,
            height,
            width,
            3 * len(RGB_QUANTILES),
        )
    )

    return (
        boolean_tensor
        .cpu()
        .numpy()
        .astype(
            np.uint32,
            copy=False,
        )
    )

def dataloader_to_numpy(
            dataloader: DataLoader,
            dataset_name: str,
        rgb_thresholds: tuple[
            tuple[float, ...],
            ...
        ],
) -> tuple[np.ndarray, np.ndarray]:
    """将 DataLoader 的所有 batch 合并为 TMU 需要的 NumPy 数组。"""
    feature_batches = []
    label_batches = []

    print(f"\nLoading {dataset_name}...")

    with torch.no_grad():
        for images, labels in dataloader:
            feature_batches.append(
                tensor_to_boolean(
                    images,
                    rgb_thresholds,
                )
            )
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
    val_macro_f1: float,
        rgb_thresholds: tuple[
            tuple[float, ...],
            ...
        ],
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        # 训练好的模型
        "model": model,
        # 类别顺序
        "class_names": list(class_names),
        # 输入预处理信息
        "input_mode": "rgb",
        "rgb_channel_thresholds": tuple(
            tuple(channel_thresholds)
            for channel_thresholds
            in rgb_thresholds
        ),

        "boolean_channels": 9,

        "boolean_channel_order": (
            "R_q25,R_q50,R_q75,"
            "G_q25,G_q50,G_q75,"
            "B_q25,B_q50,B_q75"
        ),
        # Tsetlin参数
        "patch_dim": tuple(PATCH_DIM),
        "number_of_clauses": NUM_CLAUSES,
        "T": T_THRESHOLD,
        "s": S_SPECIFICITY,
        "max_included_literals": (
            MAX_INCLUDED_LITERALS
        ),
        "weighted_clauses": (
            WEIGHTED_CLAUSES
        ),
        # 正则化参数
        "clause_drop_p": CLAUSE_DROP_P,
        "literal_drop_p": LITERAL_DROP_P,
        # 最佳验证结果
        "epoch": epoch,
        "validation_accuracy": val_accuracy,
        "validation_macro_f1": val_macro_f1,
    }

    with MODEL_PATH.open("wb") as file:
        pickle.dump(
            checkpoint,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    print("\nLoaded checkpoint information")
    print("-" * 50)
    print(
        "Clause drop:",
        checkpoint["clause_drop_p"],
    )
    print(
        "Literal drop:",
        checkpoint["literal_drop_p"],
    )
    print(
        "Best epoch:",
        checkpoint["epoch"],
    )
    print(
        "Model path:",
        MODEL_PATH.resolve(),
    )

def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trainloader, valloader, testloader, class_names = (
        create_dataloaders(
            input_mode="rgb"
        )
    )

    # 只根据训练集自动计算RGB阈值
    rgb_thresholds = calculate_rgb_thresholds(
        trainloader
    )

    print("\nFinal RGB thresholds")
    print("-" * 40)

    for channel_name, channel_thresholds in zip(
            ("R", "G", "B"),
            rgb_thresholds,
    ):
        threshold_text = ", ".join(
            f"{threshold:.4f}"
            for threshold in channel_thresholds
        )

        print(
            f"{channel_name}: "
            f"{threshold_text}"
        )

    X_train, y_train = dataloader_to_numpy(
        trainloader,
        "training set",
        rgb_thresholds,
    )

    X_val, y_val = dataloader_to_numpy(
        valloader,
        "validation set",
        rgb_thresholds,
    )

    X_test, y_test = dataloader_to_numpy(
        testloader,
        "test set",
        rgb_thresholds,
    )

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

    print("\nCurrent Tsetlin configuration")
    print("-" * 50)
    print("NUM_CLAUSES:", NUM_CLAUSES)
    print("T_THRESHOLD:", T_THRESHOLD)
    print("S_SPECIFICITY:", S_SPECIFICITY)
    print("PATCH_DIM:", PATCH_DIM)
    print("CLAUSE_DROP_P:", CLAUSE_DROP_P)
    print("LITERAL_DROP_P:", LITERAL_DROP_P)
    print("EPOCHS:", EPOCHS)
    print("PATIENCE:", PATIENCE)
    print("MODEL_PATH:", MODEL_PATH.resolve())

    model = TMClassifier(
        number_of_clauses=NUM_CLAUSES,
        T=T_THRESHOLD,
        s=S_SPECIFICITY,
        patch_dim=PATCH_DIM,
        max_included_literals=MAX_INCLUDED_LITERALS,

        weighted_clauses=WEIGHTED_CLAUSES,

        # 减少模型记住训练集中的局部规则
        clause_drop_p=CLAUSE_DROP_P,
        literal_drop_p=LITERAL_DROP_P,

        platform=PLATFORM,
        incremental=True,
        seed=RANDOM_SEED,
    )

    best_val_accuracy = -1.0
    best_val_macro_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_start = time.perf_counter()
        model.fit(X_train, y_train, shuffle=True)
        train_time = time.perf_counter() - train_start

        val_predictions = model.predict(X_val)
        val_accuracy = float(np.mean(val_predictions == y_val) * 100.0)

        val_macro_f1 = float(
            f1_score(
                y_val,
                val_predictions,
                average="macro",
                zero_division=0,
            )
        )
        val_class_recall = recall_score(
            y_val,
            val_predictions,
            labels=np.arange(len(class_names)),
            average=None,
            zero_division=0,
        )

        predicted_counts = np.bincount(
            val_predictions,
            minlength=len(class_names),
        )

        history.append({
            "epoch": epoch,
            "validation_accuracy": val_accuracy,
            "validation_macro_f1": val_macro_f1,
            "training_seconds": train_time,
        })

        print(
            f"Epoch [{epoch}/{EPOCHS}] "
            f"Val Accuracy: {val_accuracy:.2f}% "
            f"Val Macro F1: {val_macro_f1:.4f} "
            f"Train Time: {train_time:.2f}s"
        )
        print("Validation recall by class")

        for class_name, recall, predicted_count in zip(
                class_names,
                val_class_recall,
                predicted_counts,
        ):
            print(
                f"  {class_name:<15} "
                f"recall={recall:.2f} "
                f"predicted={predicted_count}"
            )


        macro_f1_improved = (
                val_macro_f1 > best_val_macro_f1 + 1e-4
        )

        same_f1_better_accuracy = (
                abs(
                    val_macro_f1
                    - best_val_macro_f1
                ) <= 1e-4
                and val_accuracy > best_val_accuracy
        )

        if macro_f1_improved or same_f1_better_accuracy:

            best_val_accuracy = val_accuracy
            best_val_macro_f1 = val_macro_f1
            epochs_without_improvement = 0

            save_checkpoint(
                model,
                class_names,
                epoch,
                val_accuracy,
                val_macro_f1,
                rgb_thresholds,
            )

            print(
                "Best Tsetlin model saved "
                "according to validation Macro F1."
            )

        else:
                epochs_without_improvement += 1

                print(
                    "No improvement for "
                    f"{epochs_without_improvement} epoch(s)."
                )

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

    test_predictions = best_model.predict(
        X_test
    )

    test_time = (
            time.perf_counter()
            - test_start
    )
    print(
        f"\nTest inference time: "
        f"{test_time:.4f} seconds"
    )

    print(
        f"Average inference time per image: "
        f"{test_time / len(X_test) * 1000:.4f} ms"
    )

    confusion = confusion_matrix(
        y_test,
        test_predictions,
        labels=np.arange(len(class_names)),
    )

    print("\nMost common confusion for each class")
    print("-" * 60)

    for class_index, class_name in enumerate(class_names):
        wrong_counts = confusion[class_index].copy()

        # 不统计预测正确的对角线
        wrong_counts[class_index] = 0

        confused_index = int(
            np.argmax(wrong_counts)
        )

        confused_count = int(
            wrong_counts[confused_index]
        )

        if confused_count == 0:
            print(
                f"{class_name:<15} -> "
                "no confusion"
            )
            continue

        class_total = int(
            confusion[class_index].sum()
        )

        confusion_rate = (
            confused_count / class_total * 100.0
            if class_total > 0
            else 0.0
        )

        print(
            f"{class_name:<15} -> "
            f"{class_names[confused_index]:<15} "
            f"{confused_count} "
            f"({confusion_rate:.1f}%)"
        )

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
        "rgb_channel_thresholds": [
            list(channel_thresholds)
            for channel_thresholds
            in rgb_thresholds
        ],
        "boolean_channels": 9,

        "boolean_channel_order": [
        "R_q25",
        "R_q50",
        "R_q75",
        "G_q25",
        "G_q50",
        "G_q75",
        "B_q25",
        "B_q50",
        "B_q75",
        ],

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
