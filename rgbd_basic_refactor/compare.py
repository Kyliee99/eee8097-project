# -*- coding: utf-8 -*-
"""
公共模型比较文件。

默认比较四种算法场景：
1. MLP
2. BNN（BinaryLinear 二值点积）
3. BNN（XNOR + Popcount 推理）
4. Tsetlin Machine

所有模型使用同一个 RGBDObjectDataset 测试集。
默认使用 CPU，便于与 CPU Tsetlin Machine 进行较公平的比较。

运行：
    python compare.py

只比较指定模型：
    python compare.py --models mlp bnn
    python compare.py --models mlp bnn bnn_xnor tsetlin

指定模型路径：
    python compare.py ^
        --mlp-model models/rgbd_mlp_2.pth ^
        --bnn-model models/rgbd_bnn_2.pth ^
        --tsetlin-model tsetlin_output/best_tsetlin_rgbd.pkl
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from bnn import RGBDBNN
from common import (
    BATCH_SIZE,
    DATASET_ROOT,
    IMAGE_SIZE,
    build_classification_metrics,
    create_dataloaders,
    set_seed,
)
from mlp import RGBDMLP


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_MLP_MODEL = (
    SCRIPT_DIR
    / "models"
    / "rgbd_mlp_2.pth"
)

DEFAULT_BNN_MODEL = (
    SCRIPT_DIR
    / "models"
    / "rgbd_bnn_2.pth"
)

DEFAULT_TSETLIN_MODEL = (
    SCRIPT_DIR
    / "tsetlin_output"
    / "best_tsetlin_rgbd.pkl"
)

DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR
    / "comparison_results"
)

SUPPORTED_MODELS = (
    "mlp",
    "bnn",
    "bnn_xnor",
    "tsetlin",
)


# ============================================================
# 1. 命令行参数
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare MLP, BNN, BNN-XNOR/Popcount "
            "and Tsetlin Machine on the same test set."
        )
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=SUPPORTED_MODELS,
        default=list(SUPPORTED_MODELS),
        help=(
            "要比较的模型。默认："
            "mlp bnn bnn_xnor tsetlin"
        ),
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="RGB-D Object Dataset 根目录。",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="测试 batch size。",
    )

    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help=(
            "PyTorch模型使用的设备。默认CPU，"
            "便于与CPU Tsetlin公平比较。"
        ),
    )

    parser.add_argument(
        "--mlp-model",
        type=Path,
        default=DEFAULT_MLP_MODEL,
        help="MLP .pth 文件。",
    )

    parser.add_argument(
        "--bnn-model",
        type=Path,
        default=DEFAULT_BNN_MODEL,
        help="BNN .pth 文件。",
    )

    parser.add_argument(
        "--tsetlin-model",
        type=Path,
        default=DEFAULT_TSETLIN_MODEL,
        help="Tsetlin .pkl 文件。",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="比较结果保存目录。",
    )

    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cpu":
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "指定了CUDA，但当前环境没有可用CUDA设备。"
            )
        return torch.device("cuda")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# 2. 模型参数加载
# ============================================================

def load_torch_weights(
    model: nn.Module,
    model_path: Path,
    device: torch.device,
) -> None:
    """
    兼容三种常见保存形式：

    1. torch.save(model.state_dict(), path)
    2. {"model_state_dict": state_dict}
    3. {"state_dict": state_dict}
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"模型文件不存在：{model_path.resolve()}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()

    elif isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint[
                "model_state_dict"
            ]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

    else:
        raise ValueError(
            f"无法识别模型文件格式：{model_path}"
        )

    model.load_state_dict(state_dict)
    model.eval()


# ============================================================
# 3. 测试数据缓存
# ============================================================

def cache_test_batches(
    testloader,
) -> tuple[
    list[tuple[torch.Tensor, torch.Tensor]],
    np.ndarray,
]:
    """
    先把测试集读取到内存。

    这样计时时不包含硬盘读取和PIL预处理时间，
    更接近模型自身的推理时间。
    """
    cached_batches = []
    labels_all = []

    for images, labels in testloader:
        cached_batches.append((
            images.contiguous(),
            labels.contiguous(),
        ))
        labels_all.append(
            labels.numpy()
        )

    if not cached_batches:
        raise RuntimeError(
            "测试集中没有读取到任何数据。"
        )

    return (
        cached_batches,
        np.concatenate(
            labels_all,
            axis=0,
        ),
    )


# ============================================================
# 4. PyTorch模型测试
# ============================================================

def evaluate_torch_model(
    model: nn.Module,
    cached_batches,
    class_names: list[str],
    device: torch.device,
) -> dict[str, Any]:
    predictions_all = []
    labels_all = []

    model.eval()

    # 预热一次，减少首次调用初始化对时间的影响。
    with torch.no_grad():
        first_images, _ = cached_batches[0]
        _ = model(
            first_images.to(device)
        )

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        for images, labels in cached_batches:
            images = images.to(
                device,
                non_blocking=True,
            )

            outputs = model(images)

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            predictions_all.extend(
                predictions.cpu().numpy()
            )
            labels_all.extend(
                labels.numpy()
            )

    if device.type == "cuda":
        torch.cuda.synchronize()

    total_time = (
        time.perf_counter()
        - start_time
    )

    return build_classification_metrics(
        labels_all,
        predictions_all,
        class_names,
        total_time=total_time,
    )


# ============================================================
# 5. Tsetlin测试数据转换
# ============================================================

def cached_batches_to_boolean_numpy(
    cached_batches,
    thresholds: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """
    将缓存的PyTorch灰度数据转换成Tsetlin的Boolean输入。

    输入：
        [B,1,32,32]

    输出：
        [N,32,32,threshold_count]
    """
    feature_batches = []
    label_batches = []

    with torch.no_grad():
        for images, labels in cached_batches:
            boolean_planes = [
                (
                    images > threshold
                ).to(torch.uint32)
                for threshold in thresholds
            ]

            boolean_tensor = torch.cat(
                boolean_planes,
                dim=1,
            )

            boolean_tensor = (
                boolean_tensor
                .permute(0, 2, 3, 1)
                .contiguous()
            )

            feature_batches.append(
                boolean_tensor
                .cpu()
                .numpy()
            )

            label_batches.append(
                labels
                .cpu()
                .numpy()
                .astype(
                    np.uint32,
                    copy=False,
                )
            )

    X_test = np.concatenate(
        feature_batches,
        axis=0,
    )
    y_test = np.concatenate(
        label_batches,
        axis=0,
    )

    return (
        np.ascontiguousarray(
            X_test,
            dtype=np.uint32,
        ),
        np.ascontiguousarray(
            y_test,
            dtype=np.uint32,
        ),
    )


def evaluate_tsetlin_model(
    model_path: Path,
    cached_batches,
    class_names: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not model_path.exists():
        raise FileNotFoundError(
            "Tsetlin模型文件不存在："
            f"{model_path.resolve()}"
        )

    # pickle只能加载可信的模型文件。
    with model_path.open("rb") as file:
        checkpoint = pickle.load(file)

    if isinstance(checkpoint, dict):
        if "model" not in checkpoint:
            raise KeyError(
                "Tsetlin checkpoint中没有'model'。"
            )

        model = checkpoint["model"]

        saved_classes = checkpoint.get(
            "class_names"
        )

        if (
            saved_classes is not None
            and list(saved_classes)
            != list(class_names)
        ):
            raise RuntimeError(
                "Tsetlin模型类别顺序与测试集不一致。"
            )

        thresholds = tuple(
            checkpoint.get(
                "boolean_thresholds",
                (-0.5, 0.0, 0.5),
            )
        )

        model_metadata = {
            "best_epoch": checkpoint.get(
                "epoch"
            ),
            "best_validation_accuracy": (
                checkpoint.get(
                    "validation_accuracy"
                )
            ),
            "boolean_thresholds": list(
                thresholds
            ),
            "patch_dim": checkpoint.get(
                "patch_dim"
            ),
        }

    else:
        # 兼容直接pickle.dump(model)
        model = checkpoint
        thresholds = (
            -0.5,
            0.0,
            0.5,
        )
        model_metadata = {
            "boolean_thresholds": list(
                thresholds
            )
        }

    X_test, y_test = (
        cached_batches_to_boolean_numpy(
            cached_batches,
            thresholds,
        )
    )

    # 预热一次。
    warmup_count = min(
        1,
        len(X_test),
    )

    if warmup_count > 0:
        _ = model.predict(
            X_test[:warmup_count]
        )

    start_time = time.perf_counter()

    predictions = model.predict(X_test)

    total_time = (
        time.perf_counter()
        - start_time
    )

    metrics = build_classification_metrics(
        y_test,
        predictions,
        class_names,
        total_time=total_time,
    )

    return metrics, model_metadata


# ============================================================
# 6. 参数、权重和文件大小
# ============================================================

def count_total_parameters(
    model: nn.Module,
) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def count_matrix_weights(
    model: nn.Module,
) -> int:
    """
    只统计Linear等二维权重，
    不统计bias和BatchNorm的一维参数。
    """
    return sum(
        parameter.numel()
        for name, parameter
        in model.named_parameters()
        if (
            "weight" in name
            and parameter.dim() >= 2
        )
    )


def get_file_size_mb(
    file_path: Path,
) -> float:
    if not file_path.exists():
        return 0.0

    return (
        file_path.stat().st_size
        / (1024 * 1024)
    )


def theoretical_size_mb(
    weight_count: int,
    bits_per_weight: int,
) -> float:
    return (
        weight_count
        * bits_per_weight
        / 8
        / (1024 * 1024)
    )


def add_common_statistics(
    model_name: str,
    model_path: Path,
    metrics: dict[str, Any],
    model: nn.Module | None = None,
    bits_per_weight: int | None = None,
    operation: str = "",
) -> dict[str, Any]:
    result = {
        "model": model_name,
        "operation": operation,
        **metrics,
        "model_file_size_mb": (
            get_file_size_mb(
                model_path
            )
        ),
    }

    if model is not None:
        total_parameters = (
            count_total_parameters(model)
        )
        matrix_weights = (
            count_matrix_weights(model)
        )

        result.update({
            "total_parameters": (
                total_parameters
            ),
            "matrix_weights": (
                matrix_weights
            ),
        })

        if bits_per_weight is not None:
            result[
                "theoretical_weight_size_mb"
            ] = theoretical_size_mb(
                matrix_weights,
                bits_per_weight,
            )

    return result


# ============================================================
# 7. 输出比较表
# ============================================================

def value_or_dash(
    value,
    number_format: str,
) -> str:
    if value is None:
        return "-"

    return format(value, number_format)


def print_comparison_table(
    results: list[dict[str, Any]],
) -> None:
    print("\n")
    print("=" * 140)
    print("不同算法场景性能比较")
    print("=" * 140)

    header = (
        f"{'模型':<18}"
        f"{'Accuracy':>11}"
        f"{'Precision':>11}"
        f"{'Recall':>11}"
        f"{'F1':>11}"
        f"{'总时间(s)':>13}"
        f"{'单张(ms)':>13}"
        f"{'吞吐(img/s)':>15}"
        f"{'文件(MB)':>12}"
        f"{'理论权重(MB)':>16}"
    )

    print(header)
    print("-" * 140)

    for result in results:
        total_time = result.get(
            "total_time",
            0.0,
        )

        time_per_image = result.get(
            "time_per_image",
            0.0,
        )

        throughput = (
            1.0 / time_per_image
            if time_per_image > 0
            else 0.0
        )

        theoretical_size = result.get(
            "theoretical_weight_size_mb"
        )

        print(
            f"{result['model']:<18}"
            f"{result['accuracy'] * 100:>10.2f}%"
            f"{result['precision']:>11.4f}"
            f"{result['recall']:>11.4f}"
            f"{result['f1']:>11.4f}"
            f"{total_time:>13.6f}"
            f"{time_per_image * 1000:>13.6f}"
            f"{throughput:>15.2f}"
            f"{result['model_file_size_mb']:>12.4f}"
            f"{value_or_dash(theoretical_size, '.4f'):>16}"
        )

    print("=" * 140)


def print_detailed_reports(
    results: list[dict[str, Any]],
    class_names: list[str],
) -> None:
    for result in results:
        print(
            f"\n{result['model']} 详细分类报告"
        )
        print("-" * 60)

        report = result[
            "classification_report"
        ]

        # classification_report已保存为字典，
        # 这里手动打印核心类别指标。
        print(
            f"{'类别':<20}"
            f"{'Precision':>12}"
            f"{'Recall':>12}"
            f"{'F1-score':>12}"
            f"{'Support':>10}"
        )

        for class_name in class_names:
            class_result = report[
                class_name
            ]

            print(
                f"{class_name:<20}"
                f"{class_result['precision']:>12.4f}"
                f"{class_result['recall']:>12.4f}"
                f"{class_result['f1-score']:>12.4f}"
                f"{int(class_result['support']):>10}"
            )


# ============================================================
# 8. 保存CSV和JSON
# ============================================================

def save_results(
    results: list[dict[str, Any]],
    output_dir: Path,
    dataset_root: Path,
    class_names: list[str],
    device: torch.device,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_dir
        / "comparison_results.json"
    )

    csv_path = (
        output_dir
        / "comparison_summary.csv"
    )

    json_data = {
        "dataset_root": str(
            dataset_root.resolve()
        ),
        "image_size": IMAGE_SIZE,
        "class_names": class_names,
        "torch_device": str(device),
        "results": results,
    }

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    csv_fields = [
        "model",
        "operation",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "total_time",
        "time_per_image",
        "throughput_images_per_second",
        "total_parameters",
        "matrix_weights",
        "model_file_size_mb",
        "theoretical_weight_size_mb",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        for result in results:
            time_per_image = result.get(
                "time_per_image",
                0.0,
            )

            row = {
                field: result.get(field)
                for field in csv_fields
            }

            row[
                "throughput_images_per_second"
            ] = (
                1.0 / time_per_image
                if time_per_image > 0
                else 0.0
            )

            writer.writerow(row)

    print("\nComparison JSON:", json_path.resolve())
    print("Comparison CSV:", csv_path.resolve())


# ============================================================
# 9. 主程序
# ============================================================

def main() -> None:
    args = parse_arguments()
    set_seed()

    device = resolve_device(
        args.device
    )

    print("PyTorch device:", device)

    if (
        device.type == "cuda"
        and "tsetlin" in args.models
    ):
        print(
            "Warning: PyTorch模型使用CUDA，"
            "Tsetlin仍可能使用CPU，"
            "推理时间不能视为完全公平比较。"
        )

    _, _, testloader, class_names = (
        create_dataloaders(
            dataset_root=args.dataset_root,
            batch_size=args.batch_size,
            image_size=IMAGE_SIZE,
        )
    )

    cached_batches, _ = (
        cache_test_batches(
            testloader
        )
    )

    results: list[dict[str, Any]] = []

    # --------------------------------------------------------
    # MLP
    # --------------------------------------------------------

    if "mlp" in args.models:
        print("\nTesting MLP...")

        mlp_model = RGBDMLP(
            num_classes=len(class_names)
        ).to(device)

        load_torch_weights(
            mlp_model,
            args.mlp_model,
            device,
        )

        mlp_metrics = evaluate_torch_model(
            mlp_model,
            cached_batches,
            class_names,
            device,
        )

        results.append(
            add_common_statistics(
                model_name="MLP",
                model_path=args.mlp_model,
                metrics=mlp_metrics,
                model=mlp_model,
                bits_per_weight=32,
                operation="FP32 Linear",
            )
        )

    # --------------------------------------------------------
    # BNN：BinaryLinear
    # --------------------------------------------------------

    if "bnn" in args.models:
        print("\nTesting BNN...")

        bnn_model = RGBDBNN(
            num_classes=len(class_names),
            use_xnor_popcount=False,
        ).to(device)

        load_torch_weights(
            bnn_model,
            args.bnn_model,
            device,
        )

        bnn_metrics = evaluate_torch_model(
            bnn_model,
            cached_batches,
            class_names,
            device,
        )

        results.append(
            add_common_statistics(
                model_name="BNN",
                model_path=args.bnn_model,
                metrics=bnn_metrics,
                model=bnn_model,
                bits_per_weight=1,
                operation=(
                    "Binary weight dot product"
                ),
            )
        )

    # --------------------------------------------------------
    # BNN：XNOR + Popcount
    # --------------------------------------------------------

    if "bnn_xnor" in args.models:
        print(
            "\nTesting BNN XNOR+Popcount..."
        )

        bnn_xnor_model = RGBDBNN(
            num_classes=len(class_names),
            use_xnor_popcount=True,
        ).to(device)

        load_torch_weights(
            bnn_xnor_model,
            args.bnn_model,
            device,
        )

        bnn_xnor_metrics = (
            evaluate_torch_model(
                bnn_xnor_model,
                cached_batches,
                class_names,
                device,
            )
        )

        results.append(
            add_common_statistics(
                model_name="BNN-XNOR",
                model_path=args.bnn_model,
                metrics=bnn_xnor_metrics,
                model=bnn_xnor_model,
                bits_per_weight=1,
                operation="XNOR + Popcount",
            )
        )

    # --------------------------------------------------------
    # Tsetlin Machine
    # --------------------------------------------------------

    if "tsetlin" in args.models:
        print("\nTesting Tsetlin Machine...")

        tsetlin_metrics, metadata = (
            evaluate_tsetlin_model(
                args.tsetlin_model,
                cached_batches,
                class_names,
            )
        )

        tsetlin_result = (
            add_common_statistics(
                model_name="Tsetlin",
                model_path=(
                    args.tsetlin_model
                ),
                metrics=tsetlin_metrics,
                operation=(
                    "Clause-based Boolean inference"
                ),
            )
        )

        tsetlin_result.update(
            metadata
        )

        results.append(
            tsetlin_result
        )

    if not results:
        raise RuntimeError(
            "没有选择任何可测试模型。"
        )

    print_comparison_table(results)

    print_detailed_reports(
        results,
        class_names,
    )

    save_results(
        results,
        args.output_dir,
        args.dataset_root,
        class_names,
        device,
    )


if __name__ == "__main__":
    main()
