"""One fair evaluation runner for MLP, BNN, XNOR-Popcount and Tsetlin."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import resolve_path
from src.data.loaders import create_dataloaders, set_seed
from src.data.preprocessing import tensor_batches_to_boolean
from src.data.split import build_or_load_manifest
from src.evaluation.acev2 import (
    calculate_tsetlin_extended_acev2,
    dense_layer_pairs,
    estimate_energy,
    extract_trained_clause_profile,
    model_acev2_profiles,
    packed_xnor_word_operations,
    patch_traversal_report,
)
from src.evaluation.exporters import export_results
from src.evaluation.metrics import classification_metrics
from src.evaluation.timing import benchmark_callable
from src.models.bnn import RGBDBNN
from src.models.mlp import RGBDMLP
from src.models.tsetlin import load_tsetlin_checkpoint


def _device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(name)


def _load_state(model, path: Path, device) -> None:
    import torch

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if hasattr(checkpoint, "state_dict"):
        state = checkpoint.state_dict()
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    model.load_state_dict(state)
    model.eval()


def _cache_batches(loader):
    batches, labels = [], []
    for images, batch_labels in loader:
        batches.append((images.contiguous(), batch_labels.contiguous()))
        labels.append(batch_labels.numpy())
    if not batches:
        raise RuntimeError("Test loader is empty")
    return batches, np.concatenate(labels)


def _torch_storage(model, checkpoint: Path, binary_weights: bool) -> dict:
    matrix_bits, other_bits = 0, 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        if binary_weights and parameter.ndim >= 2:
            matrix_bits += count
        else:
            other_bits += count * 32
    return {
        "checkpoint_size_mb": checkpoint.stat().st_size / (1024**2),
        "deployment_size_mb": (matrix_bits + other_bits) / 8 / (1024**2),
        "storage_basis": "binary matrix weights + FP32 bias/normalization" if binary_weights else "all parameters FP32",
    }


def _evaluate_torch(model, cached_batches, labels, class_names, device, evaluation_cfg):
    import torch

    device_batches = [(images.to(device), batch_labels.to(device)) for images, batch_labels in cached_batches]

    def infer():
        predictions = []
        with torch.inference_mode():
            for images, _ in device_batches:
                predictions.append(model(images).argmax(dim=1))
        return torch.cat(predictions)

    synchronize = torch.cuda.synchronize if device.type == "cuda" else None
    output, timing = benchmark_callable(
        infer,
        sample_count=len(labels),
        warmup_runs=int(evaluation_cfg["warmup_runs"]),
        repeats=int(evaluation_cfg["timed_repeats"]),
        synchronize=synchronize,
    )
    predictions = output.detach().cpu().numpy()
    return predictions, classification_metrics(labels, predictions, class_names), timing


def evaluate_models(config: dict, project_root: Path, selected_models: list[str] | None = None, regenerate_split=False):
    import torch

    seed = int(config["project"]["seed"])
    set_seed(seed)
    dataset_root = resolve_path(project_root, config["project"]["data_root"])
    manifest_path = resolve_path(project_root, config["data"]["manifest_path"])
    manifest = build_or_load_manifest(
        dataset_root,
        manifest_path,
        ratios=config["data"]["split_ratios"],
        seed=seed,
        split_unit=config["data"].get("split_unit", "image"),
        regenerate=regenerate_split,
    )
    loaders, class_names = create_dataloaders(config, dataset_root, manifest)
    cached_batches, labels = _cache_batches(loaders["test"])
    evaluation_cfg = config["evaluation"]
    names = selected_models or list(evaluation_cfg["models"])
    unsupported = set(names).difference({"mlp", "bnn", "bnn_xnor", "tsetlin"})
    if unsupported:
        raise ValueError(f"Unsupported models: {sorted(unsupported)}")
    device = _device(str(evaluation_cfg.get("device", "cpu")))
    image_size = int(config["data"]["image_size"])
    hidden_sizes = tuple(map(int, config["models"]["hidden_sizes"]))
    layers = dense_layer_pairs(image_size, hidden_sizes, len(class_names))
    energy_coefficients = evaluation_cfg.get("energy", {})
    results, bnn_predictions = [], None

    for name in names:
        if name in {"mlp", "bnn", "bnn_xnor"}:
            if name == "mlp":
                model = RGBDMLP(image_size, len(class_names), hidden_sizes).to(device)
                checkpoint = resolve_path(project_root, config["models"]["mlp_checkpoint"])
                operation, binary = "FP32 dense", False
            else:
                model = RGBDBNN(image_size, len(class_names), hidden_sizes, use_xnor=name == "bnn_xnor").to(device)
                checkpoint = resolve_path(project_root, config["models"]["bnn_checkpoint"])
                operation, binary = ("Boolean-tensor XNOR + popcount" if name == "bnn_xnor" else "binary-valued weights in FP32 Linear"), True
            _load_state(model, checkpoint, device)
            predictions, metrics, timing = _evaluate_torch(model, cached_batches, labels, class_names, device, evaluation_cfg)
            if name == "bnn":
                bnn_predictions = predictions
            if name == "bnn_xnor" and bnn_predictions is not None and not np.array_equal(predictions, bnn_predictions):
                raise RuntimeError("BNN and XNOR predictions differ; weights or inference semantics are inconsistent")
            cost = model_acev2_profiles(name, layers)
            if name == "bnn_xnor":
                cost["packed_xnor_word_operations"] = packed_xnor_word_operations(
                    layers, int(evaluation_cfg.get("xnor_word_bits", 64)), include_first_layer=False
                )
                cost["packed_word_note"] = "ideal hardware word operations for binary hidden layers; not Python operations or ACEv2"
            result = {
                "model": name,
                "operation": operation,
                "metrics": metrics,
                "timing": timing,
                "storage": _torch_storage(model, checkpoint, binary),
                "cost": cost,
                "energy": estimate_energy(cost, energy_coefficients),
            }
            results.append(result)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        model_path = resolve_path(project_root, config["models"]["tsetlin_checkpoint"])
        model, metadata = load_tsetlin_checkpoint(model_path)
        saved_classes = metadata.get("class_names")
        if saved_classes is not None and list(saved_classes) != class_names:
            raise RuntimeError("Tsetlin checkpoint class order does not match the split manifest")
        thresholds = metadata.get("boolean_thresholds", config["preprocessing"]["tsetlin_thresholds"])
        x_test, y_test = tensor_batches_to_boolean(cached_batches, thresholds)

        def infer_tsetlin():
            return model.predict(x_test)

        predictions, timing = benchmark_callable(
            infer_tsetlin, len(y_test), int(evaluation_cfg["warmup_runs"]), int(evaluation_cfg["timed_repeats"])
        )
        tm_cfg = config["models"]["tsetlin"]
        patch_dim = tuple(metadata.get("patch_dim", tm_cfg["patch_dim"]))
        clause_profile = extract_trained_clause_profile(model)
        if clause_profile is not None and clause_profile["class_count"] != len(class_names):
            raise RuntimeError(
                "Tsetlin Clause Bank class count does not match the split manifest: "
                f"{clause_profile['class_count']} != {len(class_names)}"
            )
        clauses = int(
            clause_profile["clauses_per_class"]
            if clause_profile is not None
            else metadata.get("number_of_clauses", metadata.get("clauses", tm_cfg["clauses"]))
        )
        flat_clause_lengths = (
            list(clause_profile["_flat_lengths"])
            if clause_profile is not None
            else None
        )
        public_clause_profile = (
            {key: value for key, value in clause_profile.items() if key != "_flat_lengths"}
            if clause_profile is not None
            else None
        )
        traversal = patch_traversal_report(model, public_clause_profile)
        cost = calculate_tsetlin_extended_acev2(
            image_shape=(image_size, image_size),
            boolean_channels=int(x_test.shape[-1]),
            patch_dim=(int(patch_dim[0]), int(patch_dim[1])),
            class_count=len(class_names),
            clauses_per_class=clauses,
            max_included_literals=int(
                metadata.get("max_included_literals", tm_cfg["max_included_literals"])
            ),
            weighted_clauses=bool(
                metadata.get("weighted_clauses", tm_cfg.get("weighted_clauses", True))
            ),
            accumulator_bits=int(
                metadata.get(
                    "vote_accumulator_bits",
                    evaluation_cfg.get("tsetlin_accumulator_bits", 32),
                )
            ),
            weight_bits=int(
                metadata.get(
                    "clause_weight_bits",
                    evaluation_cfg.get("tsetlin_weight_bits", 32),
                )
            ),
            logic_gate_fae=float(evaluation_cfg.get("tsetlin_logic_gate_fae", 0.2)),
            trained_clause_lengths=flat_clause_lengths,
            model_patch_count=(
                int(public_clause_profile["model_number_of_patches"])
                if public_clause_profile is not None
                else None
            ),
        )
        cost["trained_clause_profile"] = public_clause_profile
        cost["patch_evaluation"] = traversal
        cost["clause_lengths_source"] = (
            "trained Tsetlin Automata include actions"
            if public_clause_profile is not None
            else "unavailable; configured upper bound used"
        )
        results.append({
            "model": "tsetlin",
            "operation": "Boolean clauses and class voting",
            "metrics": classification_metrics(y_test, predictions, class_names),
            "timing": timing,
            "storage": {
                "checkpoint_size_mb": model_path.stat().st_size / (1024**2),
                "deployment_size_mb": None,
                "storage_basis": "TMU pickle size; a packed hardware-state size requires exported clause states",
            },
            "cost": cost,
            "energy": estimate_energy(cost, energy_coefficients),
        })

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset_root),
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": manifest["sha256"],
        "split_unit": manifest["split_unit"],
        "class_names": class_names,
        "image_size": image_size,
        "input_mode": config["preprocessing"]["input_mode"],
        "device": str(device),
        "results": results,
    }
    output_dir = resolve_path(project_root, evaluation_cfg["output_dir"])
    paths = export_results(payload, output_dir)
    return payload, paths
