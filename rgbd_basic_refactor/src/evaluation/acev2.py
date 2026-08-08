"""ACEv2 arithmetic profiles and a separate non-standard Tsetlin logic estimate."""

from __future__ import annotations

import math
from collections import Counter
from statistics import fmean, median
from typing import Any, Iterable


def mac_cost(count: int, activation_bits: int, weight_bits: int) -> int:
    """Original ACE MAC term used by ACEv2: N * activation_bits * weight_bits."""
    return int(count) * int(activation_bits) * int(weight_bits)


def multiply_cost(count: int, left_bits: int, right_bits: int) -> int:
    """ACEv2 elementwise multiplication: N * (i*j - max(i,j))."""
    return int(count) * max(int(left_bits) * int(right_bits) - max(int(left_bits), int(right_bits)), 0)


def addition_cost(count: int, left_bits: int, right_bits: int, floating_point: bool = False) -> int:
    multiplier = 6 if floating_point else 1
    return int(count) * multiplier * max(int(left_bits), int(right_bits))


def dense_layer_pairs(image_size: int, hidden_sizes: Iterable[int], num_classes: int) -> list[tuple[int, int]]:
    dimensions = [image_size * image_size, *map(int, hidden_sizes), int(num_classes)]
    return list(zip(dimensions, dimensions[1:]))


def neural_acev2_profile(
    layers: list[tuple[int, int]],
    activation_bits: list[int],
    weight_bits: list[int],
    bias_outputs: list[int] | None = None,
    batchnorm_elements: int = 0,
    include_fp32_batchnorm: bool = False,
) -> dict:
    if not (len(layers) == len(activation_bits) == len(weight_bits)):
        raise ValueError("layers, activation_bits and weight_bits must have equal length")
    macs = [inputs * outputs for inputs, outputs in layers]
    mac_terms = [mac_cost(count, a_bits, w_bits) for count, a_bits, w_bits in zip(macs, activation_bits, weight_bits)]
    bias_count = sum(bias_outputs or [])
    bias_term = addition_cost(bias_count, 32, 32, floating_point=True)
    bn_multiply = multiply_cost(batchnorm_elements, 32, 32) if include_fp32_batchnorm else 0
    bn_add = addition_cost(batchnorm_elements, 32, 32, floating_point=True) if include_fp32_batchnorm else 0
    total = sum(mac_terms) + bias_term + bn_multiply + bn_add
    return {
        "macs_per_image": sum(macs),
        "bias_additions_per_image": bias_count,
        "batchnorm_elements_per_image": int(batchnorm_elements if include_fp32_batchnorm else 0),
        "mac_acev2": sum(mac_terms),
        "bias_acev2": bias_term,
        "batchnorm_acev2": bn_multiply + bn_add,
        "acev2": total,
        "acev2_million": total / 1e6,
        "activation_bits_by_layer": list(map(int, activation_bits)),
        "weight_bits_by_layer": list(map(int, weight_bits)),
        "batchnorm_fused": not include_fp32_batchnorm,
    }


def model_acev2_profiles(model_name: str, layers: list[tuple[int, int]]) -> dict:
    layer_count = len(layers)
    hidden_outputs = [output for _, output in layers[:-1]]
    if model_name == "mlp":
        algorithmic = neural_acev2_profile(layers, [32] * layer_count, [32] * layer_count, [o for _, o in layers])
        python_tensor = algorithmic.copy()
    elif model_name in {"bnn", "bnn_xnor"}:
        # Deployment profile: continuous input in layer 1, binary hidden activations thereafter.
        # Bias + BatchNorm + sign can be folded into deployment thresholds.
        algorithmic = neural_acev2_profile(layers, [32] + [1] * (layer_count - 1), [1] * layer_count)
        if model_name == "bnn":
            # BinaryLinear still dispatches FP32 linear algebra on FP32 tensors.
            python_tensor = neural_acev2_profile(
                layers, [32] * layer_count, [32] * layer_count, hidden_outputs,
                batchnorm_elements=sum(hidden_outputs), include_fp32_batchnorm=True,
            )
        else:
            # torch.eq plus int32 reductions are runtime operations, not standard ACEv2 MACs.
            python_tensor = None
    else:
        raise ValueError(f"ACEv2 arithmetic profile is not defined for: {model_name}")
    return {
        "acev2_status": "standard arithmetic proxy; not measured energy",
        "algorithmic_acev2": algorithmic,
        "python_tensor_acev2": python_tensor,
        "python_cost_status": (
            "FP32 tensor arithmetic represented by ACEv2"
            if python_tensor is not None
            else "not represented by ACEv2: Python path uses Boolean comparisons and int32 reductions"
        ),
    }


def packed_xnor_word_operations(layers: list[tuple[int, int]], word_bits: int = 64, include_first_layer: bool = False) -> int:
    selected = layers if include_first_layer else layers[1:]
    return sum(math.ceil(inputs / word_bits) * outputs for inputs, outputs in selected)


def _summarize_clause_lengths(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        raise ValueError("Clause-length list must not be empty")
    histogram = Counter(map(int, lengths))
    return {
        "clause_count": len(lengths),
        "included_literals_total": int(sum(lengths)),
        "included_literals_min": int(min(lengths)),
        "included_literals_mean": float(fmean(lengths)),
        "included_literals_median": float(median(lengths)),
        "included_literals_max": int(max(lengths)),
        "empty_clause_count": int(sum(value == 0 for value in lengths)),
        "nonempty_clause_count": int(sum(value > 0 for value in lengths)),
        "length_histogram": {
            str(length): int(count)
            for length, count in sorted(histogram.items())
        },
    }


def extract_trained_clause_profile(model: object | None) -> dict[str, Any] | None:
    """Read the real include actions from every class-specific TMU Clause Bank."""
    if model is None:
        return None
    container = getattr(model, "clause_banks", None)
    if container is None:
        return None

    classes_method = getattr(container, "classes", None)
    class_ids = list(classes_method()) if callable(classes_method) else list(container)
    if not class_ids:
        return None

    all_lengths: list[int] = []
    per_class: dict[str, Any] = {}
    bank_types: set[str] = set()
    patch_counts: set[int] = set()
    clause_counts: set[int] = set()
    incremental_flags: set[bool] = set()

    for class_id in class_ids:
        bank = container[class_id]
        bank_types.add(f"{bank.__class__.__module__}.{bank.__class__.__name__}")
        patch_counts.add(int(getattr(bank, "number_of_patches")))
        clause_count = int(getattr(bank, "number_of_clauses"))
        clause_counts.add(clause_count)
        incremental_flags.add(
            bool(getattr(bank, "incremental", getattr(model, "incremental", False)))
        )

        include_count = getattr(bank, "number_of_include_actions", None)
        if callable(include_count):
            lengths = [int(include_count(clause)) for clause in range(clause_count)]
            extraction_method = "ClauseBank.number_of_include_actions"
        else:
            get_literals = getattr(bank, "get_literals", None)
            if not callable(get_literals):
                raise RuntimeError(
                    f"Clause Bank for class {class_id} exposes neither "
                    "number_of_include_actions() nor get_literals()."
                )
            literal_matrix = get_literals()
            lengths = [int(sum(int(value) for value in row)) for row in literal_matrix]
            extraction_method = "ClauseBank.get_literals"

        all_lengths.extend(lengths)
        per_class[str(class_id)] = {
            **_summarize_clause_lengths(lengths),
            "extraction_method": extraction_method,
        }

    if len(patch_counts) != 1:
        raise RuntimeError(f"Class-specific Clause Banks disagree on patch count: {patch_counts}")
    if len(clause_counts) != 1:
        raise RuntimeError(f"Class-specific Clause Banks disagree on Clause count: {clause_counts}")

    return {
        **_summarize_clause_lengths(all_lengths),
        "class_count": len(class_ids),
        "clauses_per_class": next(iter(clause_counts)),
        "model_number_of_patches": next(iter(patch_counts)),
        "bank_types": sorted(bank_types),
        "incremental_flags": sorted(incremental_flags),
        "per_class": per_class,
        "_flat_lengths": all_lengths,
    }


def patch_traversal_report(
    model: object | None,
    clause_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    """Report whether the loaded TMU backend visits every patch during prediction."""
    if model is None or clause_profile is None:
        return {
            "status": "unavailable: checkpoint model was not loaded",
            "all_patches_visited": None,
        }

    platform = str(getattr(model, "platform", "unknown")).upper()
    incremental_flags = clause_profile["incremental_flags"]
    incremental = incremental_flags == [True]
    dense_cpu = (
        platform == "CPU"
        and all(bank_type.endswith(".ClauseBank") for bank_type in clause_profile["bank_types"])
    )
    patches = int(clause_profile["model_number_of_patches"])
    classes = int(clause_profile["class_count"])
    clauses = int(clause_profile["clauses_per_class"])

    if dense_cpu and incremental:
        return {
            "status": "confirmed from loaded model mode and TMU dense CPU incremental control flow",
            "platform": platform,
            "backend": clause_profile["bank_types"],
            "incremental": True,
            "all_patches_visited": True,
            "patches_available_per_class_per_image": patches,
            "patches_visited_per_class_per_image": patches,
            "class_patch_visits_per_image": classes * patches,
            "clause_patch_checks_per_image": classes * clauses * patches,
            "early_exit_across_patches": False,
        }

    if dense_cpu and incremental_flags == [False]:
        return {
            "status": (
                "dynamic: dense CPU non-incremental prediction can stop after the first "
                "satisfying patch; TMU C-level counters are required for an exact mean"
            ),
            "platform": platform,
            "backend": clause_profile["bank_types"],
            "incremental": False,
            "all_patches_visited": None,
            "patches_available_per_class_per_image": patches,
            "patches_visited_per_class_per_image": None,
            "early_exit_across_patches": True,
        }

    return {
        "status": "unknown for this TMU platform/backend; C/CUDA instrumentation is required",
        "platform": platform,
        "backend": clause_profile["bank_types"],
        "incremental": incremental_flags,
        "all_patches_visited": None,
        "patches_available_per_class_per_image": patches,
        "patches_visited_per_class_per_image": None,
    }


def calculate_tsetlin_extended_acev2(
    *,
    image_shape: tuple[int, int],
    boolean_channels: int,
    patch_dim: tuple[int, int],
    class_count: int,
    clauses_per_class: int,
    max_included_literals: int,
    weighted_clauses: bool = True,
    accumulator_bits: int = 32,
    weight_bits: int = 32,
    logic_gate_fae: float = 0.2,
    trained_clause_lengths: list[int] | None = None,
    model_patch_count: int | None = None,
) -> dict[str, Any]:
    """Calculate the explicit full-adder-equivalent ACEv2 extension for a TM."""
    height, width = map(int, image_shape)
    patch_height, patch_width = map(int, patch_dim)
    positive_values = {
        "image height": height,
        "image width": width,
        "Boolean channels": boolean_channels,
        "class count": class_count,
        "clauses per class": clauses_per_class,
        "max included literals": max_included_literals,
        "accumulator bits": accumulator_bits,
        "weight bits": weight_bits,
    }
    invalid = [name for name, value in positive_values.items() if int(value) <= 0]
    if invalid:
        raise ValueError(f"These values must be positive: {', '.join(invalid)}")
    if min(patch_height, patch_width) <= 0 or patch_height > height or patch_width > width:
        raise ValueError("patch dimensions must be positive and fit inside the image")
    if logic_gate_fae < 0:
        raise ValueError("logic_gate_fae must be non-negative")

    patches = (height - patch_height + 1) * (width - patch_width + 1)
    if model_patch_count is not None and int(model_patch_count) != patches:
        raise ValueError(
            "Checkpoint Clause Bank patch count does not match image/patch configuration: "
            f"{model_patch_count} != {patches}"
        )
    features_per_patch = patch_height * patch_width * int(boolean_channels)
    available_literals = 2 * features_per_patch
    included_literals = min(available_literals, int(max_included_literals))
    clause_count = int(class_count) * int(clauses_per_class)

    # Negations are generated once per Boolean image and shared by all patches.
    input_negations = height * width * int(boolean_channels)
    clause_and = clause_count * patches * max(included_literals - 1, 0)
    patch_or = clause_count * max(patches - 1, 0)
    boolean_gate_ops = input_negations + clause_and + patch_or
    boolean_logic_fae = boolean_gate_ops * float(logic_gate_fae)
    vote_additions = int(class_count) * max(int(clauses_per_class) - 1, 0)
    vote_acev2 = vote_additions * int(accumulator_bits)
    weighted_selection_muxes = clause_count * int(weight_bits) if weighted_clauses else 0
    weighted_selection_fae = weighted_selection_muxes * float(logic_gate_fae)
    extended_upper = boolean_logic_fae + vote_acev2 + weighted_selection_fae

    trained_cost: dict[str, Any] | None = None
    if trained_clause_lengths is not None:
        lengths = list(map(int, trained_clause_lengths))
        if len(lengths) != clause_count:
            raise ValueError(
                "Actual Clause-length count does not match class_count * clauses_per_class: "
                f"{len(lengths)} != {clause_count}"
            )
        if min(lengths, default=0) < 0 or max(lengths, default=0) > available_literals:
            raise ValueError("Actual Clause lengths fall outside the available-literal range")
        nonempty_clauses = sum(length > 0 for length in lengths)
        trained_and = patches * sum(max(length - 1, 0) for length in lengths)
        trained_or = nonempty_clauses * max(patches - 1, 0)
        trained_gate_ops = input_negations + trained_and + trained_or
        trained_logic_fae = trained_gate_ops * float(logic_gate_fae)
        trained_extended = trained_logic_fae + vote_acev2 + weighted_selection_fae
        trained_cost = {
            "clause_and_ops_trained_model": trained_and,
            "patch_or_ops_trained_model": trained_or,
            "boolean_gate_ops_trained_model": trained_gate_ops,
            "boolean_logic_fae_trained_model": trained_logic_fae,
            "extended_acev2_tm_trained_model": trained_extended,
            "extended_acev2_tm_trained_model_million": trained_extended / 1e6,
            "reduction_from_upper_bound": extended_upper - trained_extended,
            "reduction_percent": (
                100.0 * (extended_upper - trained_extended) / extended_upper
                if extended_upper else 0.0
            ),
        }

    preferred_cost = (
        trained_cost["extended_acev2_tm_trained_model"]
        if trained_cost else extended_upper
    )
    preferred_basis = (
        "trained Clause include actions with all configured patches"
        if trained_cost else "configured maximum included literals (upper bound)"
    )
    threshold_comparisons = height * width * int(boolean_channels)
    argmax_comparisons = max(int(class_count) - 1, 0)

    return {
        "acev2": None,
        "acev2_status": (
            "standard ACEv2 total is not defined for TM Boolean logic; "
            "tm_extended_acev2 is an explicit full-adder-equivalent extension"
        ),
        "method": "Extended ACEv2-TM (full-adder-equivalent)",
        "tm_extended_acev2": preferred_cost,
        "tm_extended_acev2_million": preferred_cost / 1e6,
        "tm_extended_acev2_basis": preferred_basis,
        "logic_gate_fae": float(logic_gate_fae),
        "patches_per_image": patches,
        "features_per_patch": features_per_patch,
        "available_literals_per_clause": available_literals,
        "included_literals_upper_bound": included_literals,
        "clauses_total": clause_count,
        "input_negations_per_image": input_negations,
        "clause_and_ops_upper_bound": clause_and,
        "patch_or_ops_upper_bound": patch_or,
        "vote_additions_per_image": vote_additions,
        "vote_acev2_subtotal": vote_acev2,
        "weighted_selection_muxes": weighted_selection_muxes,
        "weighted_selection_fae": weighted_selection_fae,
        "logic_operations_upper_bound": boolean_gate_ops,
        "boolean_logic_fae_upper_bound": boolean_logic_fae,
        "extended_acev2_tm_upper_bound": extended_upper,
        "extended_acev2_tm_upper_bound_million": extended_upper / 1e6,
        "threshold_comparisons_reported_separately": threshold_comparisons,
        "argmax_comparisons_reported_separately": argmax_comparisons,
        "weighted_clauses": bool(weighted_clauses),
        "accumulator_bits": int(accumulator_bits),
        "weight_bits": int(weight_bits),
        "memory_access_included": False,
        **(trained_cost or {}),
    }


def tsetlin_logic_cost(
    image_shape: tuple[int, int],
    boolean_channels: int,
    patch_dim: tuple[int, int],
    class_count: int,
    clauses_per_class: int,
    max_included_literals: int | None,
    weighted_clauses: bool = True,
) -> dict[str, Any]:
    """Backward-compatible upper-bound wrapper for older callers."""
    height, width = map(int, image_shape)
    patch_height, patch_width = map(int, patch_dim)
    available_literals = 2 * patch_height * patch_width * int(boolean_channels)
    return calculate_tsetlin_extended_acev2(
        image_shape=(height, width),
        boolean_channels=boolean_channels,
        patch_dim=(patch_height, patch_width),
        class_count=class_count,
        clauses_per_class=clauses_per_class,
        max_included_literals=(
            int(max_included_literals) if max_included_literals else available_literals
        ),
        weighted_clauses=weighted_clauses,
    )


def estimate_energy(cost: dict, coefficients: dict) -> dict:
    ace_unit = coefficients.get("ace_unit_pj")
    if cost.get("algorithmic_acev2") and ace_unit is not None:
        value = cost["algorithmic_acev2"]["acev2"] * float(ace_unit)
        return {"estimated_energy_pj": value, "energy_status": "estimated from configured ACE-unit calibration; not measured"}
    logic_unit, vote_unit = coefficients.get("logic_op_pj"), coefficients.get("vote_add_pj")
    if cost.get("logic_operations_upper_bound") is not None and logic_unit is not None and vote_unit is not None:
        value = cost["logic_operations_upper_bound"] * float(logic_unit) + cost["vote_additions_per_image"] * float(vote_unit)
        return {"estimated_energy_pj": value, "energy_status": "estimated upper bound from configured logic coefficients; not measured"}
    return {"estimated_energy_pj": None, "energy_status": "unavailable: no hardware-calibrated coefficients configured"}
