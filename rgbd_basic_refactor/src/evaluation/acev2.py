"""ACEv2 arithmetic profiles and a separate non-standard Tsetlin logic estimate."""

from __future__ import annotations

import math
from typing import Iterable


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


def tsetlin_logic_cost(
    image_shape: tuple[int, int],
    boolean_channels: int,
    patch_dim: tuple[int, int],
    class_count: int,
    clauses_per_class: int,
    max_included_literals: int | None,
    weighted_clauses: bool = True,
) -> dict:
    height, width = map(int, image_shape)
    patch_height, patch_width = map(int, patch_dim)
    if patch_height > height or patch_width > width or min(patch_height, patch_width) <= 0:
        raise ValueError("patch_dim must fit inside image_shape")
    patches = (height - patch_height + 1) * (width - patch_width + 1)
    features_per_patch = patch_height * patch_width * int(boolean_channels)
    available_literals = 2 * features_per_patch
    included_literals = min(available_literals, int(max_included_literals)) if max_included_literals else available_literals
    clause_count = int(class_count) * int(clauses_per_class)
    # Negations can be generated once and shared. Clause AND and patch OR are worst-case bounds.
    input_negations = patches * features_per_patch
    clause_and = clause_count * patches * max(included_literals - 1, 0)
    patch_or = clause_count * max(patches - 1, 0)
    vote_additions = int(class_count) * max(int(clauses_per_class) - 1, 0)
    return {
        "acev2": None,
        "acev2_status": "not defined: ACEv2 covers arithmetic operations, not TM clause logic",
        "logic_cost_kind": "worst-case upper bound; report separately from ACEv2",
        "patches_per_image": patches,
        "features_per_patch": features_per_patch,
        "available_literals_per_clause": available_literals,
        "included_literals_upper_bound": included_literals,
        "clauses_total": clause_count,
        "input_negations_per_image": input_negations,
        "clause_and_ops_upper_bound": clause_and,
        "patch_or_ops_upper_bound": patch_or,
        "vote_additions_per_image": vote_additions,
        "logic_operations_upper_bound": input_negations + clause_and + patch_or,
        "weighted_clauses": bool(weighted_clauses),
        "weighted_vote_cost_status": (
            "not converted to ACEv2: learned clause-weight precision is not available in the checkpoint"
            if weighted_clauses
            else "unit clause votes"
        ),
    }


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
