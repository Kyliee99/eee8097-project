#!/usr/bin/env python3
"""Calculate upper-bound and trained-model ACEv2-inspired costs for a TM.

Standard ACEv2 does not define costs for AND, OR, NOT, comparison, or
argmax operations. This script therefore reports:

1. the standard ACEv2-compatible integer-voting subtotal;
2. an upper-bound Boolean full-adder-equivalent (FAE) estimate; and
3. when the trusted checkpoint can be loaded, a trained-model estimate
   based on every Clause's real include actions.

The default Boolean conversion follows ACEv2's approximation that one
full adder is roughly five 2:1 multiplexers. A one-bit AND, OR, NOT, or
MUX is therefore assigned 1/5 = 0.2 FAE. This is an explicit extension,
not a standard ACEv2 result.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from statistics import fmean, median
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return data


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_trusted_checkpoint(path: Path) -> dict[str, Any]:
    """Load a checkpoint produced by this project.

    Pickle can execute code during loading. Never use this helper with an
    untrusted or externally downloaded checkpoint.
    """
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as stream:
            checkpoint = pickle.load(stream)
    except (ImportError, ModuleNotFoundError) as error:
        print(
            "Warning: checkpoint could not be loaded because its runtime "
            f"dependency is unavailable ({error}). Using config upper bounds."
        )
        return {}
    if isinstance(checkpoint, dict):
        return checkpoint
    return {"model": checkpoint}


def _summarize_lengths(lengths: list[int]) -> dict[str, Any]:
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


def _extract_trained_clause_profile(model: object | None) -> dict[str, Any] | None:
    """Read real include actions from every class-specific TMU Clause Bank."""
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
        clauses = int(getattr(bank, "number_of_clauses"))
        clause_counts.add(clauses)
        incremental_flags.add(
            bool(getattr(bank, "incremental", getattr(model, "incremental", False)))
        )

        include_count = getattr(bank, "number_of_include_actions", None)
        if callable(include_count):
            lengths = [int(include_count(clause)) for clause in range(clauses)]
            extraction_method = "ClauseBank.number_of_include_actions"
        else:
            get_literals = getattr(bank, "get_literals", None)
            if not callable(get_literals):
                raise RuntimeError(
                    f"Clause Bank for class {class_id} exposes neither "
                    "number_of_include_actions() nor get_literals()."
                )
            literal_matrix = get_literals()
            lengths = [
                int(sum(int(value) for value in row))
                for row in literal_matrix
            ]
            extraction_method = "ClauseBank.get_literals"

        all_lengths.extend(lengths)
        per_class[str(class_id)] = {
            **_summarize_lengths(lengths),
            "extraction_method": extraction_method,
        }

    if len(patch_counts) != 1:
        raise RuntimeError(f"Class-specific Clause Banks disagree on patch count: {patch_counts}")
    if len(clause_counts) != 1:
        raise RuntimeError(f"Class-specific Clause Banks disagree on Clause count: {clause_counts}")

    return {
        **_summarize_lengths(all_lengths),
        "class_count": len(class_ids),
        "clauses_per_class": next(iter(clause_counts)),
        "model_number_of_patches": next(iter(patch_counts)),
        "bank_types": sorted(bank_types),
        "incremental_flags": sorted(incremental_flags),
        "per_class": per_class,
        "_flat_lengths": all_lengths,
    }


def _patch_traversal_report(
    model: object | None,
    clause_profile: dict[str, Any] | None,
) -> dict[str, Any]:
    """Describe the patch traversal used by the loaded TMU backend.

    TMU's dense CPU incremental path uses a patch-outer loop and visits every
    patch. Its non-incremental dense CPU predictor can return as soon as a
    satisfying patch is found, so dynamic counts then require C-level counters.
    """
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
            "note": (
                "The incremental CPU loop scans all patches. This is an exact "
                "control-flow count, not a timing-based inference."
            ),
        }

    if dense_cpu and incremental_flags == [False]:
        return {
            "status": (
                "dynamic: dense CPU non-incremental prediction can stop after "
                "the first satisfying patch; add counters inside TMU C code for an exact mean"
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


def calculate_extended_acev2_tm(
    *,
    image_shape: tuple[int, int],
    boolean_channels: int,
    patch_dim: tuple[int, int],
    class_count: int,
    clauses_per_class: int,
    max_included_literals: int,
    weighted_clauses: bool,
    accumulator_bits: int = 32,
    weight_bits: int = 32,
    logic_gate_fae: float = 0.2,
    trained_clause_lengths: list[int] | None = None,
    model_patch_count: int | None = None,
) -> dict[str, Any]:
    """Return an upper bound and, when available, a trained-model estimate."""
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
    if patch_height <= 0 or patch_width <= 0:
        raise ValueError("patch dimensions must be positive")
    if patch_height > height or patch_width > width:
        raise ValueError("patch dimensions must fit inside the image")
    if logic_gate_fae < 0:
        raise ValueError("logic_gate_fae must be non-negative")

    configured_patches = (
        (height - patch_height + 1)
        * (width - patch_width + 1)
    )
    if model_patch_count is not None and int(model_patch_count) != configured_patches:
        raise ValueError(
            "Checkpoint Clause Bank patch count does not match image/patch configuration: "
            f"{model_patch_count} != {configured_patches}"
        )
    patches = configured_patches
    features_per_patch = patch_height * patch_width * int(boolean_channels)
    available_literals = 2 * features_per_patch
    included_literals = min(int(max_included_literals), available_literals)
    clauses_total = int(class_count) * int(clauses_per_class)

    input_not_ops = height * width * int(boolean_channels)
    clause_and_ops_upper = (
        clauses_total
        * patches
        * max(included_literals - 1, 0)
    )
    patch_or_ops_upper = clauses_total * max(patches - 1, 0)
    boolean_gate_ops_upper = input_not_ops + clause_and_ops_upper + patch_or_ops_upper
    boolean_logic_fae_upper = boolean_gate_ops_upper * float(logic_gate_fae)

    vote_additions = int(class_count) * max(int(clauses_per_class) - 1, 0)
    vote_acev2 = vote_additions * int(accumulator_bits)
    weighted_selection_muxes = (
        clauses_total * int(weight_bits)
        if weighted_clauses
        else 0
    )
    weighted_selection_fae = weighted_selection_muxes * float(logic_gate_fae)
    extended_upper = (
        boolean_logic_fae_upper
        + vote_acev2
        + weighted_selection_fae
    )

    trained_cost: dict[str, Any] | None = None
    if trained_clause_lengths is not None:
        lengths = list(map(int, trained_clause_lengths))
        if len(lengths) != clauses_total:
            raise ValueError(
                "Actual Clause-length count does not match class_count * clauses_per_class: "
                f"{len(lengths)} != {clauses_total}"
            )
        if min(lengths, default=0) < 0 or max(lengths, default=0) > available_literals:
            raise ValueError("Actual Clause lengths fall outside the available-literal range")

        nonempty_clauses = sum(length > 0 for length in lengths)
        clause_and_ops_trained = patches * sum(
            max(length - 1, 0)
            for length in lengths
        )
        # TMU treats an empty Clause as false, so it needs no patch OR reduction.
        patch_or_ops_trained = nonempty_clauses * max(patches - 1, 0)
        boolean_gate_ops_trained = (
            input_not_ops
            + clause_and_ops_trained
            + patch_or_ops_trained
        )
        boolean_logic_fae_trained = (
            boolean_gate_ops_trained
            * float(logic_gate_fae)
        )
        extended_trained = (
            boolean_logic_fae_trained
            + vote_acev2
            + weighted_selection_fae
        )
        trained_cost = {
            "clause_and_ops_trained_model": clause_and_ops_trained,
            "patch_or_ops_trained_model": patch_or_ops_trained,
            "boolean_gate_ops_trained_model": boolean_gate_ops_trained,
            "boolean_logic_fae_trained_model": boolean_logic_fae_trained,
            "extended_acev2_tm_trained_model": extended_trained,
            "extended_acev2_tm_trained_model_million": extended_trained / 1e6,
            "reduction_from_upper_bound": extended_upper - extended_trained,
            "reduction_percent": (
                100.0 * (extended_upper - extended_trained) / extended_upper
                if extended_upper
                else 0.0
            ),
        }

    threshold_comparisons = height * width * int(boolean_channels)
    argmax_comparisons = max(int(class_count) - 1, 0)

    return {
        "method": "Extended ACEv2-TM (full-adder-equivalent)",
        "standard_total_acev2": None,
        "standard_status": (
            "N/A: standard ACEv2 has no AND/OR/NOT cost; "
            "vote_acev2_subtotal is the standard-compatible part"
        ),
        "assumptions": {
            "logic_gate_fae": float(logic_gate_fae),
            "logic_gate_basis": (
                "1 full adder ~= 5 two-to-one MUX; "
                "one 1-bit gate ~= 0.2 FAE"
            ),
            "cost_kind": (
                "gate-level proxy; trained-model estimate uses real include actions "
                "but is not measured energy or a CPU instruction count"
            ),
            "input_negations": (
                "generated once per Boolean image and shared across patches"
            ),
            "accumulator_bits": int(accumulator_bits),
            "weight_bits": int(weight_bits),
            "memory_access_included": False,
            "threshold_and_argmax_in_extended_total": False,
        },
        "model": {
            "image_shape": [height, width],
            "boolean_channels": int(boolean_channels),
            "patch_dim": [patch_height, patch_width],
            "patches_per_image": patches,
            "features_per_patch": features_per_patch,
            "available_literals_per_clause": available_literals,
            "included_literals_upper_bound": included_literals,
            "class_count": int(class_count),
            "clauses_per_class": int(clauses_per_class),
            "clauses_total": clauses_total,
            "weighted_clauses": bool(weighted_clauses),
        },
        "operations_per_image": {
            "input_not_ops": input_not_ops,
            "clause_and_ops_upper_bound": clause_and_ops_upper,
            "patch_or_ops_upper_bound": patch_or_ops_upper,
            "boolean_gate_ops_upper_bound": boolean_gate_ops_upper,
            "vote_additions": vote_additions,
            "weighted_selection_muxes": weighted_selection_muxes,
            "threshold_comparisons_reported_separately": threshold_comparisons,
            "argmax_comparisons_reported_separately": argmax_comparisons,
            **(trained_cost or {}),
        },
        "cost_per_image": {
            "boolean_logic_fae_upper_bound": boolean_logic_fae_upper,
            # Backward-compatible field retained from the first script version.
            "boolean_logic_fae": boolean_logic_fae_upper,
            "vote_acev2_subtotal": vote_acev2,
            "weighted_selection_fae": weighted_selection_fae,
            "extended_acev2_tm_upper_bound": extended_upper,
            # Backward-compatible field retained from the first script version.
            "extended_acev2_tm": extended_upper,
            "extended_acev2_tm_million": extended_upper / 1e6,
            **({
                key: value
                for key, value in trained_cost.items()
                if "fae" in key or "extended_acev2" in key or key.startswith("reduction")
            } if trained_cost else {}),
        },
    }


def _print_summary(result: dict[str, Any]) -> None:
    model = result["model"]
    ops = result["operations_per_image"]
    cost = result["cost_per_image"]
    print("Tsetlin Machine ACEv2 calculation")
    print(f"  Method: {result['method']}")
    print(
        f"  Model: {model['class_count']} classes, "
        f"{model['clauses_per_class']} clauses/class"
    )
    print(f"  Patches/image: {model['patches_per_image']}")
    print(
        "  Included literals upper bound: "
        f"{model['included_literals_upper_bound']}"
    )
    print(
        "  Boolean gate operations upper bound: "
        f"{ops['boolean_gate_ops_upper_bound']:,}"
    )
    print(
        "  Extended ACEv2-TM upper bound: "
        f"{cost['extended_acev2_tm_upper_bound']:,.1f}"
    )

    profile = result.get("trained_clause_profile")
    if profile:
        print(
            "  Real Clause length min/mean/median/max: "
            f"{profile['included_literals_min']}/"
            f"{profile['included_literals_mean']:.3f}/"
            f"{profile['included_literals_median']:.3f}/"
            f"{profile['included_literals_max']}"
        )
        print(f"  Empty Clauses: {profile['empty_clause_count']:,}")
        print(
            "  Trained-model Boolean gate operations: "
            f"{ops['boolean_gate_ops_trained_model']:,}"
        )
        print(
            "  Extended ACEv2-TM trained model: "
            f"{cost['extended_acev2_tm_trained_model']:,.1f} "
            f"({cost['extended_acev2_tm_trained_model_million']:.6f} M)"
        )
        print(
            "  Reduction from upper bound: "
            f"{cost['reduction_percent']:.2f}%"
        )
    else:
        print(
            "  Trained-model result: unavailable; install TMU and load "
            "the trusted checkpoint"
        )

    traversal = result.get("patch_evaluation", {})
    print(f"  Patch traversal: {traversal.get('status', 'unavailable')}")
    if traversal.get("all_patches_visited") is True:
        print(
            "  All patches visited per class/image: "
            f"{traversal['patches_visited_per_class_per_image']}"
        )
        print(
            "  Clause-patch checks/image: "
            f"{traversal['clause_patch_checks_per_image']:,}"
        )

    print(f"  Vote ACEv2 subtotal: {cost['vote_acev2_subtotal']:,}")
    print(f"  Weighted-selection FAE: {cost['weighted_selection_fae']:,.1f}")
    print("  Standard total ACEv2: N/A")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--class-count", type=int, default=None)
    parser.add_argument("--accumulator-bits", type=int, default=32)
    parser.add_argument("--weight-bits", type=int, default=32)
    parser.add_argument("--logic-gate-fae", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = _load_yaml(config_path)
    project_root = config_path.parent.parent
    models_cfg = config["models"]
    tm_cfg = models_cfg["tsetlin"]
    checkpoint_path = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else _resolve(project_root, models_cfg["tsetlin_checkpoint"])
    )
    checkpoint = _load_trusted_checkpoint(checkpoint_path)
    model = checkpoint.get("model")
    clause_profile = _extract_trained_clause_profile(model)
    patch_traversal = _patch_traversal_report(model, clause_profile)

    class_names = checkpoint.get("class_names")
    class_count = (
        args.class_count
        or (len(class_names) if class_names else None)
        or (clause_profile["class_count"] if clause_profile else None)
    )
    if class_count is None:
        raise ValueError(
            "Class count is unavailable. Provide --class-count, or use a project "
            "checkpoint containing class_names or a loaded TMU model."
        )

    image_size = int(config["data"]["image_size"])
    thresholds = checkpoint.get(
        "boolean_thresholds",
        config["preprocessing"]["tsetlin_thresholds"],
    )
    boolean_channels = int(
        checkpoint.get("boolean_channels", len(thresholds))
    )
    patch_dim = tuple(
        checkpoint.get("patch_dim", tm_cfg["patch_dim"])
    )
    clauses = int(
        clause_profile["clauses_per_class"]
        if clause_profile
        else checkpoint.get(
            "number_of_clauses",
            checkpoint.get("clauses", tm_cfg["clauses"]),
        )
    )
    max_literals = int(
        checkpoint.get(
            "max_included_literals",
            tm_cfg["max_included_literals"],
        )
    )
    weighted = bool(
        checkpoint.get(
            "weighted_clauses",
            tm_cfg.get("weighted_clauses", True),
        )
    )
    accumulator_bits = int(
        checkpoint.get(
            "vote_accumulator_bits",
            args.accumulator_bits,
        )
    )
    weight_bits = int(
        checkpoint.get(
            "clause_weight_bits",
            args.weight_bits,
        )
    )

    flat_lengths = (
        clause_profile.pop("_flat_lengths")
        if clause_profile
        else None
    )
    result = calculate_extended_acev2_tm(
        image_shape=(image_size, image_size),
        boolean_channels=boolean_channels,
        patch_dim=(int(patch_dim[0]), int(patch_dim[1])),
        class_count=int(class_count),
        clauses_per_class=clauses,
        max_included_literals=max_literals,
        weighted_clauses=weighted,
        accumulator_bits=accumulator_bits,
        weight_bits=weight_bits,
        logic_gate_fae=args.logic_gate_fae,
        trained_clause_lengths=flat_lengths,
        model_patch_count=(
            clause_profile["model_number_of_patches"]
            if clause_profile
            else None
        ),
    )
    result["trained_clause_profile"] = clause_profile
    result["patch_evaluation"] = patch_traversal
    result["sources"] = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_loaded": bool(checkpoint),
        "model_loaded": model is not None,
        "class_count_source": (
            "--class-count"
            if args.class_count
            else "checkpoint class_names"
            if class_names
            else "loaded model Clause Banks"
        ),
        "clause_lengths_source": (
            "trained Tsetlin Automata include actions"
            if clause_profile
            else "unavailable; upper bound only"
        ),
        "patch_count_source": (
            "loaded TMU Clause Bank"
            if clause_profile
            else "image and patch dimensions"
        ),
    }

    _print_summary(result)
    output_path = (
        args.output
        or project_root
        / "artifacts"
        / "results"
        / "tsetlin_extended_acev2.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print(f"  JSON output: {output_path}")


if __name__ == "__main__":
    main()
