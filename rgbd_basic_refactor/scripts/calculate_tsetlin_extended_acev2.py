#!/usr/bin/env python3
"""Calculate an ACEv2-inspired cost for the configured Tsetlin Machine.

Standard ACEv2 does not define costs for AND, OR, NOT, comparison, or
argmax operations.  This script therefore reports both:

1. the standard ACEv2-compatible integer-voting subtotal; and
2. ``extended_acev2_tm``, where one-bit Boolean gates are converted to
   full-adder equivalents (FAE).

The default conversion follows ACEv2's internal approximation that one
full adder is roughly five 2:1 multiplexers.  A one-bit AND, OR, NOT, or
MUX is therefore assigned 1/5 = 0.2 FAE.  This is an explicit extension,
not a standard ACEv2 result.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
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


def _load_trusted_checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Load metadata from a checkpoint produced by this project.

    Pickle can execute code during loading.  Never use this helper with an
    untrusted or externally downloaded checkpoint.
    """
    if not path.is_file():
        return {}
    try:
        with path.open("rb") as stream:
            checkpoint = pickle.load(stream)
    except (ImportError, ModuleNotFoundError) as error:
        print(
            "Warning: checkpoint metadata could not be loaded because its "
            f"runtime dependency is unavailable ({error}). Using config values."
        )
        return {}
    if isinstance(checkpoint, dict):
        return checkpoint
    return {}


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
) -> dict[str, Any]:
    """Return standard subtotals and an upper-bound extended ACEv2-TM cost."""
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

    patches = (height - patch_height + 1) * (width - patch_width + 1)
    features_per_patch = patch_height * patch_width * int(boolean_channels)
    available_literals = 2 * features_per_patch
    included_literals = min(int(max_included_literals), available_literals)
    clauses_total = int(class_count) * int(clauses_per_class)

    # Complements are generated once for the Boolean image and shared by patches.
    input_not_ops = height * width * int(boolean_channels)
    clause_and_ops = (
        clauses_total
        * patches
        * max(included_literals - 1, 0)
    )
    patch_or_ops = clauses_total * max(patches - 1, 0)
    boolean_gate_ops = input_not_ops + clause_and_ops + patch_or_ops
    boolean_logic_fae = boolean_gate_ops * float(logic_gate_fae)

    # Clause scores are integer accumulations, so this part is ACEv2-compatible.
    vote_additions = int(class_count) * max(int(clauses_per_class) - 1, 0)
    vote_acev2 = vote_additions * int(accumulator_bits)

    # A weighted clause selects either zero or its signed weight. Model the
    # selection as one 2:1 MUX per weight bit; unit-vote TMs need no such term.
    weighted_selection_muxes = clauses_total * int(weight_bits) if weighted_clauses else 0
    weighted_selection_fae = weighted_selection_muxes * float(logic_gate_fae)

    extended_total = boolean_logic_fae + vote_acev2 + weighted_selection_fae
    threshold_comparisons = height * width * int(boolean_channels)
    argmax_comparisons = max(int(class_count) - 1, 0)

    return {
        "method": "Extended ACEv2-TM (full-adder-equivalent upper bound)",
        "standard_total_acev2": None,
        "standard_status": (
            "N/A: standard ACEv2 has no AND/OR/NOT cost; "
            "vote_acev2_subtotal is the standard-compatible part"
        ),
        "assumptions": {
            "logic_gate_fae": float(logic_gate_fae),
            "logic_gate_basis": "1 full adder ~= 5 two-to-one MUX; one 1-bit gate ~= 0.2 FAE",
            "cost_kind": "upper bound from max_included_literals, not measured energy",
            "input_negations": "generated once per Boolean image and shared across patches",
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
            "clause_and_ops_upper_bound": clause_and_ops,
            "patch_or_ops_upper_bound": patch_or_ops,
            "boolean_gate_ops_upper_bound": boolean_gate_ops,
            "vote_additions": vote_additions,
            "weighted_selection_muxes": weighted_selection_muxes,
            "threshold_comparisons_reported_separately": threshold_comparisons,
            "argmax_comparisons_reported_separately": argmax_comparisons,
        },
        "cost_per_image": {
            "boolean_logic_fae": boolean_logic_fae,
            "vote_acev2_subtotal": vote_acev2,
            "weighted_selection_fae": weighted_selection_fae,
            "extended_acev2_tm": extended_total,
            "extended_acev2_tm_million": extended_total / 1e6,
        },
    }


def _print_summary(result: dict[str, Any]) -> None:
    model = result["model"]
    ops = result["operations_per_image"]
    cost = result["cost_per_image"]
    print("Tsetlin Machine ACEv2 calculation")
    print(f"  Method: {result['method']}")
    print(f"  Model: {model['class_count']} classes, {model['clauses_per_class']} clauses/class")
    print(f"  Patches/image: {model['patches_per_image']}")
    print(f"  Included literals upper bound: {model['included_literals_upper_bound']}")
    print(f"  Boolean gate operations upper bound: {ops['boolean_gate_ops_upper_bound']:,}")
    print(f"  Boolean logic FAE: {cost['boolean_logic_fae']:,.1f}")
    print(f"  Vote ACEv2 subtotal: {cost['vote_acev2_subtotal']:,}")
    print(f"  Weighted-selection FAE: {cost['weighted_selection_fae']:,.1f}")
    print(f"  Extended ACEv2-TM: {cost['extended_acev2_tm']:,.1f}")
    print(f"  Extended ACEv2-TM (million): {cost['extended_acev2_tm_million']:.6f}")
    print("  Standard total ACEv2: N/A")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
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
    metadata = _load_trusted_checkpoint_metadata(checkpoint_path)

    class_names = metadata.get("class_names")
    class_count = args.class_count or (len(class_names) if class_names else None)
    if class_count is None:
        raise ValueError(
            "Class count is unavailable. Provide --class-count, or use a project "
            "checkpoint containing class_names metadata."
        )

    image_size = int(config["data"]["image_size"])
    thresholds = metadata.get(
        "boolean_thresholds",
        config["preprocessing"]["tsetlin_thresholds"],
    )
    boolean_channels = int(metadata.get("boolean_channels", len(thresholds)))
    patch_dim = tuple(metadata.get("patch_dim", tm_cfg["patch_dim"]))
    clauses = int(
        metadata.get(
            "number_of_clauses",
            metadata.get("clauses", tm_cfg["clauses"]),
        )
    )
    max_literals = int(
        metadata.get("max_included_literals", tm_cfg["max_included_literals"])
    )
    weighted = bool(
        metadata.get("weighted_clauses", tm_cfg.get("weighted_clauses", True))
    )
    accumulator_bits = int(metadata.get("vote_accumulator_bits", args.accumulator_bits))
    weight_bits = int(metadata.get("clause_weight_bits", args.weight_bits))

    result = calculate_extended_acev2_tm(
        image_shape=(image_size, image_size),
        boolean_channels=boolean_channels,
        patch_dim=(int(patch_dim[0]), int(patch_dim[1])),
        class_count=class_count,
        clauses_per_class=clauses,
        max_included_literals=max_literals,
        weighted_clauses=weighted,
        accumulator_bits=accumulator_bits,
        weight_bits=weight_bits,
        logic_gate_fae=args.logic_gate_fae,
    )
    result["sources"] = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_metadata_loaded": bool(metadata),
        "class_count_source": "--class-count" if args.class_count else "checkpoint class_names",
    }

    _print_summary(result)
    output_path = args.output or project_root / "artifacts" / "results" / "tsetlin_extended_acev2.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  JSON output: {output_path}")


if __name__ == "__main__":
    main()
