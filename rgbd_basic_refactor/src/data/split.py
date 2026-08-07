"""Deterministic, class-stratified train/validation/test image splitting."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

SPLITS = ("train", "val", "test")


def is_rgb_crop(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.endswith("_crop.png")
        and not name.endswith("_depthcrop.png")
        and not name.endswith("_maskcrop.png")
    )


def discover_images(dataset_root: Path) -> dict[str, list[Path]]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")
    images: dict[str, list[Path]] = {}
    for class_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        class_images = sorted(
            path for path in class_dir.rglob("*.png") if path.is_file() and is_rgb_crop(path)
        )
        if not class_images:
            raise RuntimeError(f"No RGB crop images found for class: {class_dir.name}")
        images[class_dir.name] = class_images
    if not images:
        raise RuntimeError(f"No class directories found in: {dataset_root}")
    return images


def _split_counts(group_count: int, ratios: dict[str, float]) -> dict[str, int]:
    if group_count < 3:
        raise ValueError("Each class needs at least three groups for train/val/test splitting")
    if set(ratios) != set(SPLITS) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("split_ratios must contain train/val/test and sum to 1.0")
    raw = {name: group_count * float(ratios[name]) for name in SPLITS}
    counts = {name: max(1, int(raw[name])) for name in SPLITS}
    while sum(counts.values()) > group_count:
        candidates = [name for name in SPLITS if counts[name] > 1]
        if not candidates:
            raise ValueError("Unable to allocate at least one group to every split")
        counts[max(candidates, key=lambda name: (counts[name] - raw[name], counts[name]))] -= 1
    while sum(counts.values()) < group_count:
        counts[max(SPLITS, key=lambda name: (raw[name] - counts[name], ratios[name]))] += 1
    return counts


def _group_key(path: Path, class_dir: Path, split_unit: str) -> str:
    if split_unit == "image":
        return path.relative_to(class_dir).as_posix()
    if split_unit != "instance":
        raise ValueError("split_unit must be 'instance' or 'image'")
    relative_parent = path.parent.relative_to(class_dir)
    # Images directly under a class cannot be grouped by an instance directory.
    return relative_parent.as_posix() if relative_parent.parts else path.stem


def create_split_manifest(
    dataset_root: str | Path,
    ratios: dict[str, float],
    seed: int = 42,
    split_unit: str = "image",
) -> dict:
    root = Path(dataset_root).expanduser().resolve()
    discovered = discover_images(root)
    assignments: dict[str, str] = {}
    groups_by_split: dict[str, list[str]] = {name: [] for name in SPLITS}

    for class_name, paths in discovered.items():
        class_dir = root / class_name
        grouped: dict[str, list[Path]] = defaultdict(list)
        for path in paths:
            grouped[_group_key(path, class_dir, split_unit)].append(path)
        group_names = sorted(grouped)
        rng = random.Random(f"{seed}:{class_name}:{split_unit}")
        rng.shuffle(group_names)
        counts = _split_counts(len(group_names), ratios)
        cursor = 0
        for split in SPLITS:
            selected = group_names[cursor : cursor + counts[split]]
            cursor += counts[split]
            for group_name in selected:
                groups_by_split[split].append(f"{class_name}/{group_name}")
                for path in grouped[group_name]:
                    assignments[path.relative_to(root).as_posix()] = split

    digest_source = "\n".join(f"{path}:{assignments[path]}" for path in sorted(assignments))
    counts = Counter(assignments.values())
    return {
        "version": 2,
        "dataset_root": str(root),
        "seed": int(seed),
        "split_unit": split_unit,
        "ratios": {name: float(ratios[name]) for name in SPLITS},
        "class_names": sorted(discovered),
        "assignments": assignments,
        "groups": groups_by_split,
        "sample_counts": {name: counts[name] for name in SPLITS},
        "sha256": hashlib.sha256(digest_source.encode("utf-8")).hexdigest(),
    }


def validate_manifest(manifest: dict, dataset_root: Path) -> None:
    assignments = manifest.get("assignments", {})
    if not assignments:
        raise ValueError("Split manifest contains no assignments")
    missing = [relative for relative in assignments if not (dataset_root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"Manifest references {len(missing)} missing images; regenerate it")
    groups = manifest.get("groups", {})
    sets = {name: set(groups.get(name, [])) for name in SPLITS}
    if (sets["train"] & sets["val"]) or (sets["train"] & sets["test"]) or (sets["val"] & sets["test"]):
        raise ValueError("Split manifest has overlapping groups")


def build_or_load_manifest(
    dataset_root: str | Path,
    manifest_path: str | Path,
    ratios: dict[str, float],
    seed: int = 42,
    split_unit: str = "image",
    regenerate: bool = False,
) -> dict:
    root = Path(dataset_root).expanduser().resolve()
    output = Path(manifest_path).expanduser().resolve()
    if output.is_file() and not regenerate:
        with output.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        expected = (int(seed), split_unit, {name: float(ratios[name]) for name in SPLITS})
        actual = (manifest.get("seed"), manifest.get("split_unit"), manifest.get("ratios"))
        if actual != expected:
            raise ValueError("Existing split manifest does not match the configuration; use --regenerate-split")
        validate_manifest(manifest, root)
        return manifest
    manifest = create_split_manifest(root, ratios, seed=seed, split_unit=split_unit)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False)
    return manifest


def paths_for_split(manifest: dict, dataset_root: Path, split: str) -> Iterable[tuple[Path, int]]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    class_to_index = {name: index for index, name in enumerate(manifest["class_names"])}
    for relative, assigned_split in sorted(manifest["assignments"].items()):
        if assigned_split == split:
            class_name = Path(relative).parts[0]
            yield dataset_root / relative, class_to_index[class_name]
