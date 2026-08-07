import tempfile
import unittest
from pathlib import Path

from src.data.split import create_split_manifest


class SplitTests(unittest.TestCase):
    def test_images_are_randomly_split_independently_of_instance_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images_per_class = 20
            for class_name in ("apple", "banana"):
                for instance in range(2):
                    directory = root / class_name / f"{class_name}_{instance}"
                    directory.mkdir(parents=True)
                    for view in range(images_per_class // 2):
                        (directory / f"{class_name}_{instance}_{view}_crop.png").write_bytes(b"test")

            ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
            manifest = create_split_manifest(root, ratios, seed=42, split_unit="image")

            self.assertEqual(manifest["sample_counts"], {"train": 28, "val": 6, "test": 6})
            self.assertEqual(len(manifest["assignments"]), 2 * images_per_class)

            # Every class is represented in every split because splitting is
            # performed on shuffled image files within each class.
            for class_name in ("apple", "banana"):
                class_assignments = {
                    split
                    for path, split in manifest["assignments"].items()
                    if path.startswith(f"{class_name}/")
                }
                self.assertEqual(class_assignments, {"train", "val", "test"})

            # An instance directory is not kept as a single split group.
            instance_splits = {
                split
                for path, split in manifest["assignments"].items()
                if path.startswith("apple/apple_0/")
            }
            self.assertGreater(len(instance_splits), 1)

            repeated = create_split_manifest(root, ratios, seed=42, split_unit="image")
            self.assertEqual(manifest["assignments"], repeated["assignments"])

    def test_instance_groups_do_not_cross_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for class_name in ("apple", "banana"):
                for instance in range(6):
                    directory = root / class_name / f"{class_name}_{instance}"
                    directory.mkdir(parents=True)
                    for view in range(2):
                        (directory / f"{class_name}_{instance}_{view}_crop.png").write_bytes(b"test")
            manifest = create_split_manifest(
                root, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42, split_unit="instance"
            )
            assignments = manifest["assignments"]
            for class_name in ("apple", "banana"):
                for instance in range(6):
                    prefix = f"{class_name}/{class_name}_{instance}/"
                    splits = {split for path, split in assignments.items() if path.startswith(prefix)}
                    self.assertEqual(len(splits), 1)
            group_sets = [set(manifest["groups"][split]) for split in ("train", "val", "test")]
            self.assertFalse(group_sets[0] & group_sets[1])
            self.assertFalse(group_sets[0] & group_sets[2])
            self.assertFalse(group_sets[1] & group_sets[2])


if __name__ == "__main__":
    unittest.main()
