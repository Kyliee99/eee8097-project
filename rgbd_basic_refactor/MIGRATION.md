# Migration from `rgbd_basic_refactor`

| Previous file or folder | New location |
|---|---|
| `rgbd_dataset.py` | `src/data/dataset.py` and `src/data/split.py` |
| preprocessing in `common.py` | `src/data/preprocessing.py` |
| DataLoader creation in `common.py` | `src/data/loaders.py` |
| metrics in `common.py` | `src/evaluation/metrics.py` |
| `mlp.py`, `bnn.py` | `src/models/` plus `scripts/train_*.py` |
| `torch_pipeline.py` | `src/training/torch_trainer.py` |
| `TselinMachine1.py` | `src/models/tsetlin.py`, `src/training/tsetlin_trainer.py`, `scripts/train_tsetlin.py` |
| `compare_acev2.py` | `src/evaluation/*.py` plus `scripts/evaluate_models.py` |
| final `.pth/.pkl` files | `artifacts/models/` |
| final JSON/CSV/XLSX | `artifacts/results/` |
| old compare/TM variants | Git history or `archive/experiments/` when deliberately restored |

The default now follows the original per-image random split requested for this project:

1. Keep `data.split_unit: image`.
2. Delete the old `artifacts/split_manifest.json`, or run the first training command with `--regenerate-split`.
3. Train MLP, BNN and Tsetlin against the same newly generated manifest.

The splitter collects all RGB images recursively beneath each class directory and shuffles the individual files. Subdirectory or instance names do not determine whether an image enters train, validation or test. `split_unit: instance` remains an optional grouped-split experiment only.

The previous TM checkpoint may not contain `max_included_literals` or `number_of_clauses`. The runner falls back to `configs/default.yaml` and records the resulting cost assumptions.
