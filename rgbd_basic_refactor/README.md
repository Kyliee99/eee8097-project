# EEE8097 Model Efficiency Evaluation

This refactor provides one reproducible pipeline for training and comparing an MLP, a BNN, a Boolean-tensor XNOR–Popcount BNN, and a grayscale Tsetlin Machine on the Washington RGB-D Object Dataset.

The package fixes the main evaluation problems in the previous implementation:

- **Macro F1 is explicit.** `macro_f1` and `weighted_f1` are exported separately; the ambiguous `f1` field was removed.
- **Samples are randomly split by image.** Every RGB sample beneath each class is collected first, then shuffled and assigned to train/validation/test using the configured ratios. Instance-directory names do not decide the partition. The exact assignments are saved in `artifacts/split_manifest.json`.
- **Timing is repeated and scoped.** The evaluator uses warm-up runs and repeated compute-only measurements. Disk I/O and image preprocessing are excluded and this scope is written into every result.
- **ACEv2 meanings are separated.** Neural models report an algorithmic bit-width profile and a profile of the FP32 tensors executed by the current Python implementation.
- **XNOR is not misreported as packed hardware.** Ideal packed-word operation counts are reported separately from Python timing and ACEv2.
- **Tsetlin cost is not treated as zero or as standard ACEv2.** TM clause/patch/vote work is a labelled worst-case logic-operation upper bound. With a 32×32 image and 6×6 patch, the patch count is 27×27 = 729 rather than 1.
- **Energy is honest.** ACEv2 is an arithmetic proxy, not joules. Energy remains unavailable until documented hardware coefficients are configured.

The ACEv2 formulas follow the primary [PikeLPN/ACEv2 paper](https://arxiv.org/abs/2404.00103). TM is kept separate because its inference is based on propositional clauses and voting, as described in the [original Tsetlin Machine paper](https://arxiv.org/abs/1804.01508).

## Directory structure

```text
eee8097-project-refactored/
├── configs/default.yaml
├── src/
│   ├── data/          # split, manifest, dataset, preprocessing, loaders
│   ├── models/        # MLP, BNN/XNOR, Tsetlin helpers
│   ├── training/      # PyTorch and Tsetlin trainers
│   └── evaluation/    # metrics, timing, ACEv2/TM cost, export, runner
├── scripts/           # train_mlp, train_bnn, train_tsetlin, evaluate_models
├── artifacts/
│   ├── models/
│   └── results/
├── tests/
└── archive/experiments/
```

## Setup

Use Python 3.11 or later:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Put the dataset at `data/rgbd-dataset_eval`, or change `project.data_root` in `configs/default.yaml`.

Copy the final trusted checkpoints into `artifacts/models/`:

```text
rgbd_mlp_2.pth
rgbd_bnn_2.pth
best_tsetlin_rgbd.pkl
```

The model attribute names remain compatible with the existing MLP and BNN state dictionaries.

## Train

```powershell
python scripts/train_mlp.py
python scripts/train_bnn.py
python scripts/train_tsetlin.py
```

The first command creates the shared split manifest. Later commands reuse it. The default is `split_unit: image`, matching the original per-image random split. Images are shuffled separately within each class, so every class is represented in train, validation and test whenever enough samples are available.

After changing from an existing instance-level manifest, regenerate the assignments once:

```powershell
python scripts/train_mlp.py --regenerate-split
```

Then train BNN and Tsetlin without `--regenerate-split`; they reuse exactly the same manifest. The optional `split_unit: instance` mode remains available for a separate grouped-split experiment, but it is no longer the default.

## Evaluate

```powershell
python scripts/evaluate_models.py
```

Or select models:

```powershell
python scripts/evaluate_models.py --models mlp bnn bnn_xnor tsetlin
```

Outputs are written to `artifacts/results/`:

- `comparison_results.json`: complete metrics, confusion matrices, timing and cost assumptions;
- `comparison_summary.csv`: concise thesis table;
- `comparison_results.xlsx`: formatted summary for inspection.

## Meaning of the cost fields

| Field | Meaning |
|---|---|
| `algorithmic_acev2` | ACEv2 under the model's intended activation/weight bit widths; BatchNorm is treated as fused for deployment |
| `python_tensor_acev2` | Arithmetic representation used by the current PyTorch path, including FP32 BatchNorm |
| `packed_xnor_word_operations` | Ideal 64-bit word XNOR/popcount groups for binary hidden layers; not ACEv2 and not Python operations |
| `logic_operations_upper_bound` | Non-standard TM worst-case clause/patch logic estimate |
| `estimated_energy_pj` | `null` unless hardware-calibrated coefficients are explicitly supplied |

Do not rank Tsetlin directly against neural ACEv2 using `logic_operations_upper_bound`; report both columns with their definitions.

## Tests

The tests use Python's standard `unittest` framework:

```powershell
python -m unittest discover -s tests -v
```

They verify Macro-vs-Weighted F1 naming, ACEv2 formulas, the 729-patch TM calculation, deterministic image-level splitting, optional group-disjoint splitting and XNOR/dot-product equivalence.
