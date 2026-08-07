# Refactor report

## Source reviewed

- Repository: `Kyliee99/eee8097-project`
- Branch: `main`
- Reviewed head: `9bdbd887b733555c76bd2f3a66c314f1487f1516`
- Main legacy evaluator: `rgbd_basic_refactor/compare_acev2.py` (2,363 lines)

## Directory decisions

Core functionality was retained and split by responsibility. IDE files, caches, `text.txt`, duplicated compare scripts, historical RGB Tsetlin checkpoints and numbered TM experiments are not copied into the clean runnable package. Nothing was deleted from the GitHub repository; historical material remains recoverable from Git history.

The trained binary checkpoints are also not duplicated in this source-only ZIP. Copy the three final trusted files to `artifacts/models/` before evaluation.

## Corrected evaluation issues

1. **Macro F1:** the previous `f1` field used `average="weighted"`. The new schema exports `macro_f1` and `weighted_f1` independently.
2. **Reproducible image split:** all RGB images beneath each class are collected and randomly allocated 70%/15%/15% at image level. A persisted manifest makes the same assignment available to every model; instance-directory names do not control the split.
3. **Timing:** one-off timings were replaced with warm-up plus repeated measurements. Timing scope is compute-only and explicitly recorded.
4. **BNN cost:** intended low-precision arithmetic is separated from the FP32 tensor path actually used by `BinaryLinear`.
5. **XNOR cost:** Boolean-tensor Python operations, algorithmic ACEv2 and ideal packed-word operations are separate fields.
6. **Tsetlin patches:** a 32×32 input with a 6×6 stride-one patch has 729 patch positions. The previous fallback of one patch was removed.
7. **Tsetlin ACEv2:** TM clause logic is no longer assigned a misleading standard ACEv2 value. A non-standard, labelled logic upper bound is exported instead.
8. **Energy:** no energy number is produced without hardware-calibrated coefficients. ACEv2 is not labelled as measured energy.
9. **Checkpoint compatibility:** MLP/BNN attribute names remain `fc1...fc4` and `bn1...bn3`, matching the existing `.pth` state dictionaries.

## Reference costs for the configured architecture

These values are formula checks, not experimental results:

| Model/profile | Cost per image |
|---|---:|
| MLP algorithmic ACEv2 | 706,521,536 |
| BNN algorithmic ACEv2 | 16,942,720 |
| BNN current FP32-tensor ACEv2 profile | 707,579,904 |
| XNOR ideal 64-bit packed word groups (binary hidden layers) | 2,586 |
| TM logic-operation upper bound | 45,565,732 |

The TM number must not be put in the same ACEv2 column as the neural models.

## Validation performed

- All 19 Python modules imported successfully in the available lightweight environment.
- Python bytecode compilation passed for `src/`, `scripts/` and `tests/`.
- Split tests cover deterministic per-image allocation, per-class representation and the optional instance-disjoint mode. The full validation command is documented in `README.md`.

An end-to-end model evaluation was not run in this workspace because the dataset, PyTorch/TMU runtime and binary checkpoints are not locally available. The evaluator fails clearly when any required checkpoint or dataset path is missing.
