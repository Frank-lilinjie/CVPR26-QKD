# Open-source release checklist

## Code and configuration

- [x] Keep the complete QKD/QuanAdapter method implementation together with the MindSpore entry point.
- [x] Add `qkd_mindspore/main.py`, `qkd_mindspore/__main__.py` and root-level `run_mindspore.py`.
- [x] Add a synthetic, two-task MindSpore smoke configuration that does not require datasets or checkpoints.
- [x] Add portable config validation and CLI overrides for device, mode, tasks, batch size and paths.
- [x] Keep the original PyTorch implementation as a clearly labelled legacy reference.
- [x] Replace host-specific absolute dataset/checkpoint paths with configurable resolution.
- [x] Add pure-NumPy circuit tests and synthetic-data tests.
- [x] Match the reference CIFAR, ImageNet-1K and 224x224 ViT augmentation/normalisation profiles.
- [x] Use host-side StandardScaler + TruncatedSVD instead of a full SVD for QKD feature preprocessing.
- [x] Add a MindSpore checkpoint plus PQK-prototype save/load/inference round-trip test.
- [x] Reject unsafe archive paths and links when extracting CIFAR downloads.
- [x] Exclude logs, cached bytecode, results, checkpoints and duplicate backups from the public source tree.
- [x] Document the MindSpore/CANN entry point and smoke procedure.

## Local checks completed in this workspace

- [x] Python bytecode compilation of `qkd_mindspore`.
- [x] CLI `--help` and `--dry-run` without importing MindSpore.
- [x] NumPy circuit/reduced-density-matrix unit tests.
- [x] Synthetic task split and loader tests.
- [x] Dataset-profile and image-transform tests.
- [x] MindSpore 2.7.2 CPU regression suite (25 tests).
- [x] Two-task synthetic smoke with adapter training, PQK distillation, classifier alignment and finite `metrics.json`.
- [x] MindSpore `.ckpt` and PQK-prototype round-trip restored into a fresh network instance.

## Required on the 910B target before publishing

- [ ] Record the exact CANN, driver, MindSpore, Python and 910B versions.
- [ ] Run the synthetic two-task smoke and preserve its `metrics.json` and log.
- [ ] Run a two-task real-data experiment, including PQK distillation and classifier alignment.
- [ ] Test loading one actual timm ViT `.npz` or MindSpore `.ckpt` and confirm key/shape handling.
- [ ] Run the full experiment suite and compare with the paper results.
- [ ] Review comments, debug output and generated artifacts for public-release suitability.

## Legal and release metadata

- [ ] Confirm the project licence with all authors and add the chosen `LICENSE` file.
- [ ] Review dataset and pretrained-weight licences before publishing links or binaries.
- [ ] Do not publish datasets, model weights, keys, logs, checkpoints, result files or machine-specific paths.
