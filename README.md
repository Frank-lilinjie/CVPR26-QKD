# QKD / QuanAdapter

Research code for **class-incremental learning with a quantum task identifier and adapter-based ViT backbones**, organised for a public release.

> **分支说明：昇腾 910B NPU 适配版本。** This branch adds the Huawei Ascend
> 910B + CANN + MindSpore implementation while retaining the original PyTorch
> files as a legacy reference.

## Status

The supported implementation for **Huawei Ascend 910B + CANN** is the MindSpore package in `qkd_mindspore/`. It follows the QKD/QuanAdapter method while avoiding the Python-level `torch_npu` and complex-tensor dependencies of the original implementation.

The root-level PyTorch files (`main.py`, `trainer.py`, `models/`, `backbone/`, `utils/`) are retained as a **legacy reference implementation**. They are useful for comparing behaviour and for paper reproduction, but they are not the recommended 910B entry point.

> Branch contents: the MindSpore/CANN implementation is provided in
> `qkd_mindspore/`. Installation, launch commands, experiment layouts and
> target-host notes are documented in `docs/ASCEND_910B.md`.

## Method overview

1. **ViT adapter backbone** (`qkd_mindspore/model.py`): ViT with a task-specific bottleneck adapter bank.
2. **Quantum task identifier** (`qkd_mindspore/quantum.py`): StandardScaler + TruncatedSVD feature projection, data-reuploading RY circuit, Pauli-Z readout, and task logits.
3. **PQK task weighting** (`qkd_mindspore/quantum.py`, `qkd_mindspore/trainer.py`): reduced density matrices are accumulated as task prototypes and used for task-weighted distillation/routing.
4. **Global classifier and alignment** (`qkd_mindspore/trainer.py`): every adapter shares one global classifier. Class feature statistics are sampled to align that classifier from the second task onward, while old class logits remain available for distillation.

`qkd_mindspore/quantum_math.py` is a pure-NumPy reference for the circuit and reduced-density-matrix mathematics. It can be audited and tested without MindSpore.

## Repository layout

```text
qkd_mindspore/main.py           MindSpore CLI entry point (lazy trainer import)
qkd_mindspore/__main__.py       python -m qkd_mindspore
qkd_mindspore/config.py         config normalisation and validation
qkd_mindspore/device.py         Ascend/CANN context and device checks
qkd_mindspore/data.py           CIFAR/ImageFolder/synthetic loaders
qkd_mindspore/model.py          ViT, adapters, global classifier and routing
qkd_mindspore/quantum.py        quantum circuit, PQK prototypes
qkd_mindspore/quantum_math.py   NumPy circuit reference
qkd_mindspore/trainer.py        two-stage incremental training loop
qkd_mindspore/paths.py          portable dataset/checkpoint resolution

run_mindspore.py                repository-root MindSpore launcher
scripts/run_mindspore_smoke.sh  recommended 910B smoke launcher
scripts/run_npu_smoke.sh        compatibility wrapper for the same launcher
scripts/run_torch_npu_smoke.sh  legacy PyTorch/torch_npu launcher
scripts/run_qsvd_grid.py        legacy PyTorch/CUDA/Ascend grid launcher

exps/smoke/mindspore_synthetic.json   lightweight two-task smoke config
exps/                                 paper experiment configurations
tests/                                config, transforms, checkpoint and NumPy-math tests
docs/ASCEND_910B.md                   CANN + MindSpore target-host guide
```

## Environment

Install NumPy/Pillow/scikit-learn with:

```bash
pip install -r requirements-mindspore.txt
```

MindSpore for Ascend is **not** installed by that requirements file. Install the MindSpore wheel built for the host's CANN toolkit and driver from Huawei's Ascend software source; a generic PyPI CPU wheel is not sufficient for 910B execution. Then source CANN and verify the interpreter before training:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python - <<'PY'
import mindspore as ms
print("mindspore:", ms.__version__)
print("device target:", ms.get_context("device_target"))
PY
```

The current trainer is single-process and single-device. Start with `--device npu:0`, `--device-target ascend`, `--ms-mode pynative`, and `--num-workers 0`.

## Data and checkpoints

Paths are resolved portably; no host-specific absolute paths are embedded in the source:

- Dataset root: `--data-root`, config `data_root`/`data_roots`, `QKD_DATA_ROOT`, then `./data`.
- ViT checkpoint: `--checkpoint-path`, config `checkpoint_path`, `--checkpoint-dir`, `QKD_CHECKPOINT_DIR`, then `./checkpoints`.

The timm ViT-B/16 checkpoint filenames expected by `qkd_mindspore/paths.py` are:

```text
B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz
B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz
```

The synthetic smoke configuration sets `pretrained: false` and does not need either file. A full experiment should provide the matching `.npz` or a MindSpore `.ckpt` containing the backbone weights. Datasets, checkpoints, logs, results and generated artifacts are deliberately excluded from the public source tree; see `.gitignore`.

With `save_checkpoint: true`, each task writes `task_XX.ckpt` (network
parameters), `task_XX_prototypes.npz` (PQK task prototypes) and
`task_XX_stats.json` (classifier-alignment statistics). Loading these artifacts
for inference is supported and tested; resuming an interrupted optimizer state
is not implemented.

Data augmentation follows the reference dataset classes rather than one
universal transform: CIFAR-10/100 use native 32x32 random crop, horizontal flip,
brightness jitter and dataset normalisation; ImageNet-1K uses 224x224
RandomResizedCrop, horizontal flip, brightness jitter and ImageNet normalisation;
the remaining ViT datasets use the reference 224x224 crop/flip/resize pipeline.
The paired `cifar224` variant keeps its 224x224 ViT transforms.

## Quick validation without a device

If MindSpore is not installed, the configuration path can still be checked:

```bash
python -m qkd_mindspore.main --config exps/smoke/mindspore_synthetic.json --dry-run
python -m qkd_mindspore.main --help
```

After installing the test dependencies, run the regression suite:

```bash
python -m unittest discover -s tests
```

The optional MindSpore tests cover `TruncatedSVD` preprocessing and a complete
network/prototype checkpoint round-trip; they are skipped automatically when
MindSpore is not installed.

The CPU smoke command is:

```bash
python -m qkd_mindspore.main \
  --config exps/smoke/mindspore_synthetic.json \
  --exp cpu_smoke \
  --device cpu \
  --device-target cpu \
  --ms-mode pynative \
  --num-workers 0
```

## Ascend 910B smoke test

```bash
cd /path/to/CVPR26-QKD
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python -m qkd_mindspore.main \
  --config exps/smoke/mindspore_synthetic.json \
  --exp npu_smoke \
  --device npu:0 \
  --device-target ascend \
  --ms-mode pynative \
  --num-workers 0
```

The smoke config uses six synthetic classes split into two tasks, a small 64x64 ViT, one adapter and quantum epoch per task, and one classifier-alignment epoch on the second task. It exercises the MindSpore/CANN software path end to end with minimal resources.

The equivalent launcher is:

```bash
bash scripts/run_mindspore_smoke.sh
# or, for compatibility with earlier notes:
bash scripts/run_npu_smoke.sh
```

## Real incremental experiment

```bash
export QKD_DATA_ROOT=/path/to/data
export QKD_CHECKPOINT_DIR=/path/to/checkpoints

python -m qkd_mindspore.main \
  --config exps/cifar/quan_adapter.json \
  --exp cifar_mindspore \
  --device npu:0 \
  --device-target ascend \
  --ms-mode pynative \
  --num-workers 0
```

Run `--max-tasks 2` first to exercise PQK distillation and classifier alignment before a full suite. See [docs/ASCEND_910B.md](docs/ASCEND_910B.md) for target-host checks and troubleshooting.

## Release notes

Before publishing, confirm the project licence with all authors and add the chosen `LICENSE` file. Review dataset/checkpoint licences separately. Do not commit weights, datasets, logs, generated results, or machine-specific paths.
