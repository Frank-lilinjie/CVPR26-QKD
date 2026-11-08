# QKD / QuanAdapter

Research code for **class-incremental learning with a quantum task identifier and adapter-based ViT backbones**.

> **分支说明：昇腾 910B NPU 适配版本。** This branch adds the Huawei Ascend
> 910B + CANN + MindSpore implementation. The original PyTorch files are
> retained as the reference implementation.

## Implementation

The supported implementation for **Huawei Ascend 910B + CANN** is the MindSpore package in `qkd_mindspore/`. It follows the QKD/QuanAdapter method while avoiding the Python-level `torch_npu` and complex-tensor dependencies of the original implementation.

The root-level PyTorch files (`main.py`, `trainer.py`, `models/`, `backbone/`, `utils/`) provide the PyTorch reference implementation for method comparison and paper reproduction. The Ascend 910B entry point is `qkd_mindspore/`.

> Branch contents: the MindSpore/CANN implementation is provided in
> `qkd_mindspore/`. Installation, launch commands, experiment layouts and
> target-host notes are documented in `docs/ASCEND_910B.md`.

## Method overview

1. **ViT adapter backbone** (`qkd_mindspore/model.py`): ViT with a task-specific bottleneck adapter bank.
2. **Quantum task identifier** (`qkd_mindspore/quantum.py`): StandardScaler + TruncatedSVD feature projection, data-reuploading RY circuit, Pauli-Z readout, and task logits.
3. **PQK task weighting** (`qkd_mindspore/quantum.py`, `qkd_mindspore/trainer.py`): reduced density matrices are accumulated as task prototypes and used for task-weighted distillation/routing.
4. **Global classifier and alignment** (`qkd_mindspore/trainer.py`): every adapter shares one global classifier. Class feature statistics are sampled to align that classifier from the second task onward, while old class logits remain available for distillation.

`qkd_mindspore/quantum_math.py` is a pure-NumPy reference for the circuit and reduced-density-matrix mathematics, independent of MindSpore.

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
scripts/run_torch_npu_smoke.sh  PyTorch/torch_npu launcher
scripts/run_qsvd_grid.py        PyTorch/CUDA/Ascend grid launcher

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

MindSpore for Ascend is installed separately from Huawei's Ascend software source using the wheel built for the host's CANN toolkit and driver. Source CANN and check the interpreter before training:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python - <<'PY'
import mindspore as ms
print("mindspore:", ms.__version__)
print("device target:", ms.get_context("device_target"))
PY
```

The MindSpore trainer runs in one process on one device. Start with `--device npu:0`, `--device-target ascend`, `--ms-mode pynative`, and `--num-workers 0`.

## Data and checkpoints

Paths are resolved portably; no host-specific absolute paths are embedded in the source:

- Dataset root: `--data-root`, config `data_root`/`data_roots`, `QKD_DATA_ROOT`, then `./data`.
- ViT checkpoint: `--checkpoint-path`, config `checkpoint_path`, `--checkpoint-dir`, `QKD_CHECKPOINT_DIR`, then `./checkpoints`.

The timm ViT-B/16 checkpoint filenames expected by `qkd_mindspore/paths.py` are:

```text
B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz
B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz
```

The synthetic smoke configuration sets `pretrained: false`. Full experiments use the matching `.npz` or a MindSpore `.ckpt` containing the backbone weights. The public source tree contains code and configurations; datasets, checkpoints, logs and results are produced in the local experiment directories.

With `save_checkpoint: true`, each task writes `task_XX.ckpt` (network
parameters), `task_XX_prototypes.npz` (PQK task prototypes) and
`task_XX_stats.json` (classifier-alignment statistics). These artifacts provide
per-task inference restore.

Data augmentation follows the reference dataset classes rather than one
universal transform: CIFAR-10/100 use native 32x32 random crop, horizontal flip,
brightness jitter and dataset normalisation; ImageNet-1K uses 224x224
RandomResizedCrop, horizontal flip, brightness jitter and ImageNet normalisation;
the remaining ViT datasets use the reference 224x224 crop/flip/resize pipeline.
The paired `cifar224` variant keeps its 224x224 ViT transforms.

## Configuration check

The effective configuration can be checked before launching training:

```bash
python -m qkd_mindspore.main --config exps/smoke/mindspore_synthetic.json --dry-run
python -m qkd_mindspore.main --help
```

After installing the test dependencies, run the regression suite:

```bash
python -m unittest discover -s tests
```

The optional MindSpore tests cover `TruncatedSVD` preprocessing and a complete
network/prototype checkpoint round-trip.

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

The smoke config uses six synthetic classes split into two tasks, a small 64x64 ViT, one adapter and quantum epoch per task, and one classifier-alignment epoch on the second task. It covers the MindSpore/CANN execution path with minimal resources.

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
