# 昇腾 910B / CANN / MindSpore 适配指南

本文面向 Huawei Ascend 910B 单卡环境，主入口是根目录下的 `qkd_mindspore` 包。根目录的 `main.py`、`models/`、`backbone/` 和 `utils/` 是 PyTorch 参考实现；910B 入口是 `qkd_mindspore`。

本分支提供 Ascend 910B 单卡、CANN 与 MindSpore 适配代码。安装、最小
smoke、真实数据训练和排查命令见下文。

## 1. 版本检查

MindSpore 版本与主机上的 CANN、驱动和 Python 版本匹配。Ascend 版 MindSpore wheel 从华为昇腾软件源安装；目标主机运行 MindSpore/CANN 训练栈。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python - <<'PY'
import mindspore as ms
print("MindSpore:", ms.__version__)
print("device target:", ms.get_context("device_target"))
PY
```

记录并保留以下信息，便于定位算子兼容问题：

```text
CANN 版本：
驱动版本：
MindSpore 版本：
Python 版本：
910B 型号 / 卡数：
```

安装非 MindSpore 依赖：

```bash
pip install -r requirements-mindspore.txt
```

## 2. 配置检查

```bash
cd /path/to/CVPR26-QKD
python -m qkd_mindspore.main --config exps/smoke/mindspore_synthetic.json --dry-run
python -m qkd_mindspore.main --help
```

`--dry-run` 加载和校验配置，不导入 MindSpore 训练器。

## 3. 910B 最小 smoke

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python -m qkd_mindspore.main \
  --config exps/smoke/mindspore_synthetic.json \
  --exp npu_smoke \
  --device npu:0 \
  --device-target ascend \
  --ms-mode pynative \
  --num-workers 0
```

也可以执行：

```bash
bash scripts/run_mindspore_smoke.sh
```

该配置：

- 使用合成数据，不要求数据盘或预训练权重；
- 使用 6 个类、2 个增量任务、很小的 ViT 和量子电路；
- 每个任务执行 adapter 和量子路由器训练；第二个任务使用历史任务 PQK 原型权重做蒸馏，并执行一次全局分类头对齐；
- 覆盖软件链路、CANN 算子覆盖、数据搬运和反向传播。

smoke 使用 `--num-workers 0` 和 `--ms-mode pynative`；`--ms-mode graph` 启用 Graph 执行模式。

## 4. 两任务实验

smoke 配置本身就是两任务（`init_cls=3`、`increment=3`、`max_tasks=2`），因此它会经过：

- 新任务的 adapter 训练；
- 历史任务 PQK 原型权重和蒸馏项；
- 量子任务识别器训练；
- 单一全局分类头的 classifier alignment（历史类与新类共享同一个分类头）；
- 每任务评估和 `metrics.json` 写出。

可在 smoke 通过后逐步使用真实数据。先限制任务数和每类样本数：

```bash
export QKD_DATA_ROOT=/path/to/data
export QKD_CHECKPOINT_DIR=/path/to/checkpoints

python -m qkd_mindspore.main \
  --config exps/cifar/quan_adapter.json \
  --exp cifar_npu_two_task \
  --device npu:0 \
  --device-target ascend \
  --ms-mode pynative \
  --num-workers 0 \
  --pretrained \
  --max-tasks 2 \
  --max-samples-per-class 8
```

说明：`--pretrained` 需要配置中存在可加载的 `.npz`/`.ckpt` 权重；如果没有权重，保持 smoke 的 `pretrained: false`，或显式改用 `--no-pretrained`。不同实验的 `init_cls`/`increment` 不同，真实配置会启用更多类别。

## 5. 完整实验

运行 smoke 和两任务实验后，再运行完整配置：

```bash
python -m qkd_mindspore.main \
  --config exps/cifar/quan_adapter.json \
  --exp cifar_mindspore \
  --device npu:0 \
  --device-target ascend \
  --ms-mode pynative \
  --num-workers 0
```

数据路径、checkpoint 路径和输出目录都可用 CLI 或环境变量覆盖；数据和权重按实验配置放在本地目录。

## 6. 适配点概览

- `device.py`：把 `npu:0`/`Ascend:0` 映射为 MindSpore `Ascend` target，并设置 PYNATIVE/GRAPH 上下文。
- `model.py`：用 MindSpore `nn.Cell`、`Dense`、`LayerNorm`、`CellList` 重写 ViT、adapter、全局分类头和任务路由。
- `quantum.py`：用实部/虚部分离的浮点张量实现量子态，避免依赖 CANN 复数 dtype 算子；`StandardScaler + TruncatedSVD` 在主机侧按参考实现拟合一次。
- `trainer.py`：两阶段训练、PQK 蒸馏、分类头对齐、checkpoint 和指标写出。
- `data.py`：Pillow + NumPy 数据读取，不依赖 torchvision；CIFAR-10/100、ImageNet-1K 和其它 ViT 数据集分别使用与参考代码一致的增强和归一化策略。

## 7. 运行配置

- 训练入口为单进程、单卡。
- `TruncatedSVD` 预训练在宿主 CPU 侧完成一次，投影结果作为非训练参数进入 MindSpore；量子态、Pauli-Z 和约化密度矩阵计算在 MindSpore 张量上完成。
- `--ms-mode` 支持 `pynative` 和 `graph` 两种执行模式。
- `cifar10`/`cifar100` 使用原生 32x32 增强，配置中的 `img_size` 必须为 32；需要 224x224 ViT 时使用 `cifar224`。
- 每任务保存网络权重、PQK 原型和分类头对齐统计，用于推理恢复和后续分析。

## 8. 常见排查

**`ModuleNotFoundError: mindspore`**

确认在正确的 Python 环境中安装了 Ascend 版 MindSpore，并已 source CANN 的 `set_env.sh`。

**找不到 NPU / device target 错**

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
echo "$ASCEND_HOME_PATH"
python -c "import mindspore as ms; print(ms.get_context('device_target'))"
```

再显式传入 `--device npu:0 --device-target ascend`。

**CANN unsupported operator**

保持 `--ms-mode pynative --num-workers 0`，降低 `--batch-size`；记录算子名、输入 shape/dtype、CANN/MindSpore 版本和完整 traceback。若问题来自量子算子，可先用 `qkd_mindspore/quantum_math.py` 的 NumPy 参考对照数学结果。

**`ModuleNotFoundError: sklearn`**

安装 `requirements-mindspore.txt`。`scikit-learn` 只用于在主机 CPU 上拟合一次 `StandardScaler + TruncatedSVD`，不要求 CANN 提供 SVD 算子。

**显存或主机内存不足**

先使用 smoke；真实数据实验使用 `--max-tasks 1 --max-samples-per-class 8`，再逐步增加；checkpoint 保存按实验需要开启。

**checkpoint 不匹配**

确认 `.npz` key 是否包含 `module.`、`backbone.`、`vit.` 等前缀；加载器会剥离常见前缀。若 shape 不匹配，检查 ViT 规格和 `img_size`/`patch_size` 是否一致。
