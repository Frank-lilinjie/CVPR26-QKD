"""MindSpore ViT, task adapters, quantum identifier and incremental heads."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from mindspore import Parameter, Tensor, load_checkpoint, load_param_into_net, nn, ops, save_checkpoint
from mindspore.common.initializer import Normal, Zero

from .quantum import QuantumTaskIdentifier


def _dropout(rate: float) -> nn.Cell:
    """Create Dropout with the API supported by the installed MindSpore."""
    rate = float(rate)
    try:
        return nn.Dropout(p=rate)
    except TypeError:
        # MindSpore < 2.4 exposes the complementary ``keep_prob`` argument.
        return nn.Dropout(1.0 - rate)


class DropPath(nn.Cell):
    """Per-sample stochastic depth used by the reference ViT implementation."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def construct(self, x: Tensor) -> Tensor:
        if self.drop_prob <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = ops.floor(ops.rand(shape, dtype=x.dtype) + keep_prob)
        return x * mask / keep_prob


class PatchEmbed(nn.Cell):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size")
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.grid_size = (self.img_size // self.patch_size, self.img_size // self.patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            pad_mode="valid",
            has_bias=True,
            weight_init=Normal(0.02),
            bias_init=Zero(),
        )

    def construct(self, images: Tensor) -> Tensor:
        x = self.proj(images)
        batch, channels, height, width = x.shape
        x = ops.reshape(x, (batch, channels, height * width))
        return ops.transpose(x, (0, 2, 1))


class Attention(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Dense(dim, dim, has_bias=qkv_bias, weight_init=Normal(0.02), bias_init=Zero())
        self.k_proj = nn.Dense(dim, dim, has_bias=qkv_bias, weight_init=Normal(0.02), bias_init=Zero())
        self.v_proj = nn.Dense(dim, dim, has_bias=qkv_bias, weight_init=Normal(0.02), bias_init=Zero())
        self.attn_drop = _dropout(attn_drop)
        self.proj = nn.Dense(dim, dim, weight_init=Normal(0.02), bias_init=Zero())
        self.proj_drop = _dropout(proj_drop)

    def _shape(self, tensor: Tensor, seq_len: int, batch: int) -> Tensor:
        tensor = ops.reshape(tensor, (batch, seq_len, self.num_heads, self.head_dim))
        return ops.transpose(tensor, (0, 2, 1, 3))

    def construct(self, x: Tensor) -> Tensor:
        batch, seq_len, channels = x.shape
        q = self._shape(self.q_proj(x), seq_len, batch)
        k = self._shape(self.k_proj(x), seq_len, batch)
        v = self._shape(self.v_proj(x), seq_len, batch)
        attn = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = self.attn_drop(ops.softmax(attn, axis=-1))
        out = ops.matmul(attn, v)
        out = ops.transpose(out, (0, 2, 1, 3))
        out = ops.reshape(out, (batch, seq_len, channels))
        return self.proj_drop(self.proj(out))


class Mlp(nn.Cell):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Dense(dim, hidden_dim, weight_init=Normal(0.02), bias_init=Zero())
        self.fc2 = nn.Dense(hidden_dim, dim, weight_init=Normal(0.02), bias_init=Zero())
        self.drop = _dropout(drop)

    def construct(self, x: Tensor) -> Tensor:
        x = self.drop(ops.gelu(self.fc1(x)))
        return self.drop(self.fc2(x))


class Adapter(nn.Cell):
    """Bottleneck adapter inserted after the attention block."""

    def __init__(
        self,
        dim: int,
        bottleneck: int = 16,
        dropout: float = 0.25,
        init_option: str = "lora",
        adapter_scalar: str = "1.0",
        layernorm_option: str = "in",
        seed: int = 1993,
    ):
        super().__init__()
        if str(init_option).lower() != "lora":
            raise NotImplementedError(
                "Only the LoRA-style identity adapter used by the paper is supported"
            )
        self.dim = int(dim)
        self.bottleneck = int(bottleneck)
        self.layernorm_option = str(layernorm_option)
        self.adapter_layer_norm = (
            nn.LayerNorm((self.dim,), epsilon=1e-6)
            if self.layernorm_option in {"in", "out"}
            else None
        )
        self.down_proj = nn.Dense(
            self.dim, self.bottleneck, weight_init=Normal(0.02), bias_init=Zero()
        )
        self.up_proj = nn.Dense(
            self.bottleneck, self.dim, weight_init=Normal(0.02), bias_init=Zero()
        )
        self.dropout = _dropout(dropout)
        if str(adapter_scalar).lower() == "learnable_scalar":
            self.scale = Parameter(Tensor(np.ones((1,), dtype=np.float32)), name="adapter_scale")
            self.learnable_scale = True
        else:
            self.scale = float(adapter_scalar)
            self.learnable_scale = False

        # Match ``nn.init.kaiming_uniform_(down_proj.weight, a=sqrt(5))``
        # followed by zero initialisation of the up projection in the
        # reference implementation.
        rng = np.random.default_rng(int(seed))
        bound = 1.0 / math.sqrt(float(self.dim))
        down_weight = rng.uniform(
            -bound, bound, size=tuple(self.down_proj.weight.shape)
        ).astype(np.float32)
        down_bias = np.zeros(tuple(self.down_proj.bias.shape), dtype=np.float32)
        self.down_proj.weight.set_data(Tensor(down_weight))
        self.down_proj.bias.set_data(Tensor(down_bias))
        self.up_proj.weight.set_data(ops.zeros_like(self.up_proj.weight))
        self.up_proj.bias.set_data(ops.zeros_like(self.up_proj.bias))

    def construct(self, x: Tensor, add_residual: bool = True, residual: Optional[Tensor] = None) -> Tensor:
        residual = x if residual is None else residual
        if self.layernorm_option == "in":
            x = self.adapter_layer_norm(x)
        x = self.dropout(self.down_proj(x))
        x = self.up_proj(x)
        x = x * self.scale
        if self.layernorm_option == "out":
            x = self.adapter_layer_norm(x)
        return x + residual if add_residual else x


class Block(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: float,
        ffn_option: str = "parallel",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm((dim,), epsilon=1e-6)
        self.attn = Attention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm((dim,), epsilon=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)
        self.ffn_option = str(ffn_option)

    def construct(self, x: Tensor, adapter: Optional[Adapter] = None) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        adapt_x = adapter(x, add_residual=False) if adapter is not None else None
        residual = x
        x = self.drop_path(self.mlp(self.norm2(x)))
        if adapter is not None:
            if self.ffn_option == "sequential":
                x = adapter(x, add_residual=True)
            elif self.ffn_option == "parallel":
                x = x + adapt_x
            else:
                raise ValueError("ffn_option must be parallel or sequential")
        return residual + x


class TaskAdapters(nn.Cell):
    def __init__(self, depth: int, dim: int, config: dict, seed: int):
        super().__init__()
        self.blocks = nn.CellList(
            [
                Adapter(
                    dim=dim,
                    bottleneck=int(config["ffn_num"]),
                    dropout=0.1,
                    init_option=str(config["ffn_adapter_init_option"]),
                    adapter_scalar=str(config["ffn_adapter_scalar"]),
                    layernorm_option=str(config["ffn_adapter_layernorm_option"]),
                    seed=int(seed) + layer_id * 131,
                )
                for layer_id in range(depth)
            ]
        )


class VisionTransformerBackbone(nn.Cell):
    """ViT-B/16 with an independent adapter bank for every incremental task."""

    def __init__(self, config: dict, num_tasks: int):
        super().__init__()
        dim = int(config["embed_dim"])
        depth = int(config["depth"])
        self.embed_dim = dim
        self.depth = depth
        self.patch_embed = PatchEmbed(
            img_size=int(config["img_size"]),
            patch_size=int(config["patch_size"]),
            embed_dim=dim,
        )
        rng = np.random.default_rng(int(config["seed"]))
        self.cls_token = Parameter(
            Tensor(rng.normal(0.0, 0.02, size=(1, 1, dim)).astype(np.float32)),
            name="cls_token",
        )
        self.pos_embed = Parameter(
            Tensor(
                rng.normal(
                    0.0,
                    0.02,
                    size=(1, self.patch_embed.num_patches + 1, dim),
                ).astype(np.float32)
            ),
            name="pos_embed",
        )
        self.pos_drop = _dropout(config["drop"])
        dpr = np.linspace(0.0, float(config["drop_path"]), depth).tolist()
        self.blocks = nn.CellList(
            [
                Block(
                    dim=dim,
                    num_heads=int(config["num_heads"]),
                    mlp_ratio=float(config["mlp_ratio"]),
                    qkv_bias=True,
                    drop=float(config["drop"]),
                    attn_drop=float(config["attn_drop"]),
                    drop_path=float(dpr[index]),
                    ffn_option=str(config["ffn_option"]),
                )
                for index in range(depth)
            ]
        )
        self.norm = nn.LayerNorm((dim,), epsilon=1e-6)
        self.adapters = nn.CellList(
            [
                TaskAdapters(
                    depth,
                    dim,
                    config,
                    seed=int(config["seed"]) + task_id * 10_007,
                )
                for task_id in range(int(num_tasks))
            ]
        )
        self._freeze_backbone_parameters()

    def _freeze_backbone_parameters(self) -> None:
        """Freeze the ViT while leaving every task adapter trainable.

        Calling ``get_parameters()`` on the whole backbone also returns the
        adapter parameters.  Select by name instead so adapter-only training
        remains effective.
        """
        for name, parameter in self.parameters_and_names():
            parameter.requires_grad = "adapters" in name.split(".")

    def embed_tokens(self, images: Tensor) -> Tensor:
        """Embed a batch once so multiple adapters can reuse the same tokens."""
        batch = images.shape[0]
        x = self.patch_embed(images)
        cls_tokens = ops.tile(self.cls_token, (batch, 1, 1))
        x = ops.concat((cls_tokens, x), axis=1) + self.pos_embed
        return self.pos_drop(x)

    def forward_tokens(self, tokens: Tensor, task_id: int = -1) -> Tensor:
        x = tokens
        for index, block in enumerate(self.blocks):
            adapter = self.adapters[task_id].blocks[index] if task_id >= 0 else None
            x = block(x, adapter)
        return self.norm(x)[:, 0]

    def construct(self, images: Tensor, task_id: int = -1) -> Tensor:
        return self.forward_tokens(self.embed_tokens(images), task_id)


class QuanAdapterNetwork(nn.Cell):
    """Full QKD/QuanAdapter model used by the MindSpore trainer."""

    def __init__(self, config: dict, task_sizes: Sequence[int]):
        super().__init__()
        self.config = dict(config)
        self.task_sizes = [int(size) for size in task_sizes]
        self.num_tasks = len(self.task_sizes)
        self.class_starts = np.cumsum([0] + self.task_sizes[:-1]).astype(np.int64).tolist()
        embed_dim = int(config["embed_dim"])
        self.backbone = VisionTransformerBackbone(config, self.num_tasks)
        self.head = nn.Dense(
            embed_dim,
            self.total_classes,
            has_bias=True,
            weight_init=Normal(0.02),
            bias_init=Zero(),
        )
        rng = np.random.default_rng(int(config["seed"]) + 31_337)
        head_bound = 1.0 / math.sqrt(float(embed_dim))
        self.head.weight.set_data(
            Tensor(
                rng.uniform(
                    -head_bound,
                    head_bound,
                    size=tuple(self.head.weight.shape),
                ).astype(np.float32)
            )
        )
        self.head.bias.set_data(
            Tensor(
                rng.uniform(
                    -head_bound,
                    head_bound,
                    size=tuple(self.head.bias.shape),
                ).astype(np.float32)
            )
        )
        self.quantum = QuantumTaskIdentifier(
            feature_dim=int(config["svd_dim"]),
            n_qubits=int(config["n_qubits"]),
            n_layers=int(config["quantum_layers"]),
            max_tasks=self.num_tasks,
            data_reuploading=int(config["data_reuploading"]),
            n_readout_qubits=min(3, int(config["n_qubits"])),
            svd_random_state=int(config["svd_random_state"]),
            backbone_dim=embed_dim,
            pqk_temperature=float(config["soft_routing_temperature"]),
        )

    @property
    def total_classes(self) -> int:
        return int(sum(self.task_sizes))

    def task_range(self, task_id: int) -> Tuple[int, int]:
        start = self.class_starts[int(task_id)]
        return int(start), int(start + self.task_sizes[int(task_id)])

    def base_features(self, images: Tensor) -> Tensor:
        return self.backbone(images, -1)

    def task_features(self, images: Tensor, task_id: int) -> Tensor:
        return self.backbone(images, int(task_id))

    def task_feature_stack(self, images: Tensor, task_ids: Sequence[int]) -> Tensor:
        """Return ``[B, K, D]`` features without recomputing patch embeddings."""
        selected_tasks = [int(task_id) for task_id in task_ids]
        if not selected_tasks:
            raise ValueError("task_feature_stack requires at least one task id")
        tokens = self.backbone.embed_tokens(images)
        features = [
            self.backbone.forward_tokens(tokens, task_id)
            for task_id in selected_tasks
        ]
        return ops.stack(features, axis=1)

    def head_logits(self, features: Tensor, task_id: Optional[int] = None) -> Tensor:
        """Apply the single global classifier to 2-D or stacked 3-D features."""
        del task_id
        if features.ndim == 2:
            return self.head(features)
        leading_shape = tuple(features.shape[:-1])
        flat_batch = int(np.prod(leading_shape))
        flat = ops.reshape(features, (flat_batch, int(features.shape[-1])))
        logits = self.head(flat)
        return ops.reshape(logits, leading_shape + (self.total_classes,))

    def train_logits(self, images: Tensor, task_id: int) -> Tuple[Tensor, Tensor]:
        features = self.task_features(images, task_id)
        return self.head_logits(features), features

    def adapter_parameters(self, task_id: int) -> List[Parameter]:
        return list(self.backbone.adapters[int(task_id)].get_parameters())

    def head_parameters(self, task_ids: Optional[Sequence[int]] = None) -> List[Parameter]:
        """Return the global head parameters.

        ``task_ids`` is accepted only for compatibility with earlier adapters
        of this release; the paper uses one classifier shared by every task.
        """
        del task_ids
        return list(self.head.get_parameters())

    def quantum_parameters(self) -> List[Parameter]:
        return self.quantum.trainable_parameters()

    def task_weights(self, images: Tensor, task_ids: Sequence[int]) -> Tensor:
        return self.quantum.predict_weights(self.base_features(images), task_ids)

    def _routing_gate_weights(
        self, base_features: Tensor, selected_tasks: Sequence[int]
    ) -> Tensor:
        """Quantum readout gate used by the paper's hard-routing branch."""
        num_active = len(selected_tasks)
        readout, task_logits = self.quantum.readout_and_logits(base_features)
        if int(readout.shape[1]) >= num_active:
            gate_logits = readout[:, :num_active]
        elif int(task_logits.shape[1]) >= num_active:
            gate_logits = task_logits[:, :num_active]
        else:
            gate_logits = ops.ones(
                (int(base_features.shape[0]), num_active), base_features.dtype
            )
        return ops.softmax(gate_logits, axis=-1)

    @staticmethod
    def _confidence_from_logits(logits: Tensor) -> Tensor:
        probabilities = ops.softmax(logits, axis=-1)
        num_classes = int(probabilities.shape[-1])
        if num_classes < 2:
            return ops.ones_like(ops.squeeze(probabilities, axis=-1))
        top_values = ops.topk(
            probabilities, k=2, dim=-1, largest=True, sorted=True
        )[0]
        first, second = ops.unstack(top_values, axis=-1)
        return first - second

    def routed_logits(
        self,
        images: Tensor,
        seen_tasks: Sequence[int],
        soft_routing: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        selected_tasks = [int(task_id) for task_id in seen_tasks]
        if not selected_tasks:
            raise ValueError("routed_logits requires at least one seen task")

        tokens = self.backbone.embed_tokens(images)
        base_features = self.backbone.forward_tokens(tokens, -1)
        adapter_features = ops.stack(
            [
                self.backbone.forward_tokens(tokens, task_id)
                for task_id in selected_tasks
            ],
            axis=1,
        )
        task_table = Tensor(np.asarray(selected_tasks, dtype=np.int32))

        if soft_routing:
            _, task_logits = self.quantum.readout_and_logits(base_features)
            active_logits = ops.stack(
                [task_logits[:, task_id] for task_id in selected_tasks], axis=1
            )
            weights = ops.softmax(
                active_logits / max(float(temperature), 1e-6), axis=-1
            )
        else:
            gate_weights = self._routing_gate_weights(base_features, selected_tasks)
            adapter_logits = self.head_logits(adapter_features)
            confidence = self._confidence_from_logits(adapter_logits)
            weights = ops.softmax(gate_weights * confidence, axis=-1)

        routed_features = ops.sum(
            adapter_features * ops.expand_dims(weights, -1), dim=1
        )
        global_logits = self.head_logits(routed_features)
        predicted_index = ops.argmax(weights, 1)
        predicted_tasks = ops.gather(task_table, predicted_index, axis=0)
        return global_logits, predicted_tasks, weights

    def load_pretrained_npz(self, path: str | Path, strict: bool = False) -> int:
        """Load a timm ViT-B/16 ``.npz`` checkpoint into the backbone.

        Only backbone tensors are consumed. Adapter, classifier and quantum
        parameters are intentionally left at their task-specific initialisation.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        loaded = 0
        with np.load(path, allow_pickle=False) as archive:
            arrays = {str(key): np.asarray(archive[key]) for key in archive.files}

        def strip_prefix(key: str) -> str:
            prefixes = ("module.", "model.", "backbone.", "vit.", "encoder.")
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if key.startswith(prefix):
                        key = key.split(".", 1)[1]
                        changed = True
                        break
            return key

        arrays = {strip_prefix(key): value for key, value in arrays.items()}

        def assign(parameter: Parameter, name: str, value: np.ndarray) -> None:
            nonlocal loaded
            if tuple(parameter.shape) != tuple(value.shape):
                if strict:
                    raise ValueError(
                        "Shape mismatch for {}: checkpoint {} vs parameter {}".format(
                            name, tuple(value.shape), tuple(parameter.shape)
                        )
                    )
                logging.warning("Skip %s: checkpoint/parameter shape mismatch", name)
                return
            parameter.set_data(Tensor(np.ascontiguousarray(value, dtype=np.float32)))
            loaded += 1

        if "patch_embed.proj.weight" in arrays:
            assign(self.backbone.patch_embed.proj.weight, "patch_embed.proj.weight", arrays["patch_embed.proj.weight"])
        if "patch_embed.proj.bias" in arrays:
            assign(self.backbone.patch_embed.proj.bias, "patch_embed.proj.bias", arrays["patch_embed.proj.bias"])
        if "cls_token" in arrays:
            assign(self.backbone.cls_token, "cls_token", arrays["cls_token"])
        if "pos_embed" in arrays:
            assign(self.backbone.pos_embed, "pos_embed", arrays["pos_embed"])
        if "norm.weight" in arrays:
            assign(self.backbone.norm.gamma, "norm.weight", arrays["norm.weight"])
        if "norm.bias" in arrays:
            assign(self.backbone.norm.beta, "norm.bias", arrays["norm.bias"])

        for index, block in enumerate(self.backbone.blocks):
            prefix = "blocks.{}.".format(index)
            qkv_weight = arrays.get(prefix + "attn.qkv.weight")
            qkv_bias = arrays.get(prefix + "attn.qkv.bias")
            if qkv_weight is not None and qkv_weight.shape[0] == 3 * int(self.config["embed_dim"]):
                q_weight, k_weight, v_weight = np.split(qkv_weight, 3, axis=0)
                assign(block.attn.q_proj.weight, prefix + "attn.q_proj.weight", q_weight)
                assign(block.attn.k_proj.weight, prefix + "attn.k_proj.weight", k_weight)
                assign(block.attn.v_proj.weight, prefix + "attn.v_proj.weight", v_weight)
            if qkv_bias is not None and qkv_bias.shape[0] == 3 * int(self.config["embed_dim"]):
                q_bias, k_bias, v_bias = np.split(qkv_bias, 3, axis=0)
                assign(block.attn.q_proj.bias, prefix + "attn.q_proj.bias", q_bias)
                assign(block.attn.k_proj.bias, prefix + "attn.k_proj.bias", k_bias)
                assign(block.attn.v_proj.bias, prefix + "attn.v_proj.bias", v_bias)

            mappings = {
                "norm1.weight": block.norm1.gamma,
                "norm1.bias": block.norm1.beta,
                "attn.proj.weight": block.attn.proj.weight,
                "attn.proj.bias": block.attn.proj.bias,
                "norm2.weight": block.norm2.gamma,
                "norm2.bias": block.norm2.beta,
                "mlp.fc1.weight": block.mlp.fc1.weight,
                "mlp.fc1.bias": block.mlp.fc1.bias,
                "mlp.fc2.weight": block.mlp.fc2.weight,
                "mlp.fc2.bias": block.mlp.fc2.bias,
            }
            for suffix, parameter in mappings.items():
                value = arrays.get(prefix + suffix)
                if value is not None:
                    assign(parameter, prefix + suffix, value)

        logging.info("Loaded %d backbone tensors from %s", loaded, path)
        return loaded

    def load_mindspore_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        parameters = load_checkpoint(str(path))
        param_not_load, ckpt_not_load = load_param_into_net(
            self, parameters, strict_load=False
        )
        loaded_names = set(parameters) - set(ckpt_not_load)
        logging.info(
            "Loaded %d MindSpore checkpoint tensors from %s",
            len(loaded_names),
            path,
        )
        if param_not_load:
            logging.warning("Network parameters not found in checkpoint: %s", param_not_load)
        if ckpt_not_load:
            logging.warning("Checkpoint tensors not loaded: %s", ckpt_not_load)
        if self.quantum.preprocessor.is_fitted or any(
            "preprocessor.svd_components" in name for name in loaded_names
        ):
            self.quantum.preprocessor.mark_fitted()

    def save_mindspore_checkpoint(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(self, str(path))
        return path

    def construct(self, images: Tensor) -> Tensor:
        return self.base_features(images)
