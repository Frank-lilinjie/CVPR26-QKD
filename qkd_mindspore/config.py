"""Configuration helpers for the MindSpore QKD implementation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, MutableMapping


DEFAULTS: dict[str, Any] = {
    "framework": "mindspore",
    "dataset": "synthetic",
    "model_name": "quan_adapter",
    "backbone_type": "vit_base_patch16_224_in21k_quan_adapter",
    "prefix": "quan_adapter",
    "exp": "mindspore",
    "memory_size": 0,
    "memory_per_class": 0,
    "fixed_memory": False,
    "shuffle": True,
    "init_cls": 3,
    "increment": 3,
    "epochs": 1,
    "init_lr": 0.01,
    "batch_size": 4,
    "weight_decay": 0.0005,
    "min_lr": 0.0,
    "optimizer": "sgd",
    "scheduler": "cosine",
    "ffn_num": 16,
    "ffn_adapt": True,
    "ffn_option": "parallel",
    "ffn_adapter_init_option": "lora",
    "ffn_adapter_scalar": "0.1",
    "ffn_adapter_layernorm_option": "none",
    "drop": 0.0,
    "attn_drop": 0.0,
    "drop_path": 0.0,
    "pretrained": False,
    "img_size": 224,
    "patch_size": 16,
    "normalize": "auto",
    "image_mean": None,
    "image_std": None,
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "mlp_ratio": 4.0,
    "n_qubits": 3,
    "quantum_layers": 1,
    "quantum_epochs": 1,
    "quantum_fit_samples": 8192,
    "svd_dim": 6,
    "svd_random_state": 1993,
    "quantum_measure": "pqk",
    "save_reduced_states": True,
    "use_soft_routing": False,
    "soft_routing_temperature": 1.0,
    "crct_epochs": 1,
    "ca_lr": 0.005,
    "ca_samples_per_class": 0,
    "ca_storage_efficient_method": "variance",
    "n_centroids": 2,
    "use_distillation": True,
    "lambda_kd": 1.0,
    "lambda_sparse": 0.05,
    "kd_temperature": 1.0,
    "max_tasks": 0,
    "max_samples_per_class": 0,
    "print_forget": True,
    "num_workers": 0,
    "device": "npu:0",
    "device_target": "auto",
    "ms_mode": "pynative",
    "seed": 1993,
    "topk": 5,
    "output_dir": "results",
    "log_dir": "logs",
    "checkpoint_path": None,
    "checkpoint_dir": None,
    "data_root": None,
    "save_checkpoint": True,
    "clip_norm": 1.0,
    "synthetic_classes": 6,
    "synthetic_train_per_class": 8,
    "synthetic_test_per_class": 4,
}


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Configuration must be a JSON object: {path}")
    return payload


def first_scalar(value: Any, default: Any = None) -> Any:
    """Accept both scalar and legacy one-item-list configuration values."""
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def normalize_config(
    raw: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a validated, flat MindSpore experiment configuration.

    Legacy JSON files store ``device`` and ``seed`` as one-item lists. Those
    fields are normalised here while the remainder of the public configuration
    schema stays compatible with the original QuanAdapter releases.
    """
    config: dict[str, Any] = dict(DEFAULTS)
    config.update(dict(raw))
    if overrides:
        config.update({key: value for key, value in overrides.items() if value is not None})

    framework = str(config.get("framework", "mindspore")).strip().lower()
    if framework not in {"mindspore", "ms", "ascend", "npu", "cann"}:
        raise ValueError(
            "framework must be mindspore/ms/ascend when using qkd_mindspore; "
            "the legacy PyTorch entry point is main.py"
        )
    config["framework"] = "mindspore"

    config["device"] = str(first_scalar(config.get("device"), "npu:0"))
    config["seed"] = int(first_scalar(config.get("seed"), 1993))
    for key in (
        "init_cls",
        "increment",
        "batch_size",
        "max_tasks",
        "max_samples_per_class",
        "num_workers",
        "ffn_num",
        "n_qubits",
        "quantum_layers",
        "quantum_epochs",
        "quantum_fit_samples",
        "svd_dim",
        "epochs",
        "crct_epochs",
        "embed_dim",
        "depth",
        "num_heads",
        "img_size",
        "patch_size",
        "topk",
        "n_centroids",
        "ca_samples_per_class",
        "synthetic_classes",
        "synthetic_train_per_class",
        "synthetic_test_per_class",
    ):
        config[key] = int(config[key])
    for key in (
        "init_lr",
        "weight_decay",
        "min_lr",
        "ca_lr",
        "drop",
        "attn_drop",
        "drop_path",
        "mlp_ratio",
        "lambda_kd",
        "lambda_sparse",
        "kd_temperature",
        "clip_norm",
    ):
        config[key] = float(config[key])
    for key in (
        "pretrained",
        "use_distillation",
        "use_soft_routing",
        "save_checkpoint",
        "save_reduced_states",
        "ffn_adapt",
        "shuffle",
        "print_forget",
    ):
        config[key] = _as_bool(config[key])

    if config["n_qubits"] <= 0:
        raise ValueError("n_qubits must be positive")
    if config["svd_dim"] <= 0:
        raise ValueError("svd_dim must be positive")
    if config["img_size"] <= 0 or config["patch_size"] <= 0:
        raise ValueError("img_size and patch_size must be positive")
    if config["img_size"] % config["patch_size"] != 0:
        raise ValueError("img_size must be divisible by patch_size")
    if config["init_cls"] <= 0 or config["increment"] <= 0:
        raise ValueError("init_cls and increment must be positive")
    if config["batch_size"] <= 0:
        raise ValueError("batch_size must be positive")
    if config["num_workers"] < 0:
        raise ValueError("num_workers must be non-negative")
    if config["max_tasks"] < 0 or config["max_samples_per_class"] < 0:
        raise ValueError("max_tasks and max_samples_per_class must be non-negative")
    if config["topk"] <= 0:
        raise ValueError("topk must be positive")
    if config["clip_norm"] < 0:
        raise ValueError("clip_norm must be non-negative")
    if config["epochs"] <= 0:
        raise ValueError("epochs must be positive")
    if config["quantum_epochs"] < 0 or config["crct_epochs"] < 0:
        raise ValueError("quantum_epochs and crct_epochs must be non-negative")
    if config["quantum_fit_samples"] <= 0:
        raise ValueError("quantum_fit_samples must be positive")
    if config["embed_dim"] % config["num_heads"] != 0:
        raise ValueError("embed_dim must be divisible by num_heads")

    config["data_reuploading"] = int(
        config.get("data_reuploading")
        or max(1, math.ceil(config["svd_dim"] / config["n_qubits"]))
    )
    if config["data_reuploading"] <= 0:
        raise ValueError("data_reuploading must be positive")

    if str(config.get("device_target", "auto")).strip().lower() == "auto":
        legacy_backend = str(config.get("backend", "auto")).strip().lower()
        if legacy_backend in {"npu", "ascend", "cann"}:
            config["device_target"] = "ascend"
        elif legacy_backend == "cpu":
            config["device_target"] = "cpu"

    target = str(config.get("device_target", "auto")).strip().lower()
    aliases = {"ascend": "ascend", "npu": "ascend", "cann": "ascend", "auto": "auto", "cpu": "cpu"}
    if target not in aliases:
        raise ValueError("device_target must be auto, Ascend/NPU, or CPU")
    config["device_target"] = aliases[target]

    mode = str(config.get("ms_mode", "pynative")).lower()
    if mode not in {"pynative", "graph"}:
        raise ValueError("ms_mode must be pynative or graph")
    config["ms_mode"] = mode

    ffn_option = str(config.get("ffn_option", "parallel")).lower()
    if ffn_option not in {"parallel", "sequential"}:
        raise ValueError("ffn_option must be parallel or sequential")
    config["ffn_option"] = ffn_option

    layernorm_option = str(config.get("ffn_adapter_layernorm_option", "none")).lower()
    if layernorm_option not in {"none", "in", "out"}:
        raise ValueError("ffn_adapter_layernorm_option must be none, in, or out")
    config["ffn_adapter_layernorm_option"] = layernorm_option
    config["ffn_adapter_scalar"] = str(config.get("ffn_adapter_scalar", "0.1"))
    config["ffn_adapter_init_option"] = str(
        config.get("ffn_adapter_init_option", "lora")
    ).lower()

    storage = str(config["ca_storage_efficient_method"]).lower()
    if storage not in {"covariance", "variance", "multi-centroid"}:
        raise ValueError(f"Unsupported ca_storage_efficient_method: {storage}")
    config["ca_storage_efficient_method"] = storage

    config["optimizer"] = str(config["optimizer"]).lower()
    if config["optimizer"] not in {"sgd", "adam", "adamw"}:
        raise ValueError("optimizer must be sgd, adam, or adamw")
    config["scheduler"] = str(config["scheduler"]).lower()
    if config["scheduler"] not in {"cosine", "steplr", "constant"}:
        raise ValueError("scheduler must be cosine, steplr, or constant")

    if int(config.get("ca_samples_per_class") or 0) <= 0:
        config["ca_samples_per_class"] = config["batch_size"] * 5
    if config.get("output_dir"):
        config["output_dir"] = str(config["output_dir"])
    if config.get("log_dir"):
        config["log_dir"] = str(config["log_dir"])
    return config


def apply_cli_overrides(config: MutableMapping[str, Any], namespace: Any) -> None:
    """Copy non-None argparse values into a raw or normalised configuration."""
    mapping = {
        "num_workers": "num_workers",
        "epochs": "epochs",
        "quantum_epochs": "quantum_epochs",
        "crct_epochs": "crct_epochs",
        "max_tasks": "max_tasks",
        "max_samples_per_class": "max_samples_per_class",
        "checkpoint_path": "checkpoint_path",
        "checkpoint_dir": "checkpoint_dir",
        "data_root": "data_root",
        "n_qubits": "n_qubits",
        "svd_dim": "svd_dim",
        "batch_size": "batch_size",
        "device_target": "device_target",
        "ms_mode": "ms_mode",
        "init_cls": "init_cls",
        "increment": "increment",
        "dataset": "dataset",
        "exp": "exp",
        "output_dir": "output_dir",
        "log_dir": "log_dir",
        "pretrained": "pretrained",
    }
    for source, destination in mapping.items():
        value = getattr(namespace, source, None)
        if value is not None:
            config[destination] = value
    if getattr(namespace, "device", None) is not None:
        config["device"] = namespace.device
    if getattr(namespace, "seed", None) is not None:
        config["seed"] = namespace.seed
