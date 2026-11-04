"""Portable path resolution shared by the MindSpore training path."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_FILENAMES = {
    "in1k": "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz",
    "in21k": "B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
}

DATASET_SUBDIRS = {
    "cifar10": "",
    "cifar100": "",
    "cifar224": "",
    "imagenet100": "imagenet100_subset",
    "imagenet1000": "imagenet",
    "imagenetr": "imagenet-r",
    "imagenet-r": "imagenet-r",
    "imageneta": "imagenet-a",
    "imagenet-a": "imagenet-a",
    "cub": "cub",
    "objectnet": "objectnet",
    "omnibenchmark": "omnibenchmark",
    "vtab": "vtab-cil/vtab",
}


def resolve_dataset_dir(dataset_name: str, config: Mapping[str, Any]) -> Path:
    name = dataset_name.lower()
    roots = config.get("data_roots") or config.get("dataset_roots") or {}
    aliases = {name, name.replace("-", ""), name.replace("-", "_")}
    if isinstance(roots, Mapping):
        for key, value in roots.items():
            if str(key).lower() in aliases:
                return Path(value).expanduser()

    common_root = config.get("data_root") or os.environ.get("QKD_DATA_ROOT")
    root = Path(common_root).expanduser() if common_root else REPO_ROOT / "data"
    subdir = DATASET_SUBDIRS.get(name, name)
    return root / subdir if subdir else root


def resolve_checkpoint_path(kind: str, config: Mapping[str, Any]) -> Path:
    if kind not in CHECKPOINT_FILENAMES:
        raise ValueError("checkpoint kind must be in1k or in21k")
    filename = CHECKPOINT_FILENAMES[kind]
    candidates: list[Path] = []

    for key in ("checkpoint_path", "pretrained_checkpoint"):
        value = config.get(key)
        if value:
            path = Path(value).expanduser()
            candidates.append(path / filename if path.is_dir() else path)
    for key in ("checkpoint_dir", "pretrained_dir"):
        value = config.get(key)
        if value:
            candidates.append(Path(value).expanduser() / filename)
    for env_name in ("QKD_CHECKPOINT_DIR", "QKD_PRETRAINED_DIR"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value).expanduser() / filename)
    candidates.append(REPO_ROOT / "checkpoints" / filename)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]
