"""Portable dataset and pretrained-checkpoint path resolution."""

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


def _args(args: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return args or {}


def _checkpoint_file(value: str | os.PathLike[str], filename: str) -> Path:
    path = Path(value).expanduser()
    return path / filename if path.is_dir() or str(value).endswith(os.sep) else path


def resolve_checkpoint_path(
    kind: str,
    args: Mapping[str, Any] | None = None,
) -> str:
    """Resolve a timm ViT ``.npz`` checkpoint without machine-specific paths.

    Priority: explicit ``checkpoint_path``/``pretrained_checkpoint``, then
    ``checkpoint_dir``, ``QKD_CHECKPOINT_DIR``, repository ``checkpoints/`` and
    the old Lambda paths as a final compatibility fallback.
    """
    if kind not in CHECKPOINT_FILENAMES:
        raise ValueError(f"Unknown checkpoint kind {kind!r}; use in1k or in21k.")

    args = _args(args)
    filename = CHECKPOINT_FILENAMES[kind]
    candidates: list[Path] = []

    for key in ("checkpoint_path", "pretrained_checkpoint"):
        value = args.get(key)
        if value:
            candidates.append(_checkpoint_file(value, filename))

    for key in ("checkpoint_dir", "pretrained_dir"):
        value = args.get(key)
        if value:
            candidates.append(Path(value).expanduser() / filename)

    for env_name in ("QKD_CHECKPOINT_DIR", "QKD_PRETRAINED_DIR"):
        value = os.environ.get(env_name)
        if value:
            candidates.append(Path(value).expanduser() / filename)

    candidates.extend(
        [
            REPO_ROOT / "checkpoints" / filename,
        ]
    )

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(candidates[0])


def _dataset_mapping(args: Mapping[str, Any]) -> Mapping[str, Any]:
    roots = args.get("data_roots") or args.get("dataset_roots") or {}
    return roots if isinstance(roots, Mapping) else {}


def resolve_dataset_dir(
    dataset_name: str,
    args: Mapping[str, Any] | None = None,
    *,
    legacy_root: str | os.PathLike[str] | None = None,
) -> str:
    """Return the dataset directory for this project.

    ``data_roots`` may map a dataset name directly to its exact directory.
    Otherwise ``data_root`` (or ``QKD_DATA_ROOT``) is treated as a common
    parent containing the conventional dataset subdirectory.
    """
    args = _args(args)
    name = dataset_name.lower()
    aliases = {name, name.replace("-", ""), name.replace("-", "_")}
    mapping = _dataset_mapping(args)
    for key, value in mapping.items():
        if str(key).lower() in aliases:
            return str(Path(value).expanduser())

    subdir = DATASET_SUBDIRS.get(name, name)
    common_root = args.get("data_root") or os.environ.get("QKD_DATA_ROOT")
    if common_root:
        root = Path(common_root).expanduser()
    elif legacy_root is not None and Path(legacy_root).expanduser().exists():
        root = Path(legacy_root).expanduser()
    else:
        root = REPO_ROOT / "data"
    return str(root / subdir if subdir else root)
