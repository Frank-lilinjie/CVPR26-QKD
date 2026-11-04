"""MindSpore context and Ascend device helpers."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Tuple


def _auto_target() -> str:
    backend = os.environ.get("QKD_BACKEND", "").strip().lower()
    if backend in {"npu", "ascend", "cann"}:
        return "Ascend"
    if backend == "cpu":
        return "CPU"
    if os.environ.get("ASCEND_HOME_PATH"):
        return "Ascend"
    if os.path.isdir("/usr/local/Ascend/ascend-toolkit"):
        return "Ascend"
    return "CPU"


def parse_device_spec(device: str, requested_target: str = "auto") -> Tuple[str, int]:
    """Return ``(device_target, device_id)`` for a QKD device string.

    Accepted examples: ``Ascend:0``, ``ascend``, ``npu:0``, ``0``, ``CPU`` and
    ``auto``. ``npu`` is an alias of MindSpore's ``Ascend`` target. ``auto``
    selects Ascend when ``QKD_BACKEND=npu`` or a CANN installation is visible;
    otherwise it selects CPU.
    """
    value = str(device).strip()
    target_hint = str(requested_target).strip().lower()
    target_aliases = {"ascend": "Ascend", "npu": "Ascend", "cann": "Ascend", "cpu": "CPU"}

    if target_hint in target_aliases:
        forced_target = target_aliases[target_hint]
    elif target_hint == "auto":
        forced_target = None
    else:
        raise ValueError("device_target must be auto, Ascend/NPU, or CPU")

    if value.lower() == "cpu":
        return "CPU", 0
    if value.lower().startswith("cuda"):
        raise ValueError(
            "MindSpore on Ascend does not use CUDA device strings. "
            "Use npu:0/Ascend:0 or --backend cpu."
        )

    match = re.fullmatch(r"(?i)(ascend|npu|cpu|auto)(?::(\d+))?", value)
    if match:
        family, device_id = match.groups()
        family = family.lower()
        if family == "cpu":
            return "CPU", 0
        if family in {"ascend", "npu"}:
            return "Ascend", int(device_id or 0)
        selected = forced_target or _auto_target()
        return selected, int(device_id or 0) if selected == "Ascend" else 0

    if value.isdigit():
        selected = forced_target or _auto_target()
        return selected, int(value) if selected == "Ascend" else 0
    raise ValueError(f"Unrecognised device specification: {device!r}")


def setup_context(config: dict[str, Any]) -> dict[str, Any]:
    """Initialise the MindSpore runtime and return effective device metadata."""
    import mindspore as ms

    target, device_id = parse_device_spec(
        str(config.get("device", "npu:0")),
        str(config.get("device_target", "auto")),
    )
    if target != "Ascend":
        device_id = 0
    mode = ms.GRAPH_MODE if config.get("ms_mode") == "graph" else ms.PYNATIVE_MODE

    # MindSpore 2.7+ deprecates device_target/device_id on set_context. Keep
    # the old path for vendor wheels built against earlier MindSpore releases.
    set_device = getattr(ms, "set_device", None)
    if callable(set_device):
        set_device(target, device_id)
        ms.set_context(mode=mode)
    else:
        context_kwargs: dict[str, Any] = {
            "mode": mode,
            "device_target": target,
        }
        if target == "Ascend":
            context_kwargs["device_id"] = device_id
        ms.set_context(**context_kwargs)
    ms.set_seed(int(config.get("seed", 1993)))

    if mode == ms.GRAPH_MODE:
        logging.warning(
            "MindSpore graph mode is experimental for this dynamic incremental "
            "trainer. Use ms_mode=pynative for the first 910B validation."
        )

    info = {
        "mindspore_version": getattr(ms, "__version__", "unknown"),
        "device_target": target,
        "device_id": device_id,
        "mode": "graph" if mode == ms.GRAPH_MODE else "pynative",
        "ascend_home_path": os.environ.get("ASCEND_HOME_PATH", ""),
    }
    return info


def tensor_to_numpy(value: Any):
    return value.asnumpy()


def synchronize() -> None:
    """Best-effort device synchronisation for timing code."""
    try:
        import mindspore as ms

        target = str(ms.get_context("device_target")).strip().lower()
        if target not in {"ascend", "npu"}:
            return
        runtime = getattr(ms, "runtime", None)
        if runtime is not None and hasattr(runtime, "synchronize"):
            runtime.synchronize()
            return
        from mindspore import hal

        hal.synchronize()
    except (ImportError, AttributeError, RuntimeError):
        pass


def describe_device(info: dict[str, Any]) -> str:
    return (
        "MindSpore {} | target={} | id={} | mode={}".format(
            info["mindspore_version"],
            info["device_target"],
            info["device_id"],
            info["mode"],
        )
    )
