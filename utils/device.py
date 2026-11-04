"""Device selection and backend helpers for CPU, CUDA and Ascend NPU."""

from __future__ import annotations

import logging
import os
from typing import Any, Iterable, Mapping

import numpy as np
import torch


_TORCH_NPU_IMPORT_ERROR: Exception | None = None

try:
    import torch_npu  # noqa: F401

    _HAS_TORCH_NPU = True
except Exception as exc:  # pragma: no cover - depends on the Ascend runtime
    torch_npu = None
    _HAS_TORCH_NPU = False
    _TORCH_NPU_IMPORT_ERROR = exc


def npu_available() -> bool:
    """Return whether torch_npu is installed, registered and usable."""
    if not _HAS_TORCH_NPU or not hasattr(torch, "npu"):
        return False
    try:
        return bool(torch.npu.is_available())
    except Exception:
        return False


def cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def _normalise_backend(backend: str | None) -> str:
    value = (backend or os.environ.get("QKD_BACKEND", "auto")).strip().lower()
    aliases = {
        "": "auto",
        "ascend": "npu",
        "cann": "npu",
        "gpu": "cuda",
    }
    value = aliases.get(value, value)
    if value not in {"auto", "npu", "cuda", "cpu"}:
        raise ValueError(
            f"Unsupported backend {backend!r}. Use auto, npu, cuda or cpu."
        )
    return value


def _auto_backend() -> str:
    if npu_available():
        return "npu"
    if cuda_available():
        return "cuda"
    return "cpu"


def _check_backend(backend: str) -> None:
    if backend == "npu" and not npu_available():
        detail = ""
        if _TORCH_NPU_IMPORT_ERROR is not None:
            detail = f" Import error: {_TORCH_NPU_IMPORT_ERROR}"
        raise RuntimeError(
            "Ascend NPU was requested, but torch_npu is unavailable. "
            "Source the CANN environment and install a torch_npu wheel that "
            "matches the installed PyTorch version." + detail
        )
    if backend == "cuda" and not cuda_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")


def resolve_device(
    spec: str | int | torch.device | None = None,
    *,
    backend: str | None = None,
) -> torch.device:
    """Resolve a device spec.

    Numeric specs (``0`` or ``"0"``) follow ``backend``/``QKD_BACKEND``.
    ``auto`` prefers NPU, then CUDA, then CPU. Explicit ``npu``/``cuda``
    requests fail loudly instead of silently falling back to CPU.
    """
    if isinstance(spec, torch.device):
        if spec.type == "npu":
            _check_backend("npu")
        elif spec.type == "cuda":
            _check_backend("cuda")
        return spec

    selected_backend = _normalise_backend(backend)
    raw = "auto" if spec is None else str(spec).strip().lower()
    if raw in {"-1", "cpu"}:
        return torch.device("cpu")
    if raw in {"auto", ""}:
        selected_backend = _auto_backend()
        _check_backend(selected_backend)
        return torch.device("cpu") if selected_backend == "cpu" else torch.device(selected_backend, 0)
    if raw.startswith("npu"):
        _check_backend("npu")
        return torch.device(raw if ":" in raw else "npu:0")
    if raw.startswith("cuda"):
        _check_backend("cuda")
        return torch.device(raw if ":" in raw else "cuda:0")
    if raw.startswith("cpu:"):
        return torch.device("cpu")

    if raw.isdigit():
        if selected_backend == "auto":
            selected_backend = _auto_backend()
        _check_backend(selected_backend)
        return torch.device(selected_backend, int(raw))

    raise ValueError(
        f"Unsupported device spec {spec!r}. Examples: cpu, npu:0, cuda:0, 0."
    )


def resolve_devices(
    specs: str | int | torch.device | Iterable[str | int | torch.device] | None,
    *,
    backend: str | None = None,
) -> list[torch.device]:
    if specs is None:
        specs = ["auto"]
    elif isinstance(specs, (str, int, torch.device)):
        specs = [specs]
    devices = [resolve_device(spec, backend=backend) for spec in specs]
    if not devices:
        raise ValueError("At least one device must be specified.")
    return devices


def resolve_quantum_device(
    value: str | None,
    reference_device: torch.device,
    *,
    prefer_cpu_on_npu: bool = True,
) -> torch.device:
    """Pick the device used by the small complex-valued quantum simulator.

    ``auto`` uses CPU on Ascend because complex64 coverage differs between
    CANN versions. The circuit is tiny, so CPU simulation is inexpensive.
    """
    raw = (value or os.environ.get("QKD_QUANTUM_DEVICE", "auto")).strip().lower()
    if raw in {"device", "same"}:
        return reference_device
    if raw in {"auto", ""}:
        if reference_device.type == "npu" and prefer_cpu_on_npu:
            return torch.device("cpu")
        return reference_device
    return resolve_device(raw)


def seed_all(seed: int) -> None:
    """Seed every active backend without invoking unavailable CUDA APIs."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if cuda_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if npu_available():
        torch.npu.manual_seed_all(seed)


def set_active_device(device: torch.device) -> None:
    """Make an indexed accelerator the current device for this process."""
    if device.type == "npu" and npu_available():
        torch.npu.set_device(device.index if device.index is not None else 0)
    elif device.type == "cuda" and cuda_available():
        torch.cuda.set_device(device.index if device.index is not None else 0)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda" and cuda_available():
        torch.cuda.synchronize(device)
    elif device.type == "npu" and npu_available():
        torch.npu.synchronize()


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda" and cuda_available():
        torch.cuda.empty_cache()
    elif device.type == "npu" and npu_available():
        torch.npu.empty_cache()


def device_name(device: torch.device) -> str:
    try:
        if device.type == "npu" and npu_available():
            index = device.index if device.index is not None else torch.npu.current_device()
            return str(torch.npu.get_device_name(index))
        if device.type == "cuda" and cuda_available():
            index = device.index if device.index is not None else torch.cuda.current_device()
            return str(torch.cuda.get_device_name(index))
    except Exception as exc:
        logging.debug("Could not query device name for %s: %s", device, exc)
    return "CPU"


def describe_devices(devices: Iterable[torch.device]) -> str:
    return ", ".join(f"{device} ({device_name(device)})" for device in devices)


def get_num_workers(
    args: Mapping[str, Any] | None,
    *,
    default: int = 4,
) -> int:
    args = args or {}
    value = args.get("num_workers")
    if value is None:
        value = os.environ.get("QKD_NUM_WORKERS", default)
    return max(0, int(value))


def loader_kwargs(
    args: Mapping[str, Any] | None,
    device: torch.device,
) -> dict[str, Any]:
    """Return DataLoader options that are safe for the active accelerator."""
    args = args or {}
    workers = get_num_workers(args, default=0 if device.type == "npu" else 4)
    options: dict[str, Any] = {
        "num_workers": workers,
        # pin_memory is a CUDA optimisation and is unnecessary on Ascend.
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        options["persistent_workers"] = bool(args.get("persistent_workers", False))
        if "prefetch_factor" in args:
            options["prefetch_factor"] = int(args["prefetch_factor"])
    return options
