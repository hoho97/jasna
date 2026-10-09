from __future__ import annotations

from collections.abc import MutableMapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from enum import StrEnum
import os
from typing import Any

import torch

# NORMAL benchmarks every unseen convolution problem. BasicVSR++ has fixed
# spatial dimensions but a variable temporal clip length (and therefore variable
# effective convolution batches), so FAST avoids repeated runtime profiling while
# still using MIOpen's system/user performance databases. Users can override this.
#
# Expandable segments release memory by unmapping virtual address ranges rather
# than by a device-synchronizing free, so a VramOffloader empty_cache() could
# pull pages out from under kernels another thread still had in flight — issue
# #252 caught a restorer fp16 GEMM faulting with "Page not present". The env
# names cover the versions in the field; whichever one the build reads wins.
def apply_rocm_env_defaults(environ: MutableMapping[str, str]) -> None:
    environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:False")
    environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "expandable_segments:False")
    environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")


def configure_rocm_process_env() -> None:
    """Apply the ROCm defaults to this process; call at entry points before GPU work."""
    if torch.version.hip:
        apply_rocm_env_defaults(os.environ)


class AcceleratorVendor(StrEnum):
    NVIDIA = "nvidia"
    AMD = "amd"
    APPLE = "apple"
    CPU = "cpu"


@dataclass(frozen=True)
class AcceleratorCapabilities:
    """Execution primitives and supported application features for each backend."""

    streams: bool
    events: bool
    ipc_collect: bool
    mem_get_info: bool
    tensorrt: bool = False
    secondary_restoration: bool = False
    ltx: bool = False
    advanced_video: bool = False


_CUDA_LIKE_CAPABILITIES = AcceleratorCapabilities(
    streams=True,
    events=True,
    ipc_collect=True,
    mem_get_info=True,
    tensorrt=True,
    secondary_restoration=True,
    ltx=True,
    advanced_video=True,
)
_MPS_CAPABILITIES = AcceleratorCapabilities(
    streams=False,
    events=False,
    ipc_collect=False,
    mem_get_info=False,
)
_CPU_CAPABILITIES = AcceleratorCapabilities(
    streams=False,
    events=False,
    ipc_collect=False,
    mem_get_info=False,
)


def _mps_backend():
    return getattr(torch.backends, "mps", None)


def _require_mps_available() -> None:
    backend = _mps_backend()
    if backend is None or not backend.is_built():
        raise RuntimeError(
            "MPS device requested, but this PyTorch build was not built with MPS support"
        )
    if not backend.is_available():
        raise RuntimeError(
            "MPS device requested, but MPS is not available on this macOS/device configuration"
        )


def vendor_for_device(device: torch.device | str | None = None) -> AcceleratorVendor:
    resolved = torch.device(device) if device is not None else None
    if resolved is not None:
        if resolved.type == "cpu":
            return AcceleratorVendor.CPU
        if resolved.type == "mps":
            _require_mps_available()
            return AcceleratorVendor.APPLE

    if torch.version.hip:
        return AcceleratorVendor.AMD
    if torch.version.cuda:
        return AcceleratorVendor.NVIDIA

    backend = _mps_backend()
    if resolved is None and backend is not None and backend.is_built() and backend.is_available():
        return AcceleratorVendor.APPLE
    return AcceleratorVendor.CPU


def capabilities_for_device(
    device: torch.device | str | None = None,
) -> AcceleratorCapabilities:
    vendor = vendor_for_device(device)
    if vendor is AcceleratorVendor.NVIDIA:
        return _CUDA_LIKE_CAPABILITIES
    if vendor is AcceleratorVendor.AMD:
        return replace(_CUDA_LIKE_CAPABILITIES, tensorrt=False, secondary_restoration=False)
    if vendor is AcceleratorVendor.APPLE:
        return _MPS_CAPABILITIES
    return _CPU_CAPABILITIES


def is_nvidia_device(device: torch.device | str | None = None) -> bool:
    return vendor_for_device(device) is AcceleratorVendor.NVIDIA


def is_amd_device(device: torch.device | str | None = None) -> bool:
    return vendor_for_device(device) is AcceleratorVendor.AMD


def is_apple_device(device: torch.device | str | None = None) -> bool:
    return vendor_for_device(device) is AcceleratorVendor.APPLE


def device_module(device: torch.device | str):
    return torch.get_device_module(torch.device(device))


def device_context(device: torch.device | str):
    resolved = torch.device(device)
    if resolved.type in {"cpu", "mps"}:
        if resolved.type == "mps":
            _require_mps_available()
        return nullcontext()
    return device_module(resolved).device(resolved)


def stream_context(stream: Any):
    if stream is None:
        return nullcontext()
    try:
        return device_module(stream.device).stream(stream)
    except (TypeError, ValueError):
        # Also supports lightweight CUDA/ROCm stream doubles used by callers/tests.
        return torch.cuda.stream(stream)


def new_stream(device: torch.device | str):
    resolved = torch.device(device)
    if resolved.type == "mps":
        _require_mps_available()
        return None
    if resolved.type == "cpu":
        return None
    return device_module(resolved).Stream(resolved)


def current_stream(device: torch.device | str):
    resolved = torch.device(device)
    if resolved.type == "mps":
        _require_mps_available()
        return None
    if resolved.type == "cpu":
        return None
    return device_module(resolved).current_stream(resolved)


def new_event(device: torch.device | str):
    resolved = torch.device(device)
    if resolved.type == "mps":
        _require_mps_available()
        return None
    if resolved.type == "cpu":
        return None
    return device_module(resolved).Event()


def set_device(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if resolved.type == "mps":
        _require_mps_available()
        return
    if resolved.type != "cpu":
        device_module(resolved).set_device(resolved)


def synchronize(device: torch.device | str | None = None) -> None:
    if device is None:
        torch.accelerator.synchronize()
        return
    resolved = torch.device(device)
    if resolved.type == "mps":
        _require_mps_available()
        torch.mps.synchronize()
    elif resolved.type != "cpu":
        device_module(resolved).synchronize(resolved)


def empty_cache(device: torch.device | str | None = None) -> None:
    if device is not None:
        resolved = torch.device(device)
        if resolved.type == "mps":
            _require_mps_available()
            torch.mps.empty_cache()
            return
        if resolved.type == "cpu":
            return

    if hasattr(torch, "accelerator") and torch.accelerator.is_available():
        torch.accelerator.empty_cache()
        return
    if device is not None:
        module = device_module(torch.device(device))
        if hasattr(module, "empty_cache"):
            module.empty_cache()


def ipc_collect(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if resolved.type == "mps":
        _require_mps_available()
        return
    if resolved.type == "cpu":
        return
    module = device_module(resolved)
    if hasattr(module, "ipc_collect"):
        module.ipc_collect()


def reset_peak_memory_stats(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if resolved.type == "mps":
        _require_mps_available()
        return
    if resolved.type == "cpu":
        return
    module = device_module(resolved)
    if hasattr(module, "reset_peak_memory_stats"):
        module.reset_peak_memory_stats(resolved)


def mem_get_info(device: torch.device | str) -> tuple[int, int]:
    resolved = torch.device(device)
    if resolved.type == "mps":
        _require_mps_available()
        raise NotImplementedError(
            "MPS uses unified memory and does not expose CUDA-style free/total device memory"
        )
    if resolved.type == "cpu":
        raise NotImplementedError("CPU does not expose accelerator free/total device memory")
    module = device_module(resolved)
    return module.mem_get_info(resolved)


def device_name(device: torch.device | str) -> str:
    resolved = torch.device(device)
    if resolved.type == "cpu":
        return "CPU"
    if resolved.type == "mps":
        _require_mps_available()
        backend = _mps_backend()
        get_name = getattr(backend, "get_name", None)
        return str(get_name()) if get_name is not None else "Apple MPS"
    return str(device_module(resolved).get_device_name(resolved))
