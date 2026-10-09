"""Shared feature gates, evaluated before loading optional backend modules."""
from __future__ import annotations

from jasna.accelerator import capabilities_for_device


def validate_backend_options(
    device,
    *,
    secondary_restoration: str = "none",
    restoration_model_name: str = "basicvsrpp",
    ltx_fast: bool = False,
    advanced_video: bool = False,
    decode_backend: str = "auto",
) -> None:
    caps = capabilities_for_device(device)
    if secondary_restoration != "none" and not caps.secondary_restoration:
        raise ValueError(
            f"Secondary restoration '{secondary_restoration}' is not supported on {device}. "
            "Use secondary restoration 'none'. RTX/UNet require NVIDIA; "
            "TVAI uses an external Topaz FFmpeg and is not validated on this backend."
        )
    if (restoration_model_name.startswith("ltx") or ltx_fast) and not caps.ltx:
        raise ValueError(f"LTX restoration (including fast/trial) is not supported on {device} yet.")
    if advanced_video and not caps.advanced_video:
        raise ValueError(f"VR, streaming, smart rendering and benchmarks are not supported on {device} yet.")
    if decode_backend in {"vali", "pyav-hw"} and not caps.advanced_video:
        raise ValueError(
            f"Hardware decode '{decode_backend}' is not supported on {device}. "
            "Use JASNA_DECODE_BACKEND=auto or pyav-sw; VALI/NVDEC/AMF require NVIDIA/AMD."
        )
