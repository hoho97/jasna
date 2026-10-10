"""Apple detector selection; optional runtimes are never imported by this module."""

from __future__ import annotations

import os
import torch


def apple_rfdetr_backend(device: torch.device) -> str:
    # Foreign vendors ignore Apple-only configuration, including invalid values.
    if device.type != "mps":
        return "torch"
    backend = os.environ.get("JASNA_APPLE_RFDETR_BACKEND", "torch").strip().lower()
    if backend not in {"torch", "mlx", "coreml"}:
        raise ValueError("JASNA_APPLE_RFDETR_BACKEND must be torch, mlx or coreml")
    return backend
