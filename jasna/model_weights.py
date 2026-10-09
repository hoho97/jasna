"""Portable checkpoint validation; never convert or write the source weights."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch


def validate_weights_path(path: str | Path, *, suffix: str, model: str) -> Path:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"{model} weights not found: {path}. Supply an explicit model path or JASNA_MODEL_WEIGHTS_DIR.")
    if path.suffix.lower() == ".enc":
        raise ValueError(f"{model}: encrypted weights are unsupported on Apple/MPS: {path}. Use a free .pt/.pth checkpoint.")
    if path.suffix.lower() != suffix:
        raise ValueError(f"{model} requires {suffix} checkpoint weights, got {path}. ONNX graphs and TensorRT engines cannot be loaded by PyTorch MPS.")
    return path


def _load_cpu(path: Path, *, weights_only: bool):
    try:
        return torch.load(path, map_location="cpu", weights_only=weights_only)
    except Exception as exc:
        raise ValueError(f"Cannot load checkpoint {path} on CPU: {exc}. Check the model format and dependency versions.") from exc


def load_rfdetr_checkpoint(path: str | Path) -> Mapping:
    path = validate_weights_path(path, suffix=".pt", model="RF-DETR")
    # RF-DETR checkpoints contain args as well as tensors; use trusted weights only.
    checkpoint = _load_cpu(path, weights_only=False)
    state = checkpoint.get("model") if isinstance(checkpoint, Mapping) else None
    weight = state.get("class_embed.weight") if isinstance(state, Mapping) else None
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.shape[0] < 2:
        raise ValueError(f"RF-DETR checkpoint {path} must contain model/class_embed.weight [classes + 1, hidden_dim]; a YOLO serialized model or plain state_dict is not interchangeable.")
    return checkpoint


def load_restoration_state_dict(path: str | Path) -> Mapping:
    path = validate_weights_path(path, suffix=".pth", model="BasicVSR++")
    checkpoint = _load_cpu(path, weights_only=True)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, Mapping) else None
    if not isinstance(state, Mapping) or not state or not all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError(f"BasicVSR++ checkpoint {path} must be a tensor state_dict (or contain state_dict).")
    return state
