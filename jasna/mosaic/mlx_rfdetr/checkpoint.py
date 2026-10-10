"""Strict, CPU-testable v6 checkpoint mapping. No MLX import required."""

from __future__ import annotations

import json
import torch


def mapped_tensors(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {}
    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Non-tensor checkpoint entry: {key}")
        if key == "_kp_active_mask":
            if tensor.shape != (0, 0):
                raise ValueError("v6 segmentation requires empty _kp_active_mask")
            # MLX excludes underscore-prefixed attributes from parameters().
            key = "kp_active_mask"
        if tensor.ndim == 4:
            tensor = (
                tensor.permute(1, 2, 3, 0)
                if "stages_sampling" in key
                else tensor.permute(0, 2, 3, 1)
            )
        result[key] = tensor.detach().cpu().contiguous()
    return result


def validate_mapping(actual, expected) -> dict:
    report = {
        "missing_keys": sorted(set(expected) - set(actual)),
        "unexpected_keys": sorted(set(actual) - set(expected)),
        "shape_mismatches": {
            key: {
                "checkpoint": list(actual[key].shape),
                "model": list(expected[key].shape),
            }
            for key in sorted(set(actual) & set(expected))
            if actual[key].shape != expected[key].shape
        },
    }
    if any(report.values()):
        raise ValueError("Incompatible Jasna v6 checkpoint: " + json.dumps(report))
    return report
