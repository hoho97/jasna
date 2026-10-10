"""Audit conversion: v6 .pt -> scoped MLX safetensors plus empty-buffer metadata.

The production runner CPU-loads the original checkpoint; this artifact is for
reproducibility. MLX safetensors cannot serialize empty keypoint buffers.
"""

from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from jasna.model_weights import load_rfdetr_checkpoint
from jasna.mosaic.mlx_rfdetr.checkpoint import mapped_tensors, validate_mapping
from jasna.mosaic.mlx_rfdetr.config import JasnaV6MLXConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.weights.resolve().parent):
        raise ValueError("Output must be outside the readonly checkpoint directory")
    if args.output.exists():
        raise FileExistsError(args.output)
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from jasna.mosaic.mlx_rfdetr.model import RFDETRForInference

    state = load_rfdetr_checkpoint(args.weights)["model"]
    weights = mapped_tensors(state)
    model = RFDETRForInference(JasnaV6MLXConfig)
    model.kp_active_mask = mx.zeros((0, 0))
    report = validate_mapping(weights, dict(tree_flatten(model.parameters())))
    converted = {k: mx.array(v.numpy()) for k, v in weights.items()}
    model.load_weights(list(converted.items()), strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(args.output), {k: v for k, v in converted.items() if k != "kp_active_mask"}
    )
    restored = mx.load(str(args.output))
    restored["kp_active_mask"] = mx.zeros((0, 0))
    model.load_weights(list(restored.items()), strict=True)
    with args.weights.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    args.output.with_suffix(".json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": digest,
                "config": asdict(JasnaV6MLXConfig),
                "strict_mapping": report,
                "checkpoint_tensors": len(weights),
                "serialized_tensors": len(restored) - 1,
                "reconstructed_empty_buffers": {"kp_active_mask": [0, 0]},
                "mlx": mx.__version__,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
