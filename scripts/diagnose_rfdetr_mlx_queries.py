"""Isolate low-margin encoder ranking drift; never patches production execution.

Compare native outputs, then force reference proposal order in a shadow forward.
A large raw difference can be caused by proposal rank swaps; selected parity
still needs an independent gate. Run alone on the GPU, with the original weights.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from jasna.mosaic.detection_registry import build_detection_model
from scripts.benchmark_rfdetr_apple import read_frames, raw_metrics, selected_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs=4, default=[300, 900, 1800, 2700])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import mlx.core as mx
    from jasna.mosaic.mlx_rfdetr import model as graph

    os.environ["JASNA_MPS_RFDETR_EXPORT"] = "0"

    def build(backend):
        os.environ["JASNA_APPLE_RFDETR_BACKEND"] = backend
        return build_detection_model(
            "rfdetr-v6",
            args.weights,
            batch_size=4,
            device=torch.device("mps"),
            score_threshold=0.35,
            fp16=False,
        )

    ref, native = build("torch"), build("mlx")
    frames = read_frames(args.input, args.frames)
    captured = {}

    def hook(module, inputs, out):
        # The checkpoint shares this classifier with decoder heads. Only the
        # Bx2304xC encoder output determines the two-stage proposal ordering.
        if out.ndim != 3 or out.shape[1] <= 200:
            return
        scores = out.max(-1).values
        captured["reference_indices"] = scores.topk(200, dim=1).indices.cpu().numpy()
        captured["reference_scores"] = scores.cpu().numpy()

    handle = ref.runner._core.transformer.enc_out_class_embed[0].register_forward_hook(
        hook
    )
    original = graph._topk_with_indices
    try:
        with torch.inference_mode():
            x = ref._preprocess(frames)
            r = ref._infer(x)
            handle.remove()

            def capture(values, k):
                scores, indices = original(values, k)
                mx.eval(values, indices)
                captured["mlx_indices"] = np.array(indices)
                captured["mlx_scores"] = np.array(values)
                return scores, indices

            with patch.object(graph, "_topk_with_indices", capture):
                a = native._infer(x)

            def reference_order(values, k):
                indices = mx.array(captured["reference_indices"])
                return mx.take_along_axis(values, indices, axis=1), indices

            with patch.object(graph, "_topk_with_indices", reference_order):
                forced = native._infer(x)
            report = {
                "original_raw": raw_metrics(a, r),
                "reference_order_forced_raw": raw_metrics(forced, r),
                "selected": selected_metrics(
                    native(frames, target_hw=frames.shape[-2:]),
                    ref(frames, target_hw=frames.shape[-2:]),
                ),
                "frames": [],
            }
            for i, frame in enumerate(args.frames):
                ri, mi = captured["reference_indices"][i], captured["mlx_indices"][i]
                mismatch = np.flatnonzero(ri != mi)
                scores = captured["reference_scores"][i]
                report["frames"].append(
                    {
                        "frame": frame,
                        "rank_mismatches": mismatch.tolist(),
                        "reference_indices": ri[mismatch].tolist(),
                        "mlx_indices": mi[mismatch].tolist(),
                        "reference_score_gap_between_swapped_indices": abs(
                            scores[ri[mismatch]] - scores[mi[mismatch]]
                        ).tolist(),
                        "encoder_score_max_abs": float(
                            abs(scores - captured["mlx_scores"][i]).max()
                        ),
                    }
                )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(report, indent=2))
    finally:
        handle.remove()
        native.close()
        ref.close()


if __name__ == "__main__":
    main()
