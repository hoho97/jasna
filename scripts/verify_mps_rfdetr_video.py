"""Issue #8: real video frames -> MPS v6 detections -> annotated software video.

Diagnostic only: does not exercise Jasna's threaded decoder/encoder/restorer.
Run from the repo: python -m scripts.verify_mps_rfdetr_video --weights-dir DIR OUTPUT.mp4
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path
from time import perf_counter

import av
from PIL import Image, ImageDraw
import numpy as np
import torch

from jasna.mosaic.detection_registry import build_detection_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=Path("assets/test_clip1_1080p.mp4"))
    parser.add_argument("--start-frame", type=int, default=120)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    if args.frames < 1 or args.start_frame < 0:
        parser.error("--frames must be positive and --start-frame non-negative")
    assert torch.backends.mps.is_available(), "Real MPS required"
    model = build_detection_model(
        "rfdetr-v6", args.weights_dir / "rfdetr-v6.pt", batch_size=2,
        device=torch.device("mps"), score_threshold=.35, fp16=False,
    )
    images = []
    with av.open(str(args.input)) as source:
        for index, frame in enumerate(source.decode(video=0)):
            if index >= args.start_frame:
                images.append(frame.to_ndarray(format="rgb24"))
            if len(images) == args.frames:
                break
    assert len(images) == args.frames
    total, timings = 0, []
    try:
        with av.open(str(args.output), "w") as dest:
            stream = dest.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = 960, 540, "yuv420p"
            stream.options = {"crf": "18", "preset": "fast"}
            for start in range(0, len(images), 2):
                batch = torch.from_numpy(np.stack(images[start:start + 2])).permute(0, 3, 1, 2).to("mps")
                torch.mps.synchronize()
                timer = perf_counter()
                result = model(batch, target_hw=(540, 960))
                torch.mps.synchronize()
                timings.append(perf_counter() - timer)
                for offset, (boxes, masks) in enumerate(zip(result.boxes_xyxy, result.masks)):
                    assert masks.device.type == "mps" and masks.dtype == torch.bool
                    image = np.array(Image.fromarray(images[start + offset]).resize((960, 540), Image.Resampling.BILINEAR))
                    # CPU overlay/encode is a diagnostic rendering boundary.
                    if len(masks):
                        union = masks.any(0).cpu().numpy().astype(np.uint8)
                        union = np.asarray(Image.fromarray(union).resize((960, 540), Image.Resampling.NEAREST)).astype(bool)
                        image[union] = (image[union] * .6 + np.array([255, 32, 32]) * .4).astype(np.uint8)
                    canvas = Image.fromarray(image)
                    draw = ImageDraw.Draw(canvas)
                    for box in boxes:
                        x1, y1, x2, y2 = np.rint(box).astype(int)
                        draw.rectangle((x1, y1, x2, y2), outline=(32, 255, 32), width=2)
                    total += len(boxes)
                    output = av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")
                    output.pts, output.time_base = start + offset, Fraction(1, 30)
                    for packet in stream.encode(output):
                        dest.mux(packet)
            for packet in stream.encode(None):
                dest.mux(packet)
    finally:
        model.close()
    assert total > 0, "No positive detections; cannot claim positive detector verification"
    with av.open(str(args.output)) as source:
        decoded = list(source.decode(video=0))
        assert len(decoded) == args.frames
        assert all((f.width, f.height) == (960, 540) for f in decoded)
        assert all(a.pts < b.pts for a, b in zip(decoded, decoded[1:]))
    print(f"PASS frames={len(decoded)} detections={total} size=960x540 codec=H.264; decode/PTS verified")
    print(f"batch2_seconds={timings}; median_seconds={np.median(timings):.3f}")
    print("CPU: PyAV RGB decode, diagnostic overlay, libx264 encode. No restoration or audio in diagnostic output.")


if __name__ == "__main__":
    main()
