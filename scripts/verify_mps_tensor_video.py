"""Issue #7 smoke: PyAV decode -> MPS color math -> host software encode.

This deliberately tests converters, not VideoReader/VideoEncoder integration
(issues #10/#11). No detection/restoration model or weights are used.
Run from the repository with: python -m scripts.verify_mps_tensor_video OUTPUT.mp4
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path
from time import perf_counter

import av
import numpy as np
import torch
from av.video.reformatter import Colorspace, VideoReformatter

from jasna.media.rgb_to_yuv import RgbToYuvConverter
from jasna.media.yuv_to_rgb import YuvToRgbConverter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--input", type=Path, default=Path("assets/test_clip1_1080p.mp4"))
    parser.add_argument("--frames", type=int, default=24)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Real MPS is required")
    # Fail if these real converters accidentally reach a CUDA boundary.
    from unittest.mock import patch
    with patch("jasna.media.cuda_kernel.cuda_driver", side_effect=AssertionError("CUDA driver on MPS")):
        run(args)


def run(args):
    device = torch.device("mps")
    h, w = 180, 320
    decoder = YuvToRgbConverter(h, w, Colorspace.ITU709, False, False, device)
    encoder = RgbToYuvConverter("nv12_bt709_limited", device=device)
    reformatter = VideoReformatter()
    count, max_error, elapsed = 0, 0, []
    with av.open(str(args.input)) as source, av.open(str(args.output), mode="w") as dest:
        stream = dest.add_stream("libx264", rate=24)
        stream.width, stream.height = w, h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "preset": "fast"}
        stream.codec_context.colorspace = 1
        stream.codec_context.color_range = 1
        for index, frame in enumerate(source.decode(video=0)):
            nv12 = reformatter.reformat(frame, width=w, height=h, format="nv12",
                                        dst_colorspace=Colorspace.ITU709, dst_color_range=1)
            y_plane, uv_plane = nv12.planes
            y = torch.frombuffer(y_plane, dtype=torch.uint8).reshape(h, y_plane.line_size)[:, :w]
            uv = torch.frombuffer(uv_plane, dtype=torch.uint8).reshape(h // 2, uv_plane.line_size)[:, :w].unflatten(1, (w // 2, 2))
            torch.mps.synchronize()
            start = perf_counter()
            rgb = decoder.convert(y.to(device), uv.to(device))
            packed = encoder.convert(rgb)
            host = packed.cpu()  # Explicit software I/O boundary, no CPU tensor-op fallback.
            elapsed.append(perf_counter() - start)
            assert rgb.device.type == "mps" and packed.device.type == "mps"
            assert rgb.dtype == packed.dtype == torch.uint8
            assert packed.shape == (h * 3 // 2, w)
            reference = YuvToRgbConverter(h, w, Colorspace.ITU709, False, False, torch.device("cpu")).convert(y, uv)
            error = (rgb.cpu().float() - reference.float()).abs().max().item()
            assert error <= 1, error
            max_error = max(max_error, error)
            output = av.VideoFrame.from_ndarray(host.numpy(), format="nv12")
            output.pts, output.time_base = index, Fraction(1, 24)
            for packet in stream.encode(output):
                dest.mux(packet)
            count += 1
            if count == args.frames:
                break
        for packet in stream.encode(None):
            dest.mux(packet)
    assert count == args.frames, (count, args.frames)
    with av.open(str(args.output)) as result:
        decoded = list(result.decode(video=0))
        assert len(decoded) == count
        assert all((f.width, f.height) == (w, h) for f in decoded)
        assert all(a.pts < b.pts for a, b in zip(decoded, decoded[1:]))
    print(f"PASS frames={count} size={w}x{h} max_rgb_cpu_error={max_error}")
    print(f"MPS conversions + transfers median_ms={np.median(elapsed) * 1000:.3f}; first_ms={elapsed[0] * 1000:.3f}")
    print("No weights; no audio in this converter-only smoke; not a full restoration pipeline test.")


if __name__ == "__main__":
    main()
