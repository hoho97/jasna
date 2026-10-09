"""Opt-in real FP32 temporal restoration; never skip unavailable MPS after opt-in.

JASNA_TEST_MODEL_WEIGHTS_DIR selects trusted read-only release weights.
CPU parity tolerance is fixed before measurement: atol=rtol=1e-3 in [0,1] scale.
"""
import os
from pathlib import Path
from time import perf_counter
from unittest.mock import Mock

import av
import numpy as np
import pytest
import torch
import torchvision

from jasna.crop_buffer import RawCrop
from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
from jasna.restorer.restoration_pipeline import RestorationPipeline
from jasna.tracking.clip_tracker import TrackedClip


@pytest.fixture(scope="module")
def real_inputs():
    weights = os.environ.get("JASNA_TEST_MODEL_WEIGHTS_DIR")
    if not weights:
        pytest.skip("set JASNA_TEST_MODEL_WEIGHTS_DIR for real MPS restoration")
    assert torch.backends.mps.is_built() and torch.backends.mps.is_available()
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1"
    images = []
    source = Path(__file__).resolve().parents[1] / "assets/test_clip1_1080p.mp4"
    with av.open(str(source)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if 120 <= index <= 122:
                rgb = frame.to_ndarray(format="rgb24")
                # Varying crop heights exercise resize plus reflection padding.
                images.append(torch.from_numpy(rgb[400:(620, 600, 610)[index - 120], 800:1100].copy()).permute(2, 0, 1))
            if len(images) == 3:
                break
    assert len(images) == 3 and not torch.equal(images[0][:, :200], images[1])
    return Path(weights) / "lada_mosaic_restoration_model_generic_v1.2.pth", images


@pytest.mark.parametrize("length", [2, 3])
def test_real_temporal_restoration_matches_cpu(real_inputs, monkeypatch, length):
    path, images = real_inputs
    forbidden = Mock(side_effect=AssertionError("CUDA/TensorRT reached on MPS"))
    monkeypatch.setattr(torch.cuda, "current_stream", forbidden)
    monkeypatch.setattr(torch.cuda, "set_device", forbidden)
    monkeypatch.setattr("jasna.media.cuda_kernel.cuda_driver", forbidden)
    restorer = BasicvsrppMosaicRestorer(str(path), torch.device("mps"), length,
                                      use_tensorrt=True, fp16=True)
    try:
        assert restorer.input_dtype == torch.float32 and restorer._split_forward is None
        tensors = list(restorer.model.parameters()) + list(restorer.model.buffers())
        assert tensors and all(t.device.type == "mps" for t in tensors)
        assert all(t.dtype == torch.float32 for t in tensors if t.is_floating_point())
        alignment_calls = []
        original = torchvision.ops.deform_conv2d
        def deform(x, offset, weight, bias, stride, padding, dilation, mask):
            alignment_calls.append((x.device.type, tuple(x.shape), tuple(offset.shape), tuple(mask.shape)))
            assert all(t.device == x.device for t in (offset, weight, bias, mask))
            return original(x, offset, weight, bias, stride, padding, dilation, mask)
        monkeypatch.setattr(torchvision.ops, "deform_conv2d", deform)
        pipeline = RestorationPipeline(restorer)
        clip = TrackedClip(track_id=9, start_frame=120, mask_resolution=(8, 8),
                           bboxes=[np.array([800, 400, 1100, 400 + image.shape[1]], dtype=np.float32) for image in images[:length]],
                           masks=[torch.ones(8, 8, dtype=torch.bool, device="mps")] * length)
        crops = [RawCrop(image.clone(), (800, 400, 1100, 400 + image.shape[1]), (image.shape[1], 300)) for image in images[:length]]
        torch.mps.synchronize()
        start = perf_counter()
        primary = pipeline.prepare_and_run_primary(clip, crops, (1080, 1920), 0, length, None)
        torch.mps.synchronize()
        mps_seconds = perf_counter() - start
        actual = primary.primary_raw
        assert actual.shape == (length, 3, 256, 256)
        assert actual.dtype == torch.float32 and actual.device.type == "mps"
        assert torch.isfinite(actual).all().item()
        prepared = pipeline._prepare_from_raw_crops(crops)[0]
        change = (actual - torch.stack(prepared).div(255)).abs().mean().item()
        assert change > 1e-4, "real restoration must change the input"
        assert len(alignment_calls) == 4 * (length - 1)
        assert all(device == "mps" and shape == (1, 128, 64, 64)
                   and offset == (1, 288, 64, 64) and mask == (1, 144, 64, 64)
                   for device, shape, offset, mask in alignment_calls)
        assert primary.resize_shapes == [(256, 256), (232, 256), (244, 256)][:length]
        assert primary.pad_offsets == [(0, 0), (0, 12), (0, 6)][:length]
        frames = pipeline._run_secondary(actual, 0, length)
        assert len(frames) == length
        assert all(f.shape == (3, 256, 256) and f.dtype == torch.uint8 and f.device.type == "mps" for f in frames)
        # Explicit test-only CPU baseline; production model never falls back.
        cpu_crops = [RawCrop(image.clone(), (800, 400, 1100, 400 + image.shape[1]), (image.shape[1], 300)) for image in images[:length]]
        reference = BasicvsrppMosaicRestorer(str(path), torch.device("cpu"), length, False, False)
        try:
            start = perf_counter()
            expected = RestorationPipeline(reference).prepare_and_run_primary(clip, cpu_crops, (1080, 1920), 0, length, None).primary_raw
            cpu_seconds = perf_counter() - start
        finally:
            reference.close()
        error = (actual.cpu() - expected).abs()
        torch.testing.assert_close(actual.cpu(), expected, atol=1e-3, rtol=1e-3)
        assert all(t.device.type == "mps" for t in tensors)
        forbidden.assert_not_called()
        print(f"\nT={length}: MPS={mps_seconds:.3f}s CPU={cpu_seconds:.3f}s max_abs={error.max().item():.9f} RMSE={error.square().mean().sqrt().item():.9f} mean_change={change:.6f} range=({actual.min().item():.6f},{actual.max().item():.6f}) MPS_allocated={torch.mps.current_allocated_memory()} driver={torch.mps.driver_allocated_memory()}")
        output_dir = os.environ.get("JASNA_TEST_VIDEO_OUTPUT_DIR")
        if output_dir:
            output = Path(output_dir) / f"basicvsrpp-t{length}.mp4"
            with av.open(str(output), "w") as container:
                stream = container.add_stream("libx264", rate=30)
                stream.width = stream.height = 256
                stream.pix_fmt = "yuv420p"
                for image in frames:
                    frame = av.VideoFrame.from_ndarray(image.cpu().permute(1, 2, 0).contiguous().numpy(), format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            with av.open(str(output)) as container:
                decoded = list(container.decode(video=0))
                assert len(decoded) == length
                assert all(f.width == f.height == 256 for f in decoded)
                assert all(a.pts < b.pts for a, b in zip(decoded, decoded[1:]))
    finally:
        restorer.close()
        torch.mps.empty_cache()


def test_direct_loader_keeps_mps_fp32(real_inputs):
    from jasna.models.basicvsrpp.inference import load_model
    model = load_model(None, str(real_inputs[0]), "mps", True)
    tensors = list(model.parameters()) + list(model.buffers())
    assert all(t.device.type == "mps" for t in tensors)
    assert all(t.dtype == torch.float32 for t in tensors if t.is_floating_point())
    del model
    torch.mps.empty_cache()


def test_native_production_deform_op_matches_cpu(real_inputs):
    # Production uses C_in=128, C_out=64, 16 offset groups at 64x64 feature size.
    assert torch._C._dispatch_has_kernel_for_dispatch_key("torchvision::deform_conv2d", "MPS")
    generator = torch.Generator().manual_seed(9)
    x = torch.randn(1, 128, 64, 64, generator=generator)
    offset = torch.randn(1, 288, 64, 64, generator=generator) * .25
    mask = torch.randn(1, 144, 64, 64, generator=generator).sigmoid()
    weight = torch.randn(64, 128, 3, 3, generator=generator) * .01
    bias = torch.randn(64, generator=generator) * .01
    with torch.inference_mode():
        expected = torchvision.ops.deform_conv2d(x, offset, weight, bias, stride=1, padding=1, dilation=1, mask=mask)
        actual = torchvision.ops.deform_conv2d(x.to("mps"), offset.to("mps"), weight.to("mps"), bias.to("mps"), stride=1, padding=1, dilation=1, mask=mask.to("mps"))
    assert actual.device.type == "mps" and actual.dtype == torch.float32
    assert actual.shape == (1, 64, 64, 64) and torch.isfinite(actual).all().item()
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("padding_mode", ["zeros", "border"])
def test_native_flow_warp_matches_cpu(real_inputs, padding_mode):
    from jasna.models.basicvsrpp.mmagic.flow_warp import flow_warp
    generator = torch.Generator().manual_seed(9)
    x = torch.randn(1, 64, 64, 64, generator=generator)
    flow = torch.randn(1, 64, 64, 2, generator=generator) * 2
    actual = flow_warp(x.to("mps"), flow.to("mps"), padding_mode=padding_mode)
    expected = flow_warp(x, flow, padding_mode=padding_mode)
    assert actual.device.type == "mps" and torch.isfinite(actual).all().item()
    torch.testing.assert_close(actual.cpu(), expected, atol=1e-4, rtol=1e-4)
