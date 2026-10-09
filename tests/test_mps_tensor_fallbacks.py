"""Real MPS numerical/contract checks; no global PyTorch CPU fallback required."""
from types import SimpleNamespace

import pytest
import torch
from av.video.reformatter import Colorspace

from jasna.media.cas import GpuCasSharpener
from jasna.media.lut import CubeLut, GpuLutApplier
from jasna.media.resize_normalize import ResizeNormalizer
from jasna.media.rgb_to_yuv import RgbToYuvConverter
from jasna.media.yuv_to_rgb import YuvToRgbConverter
from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel
from jasna.mosaic.yolo import YoloMosaicDetectionModel
from jasna.restorer.denoise import spatial_denoise

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires real MPS")
MPS = torch.device("mps")
CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("MPS fallback attempted CUDA driver/kernel/stream access")
    monkeypatch.setattr("jasna.media.cuda_kernel.cuda_driver", forbidden)
    monkeypatch.setattr("jasna.media.cuda_kernel.Kernel.launch", forbidden)
    monkeypatch.setattr(torch.cuda, "current_stream", forbidden)


def _random(shape, dtype=torch.uint8):
    return torch.randint(0, 256, shape, generator=torch.Generator().manual_seed(7), dtype=dtype)


def _close(actual, reference, tolerance):
    assert actual.device.type == "mps"
    assert actual.dtype == reference.dtype
    assert actual.shape == reference.shape
    assert torch.isfinite(actual.float()).all().item()
    assert (actual.cpu().float() - reference.float()).abs().max().item() <= tolerance


@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 2e-5), (torch.float16, 0.008)])
@pytest.mark.parametrize("detector", [RfDetrMosaicDetectionModel, YoloMosaicDetectionModel])
def test_detector_resize_normalize_caller_fallback(detector, dtype, tolerance):
    # Odd source size and non-contiguous rows, both shrink and letterbox.
    frames = _random((2, 3, 35, 106))
    def run(device):
        normalizer = ResizeNormalizer(device=device, dtype=dtype, mean=(0.,) * 3,
                                      std=(1.,) * 3, fill=(0.,) * 3)
        assert not normalizer.available
        obj = SimpleNamespace(device=device, input_dtype=dtype, resolution=32,
                              imgsz=64, stride=32, _resizer=None, _normalization_cache={})
        obj._normalization = lambda x: detector._normalization(obj, x)
        result = detector._preprocess(obj, frames.to(device)[:, :, :, ::2])
        return result[0] if isinstance(result, tuple) else result
    _close(run(MPS), run(CPU), tolerance)


@pytest.mark.parametrize("space,name", [(Colorspace.ITU601, "bt601"),
                                         (Colorspace.ITU709, "bt709"),
                                         (Colorspace.BT2020, "bt2020")])
@pytest.mark.parametrize("full_range", [False, True])
@pytest.mark.parametrize("ten_bit", [False, True])
def test_color_conversion_pitched_planes(space, name, full_range, ten_bit):
    height, width = 18, 26
    frame_storage = _random((3, height, width + 6))
    frame = frame_storage[:, :, :width]
    variant = f"{'p010' if ten_bit else 'nv12'}_{name}_{'full' if full_range else 'limited'}"
    cpu_converter = RgbToYuvConverter(variant, device=CPU)
    mps_converter = RgbToYuvConverter(variant, device=MPS)
    assert not mps_converter.uses_kernel
    reference = cpu_converter.convert(frame)
    pitched = torch.empty((height * 3 // 2, width + 6), dtype=reference.dtype, device=MPS)
    output = pitched[:, :width]
    mps_frame = frame_storage.to(MPS)[:, :, :width]
    assert not mps_frame.is_contiguous()
    mps_converter.convert_into(mps_frame, output[:height], output[height:])
    # int16 carries unsigned P010 bits; compare in unsigned code units.
    scale = 64 if ten_bit else 1
    a = output.cpu().to(torch.int32) & 0xFFFF
    b = reference.to(torch.int32) & 0xFFFF
    assert (a - b).abs().max().item() <= scale
    if ten_bit:
        assert ((a & 63) == 0).all()
    # Decoder receives uint16 P010, encoder uses signed int16 storage.
    planes = output.view(torch.uint16) if ten_bit else output
    cpu_planes = reference.view(torch.uint16) if ten_bit else reference
    mps_yuv = YuvToRgbConverter(height, width, space, full_range, ten_bit, MPS)
    cpu_yuv = YuvToRgbConverter(height, width, space, full_range, ten_bit, CPU)
    got = mps_yuv.convert(planes[:height], planes[height:].unflatten(1, (width // 2, 2)))
    expected = cpu_yuv.convert(cpu_planes[:height], cpu_planes[height:].unflatten(1, (width // 2, 2)))
    _close(got, expected, 1)


@pytest.mark.parametrize("height,width", [(3, 4), (4, 3), (5, 5)])
def test_420_odd_dimensions_rejected(height, width):
    with pytest.raises(ValueError, match="even dimensions"):
        RgbToYuvConverter("nv12_bt709_limited", device=MPS).convert(torch.zeros((3, height, width), dtype=torch.uint8, device=MPS))
    with pytest.raises(ValueError, match="even dimensions"):
        YuvToRgbConverter(height, width, Colorspace.ITU709, False, False, MPS)


@pytest.mark.parametrize("ten_bit", [False, True])
@pytest.mark.parametrize("strength", [0.1, 0.5, 1.0])
def test_cas_pitched_luma_preserves_chroma(ten_bit, strength):
    packed = _random((27, 32))[:, :26]
    if ten_bit:
        packed = (packed.to(torch.int32) * 4 * 64).to(torch.int16)
    reference = packed.clone()
    storage = torch.empty((27, 32), dtype=packed.dtype, device=MPS)
    actual = storage[:, :26]
    actual.copy_(packed)
    assert not actual.is_contiguous()
    before = actual[18:].cpu().clone()
    for device, target in [(CPU, reference), (MPS, actual)]:
        sharpener = GpuCasSharpener(strength, ten_bit=ten_bit, device=device)
        sharpener.apply_luma_(target, 18)
        # Also exercise encoder's separate luma output contract.
        separate = torch.empty_like(target[:18])
        sharpener.sharpen_into(packed[:18].to(device), separate)
        assert torch.equal(separate.cpu(), target[:18].cpu())
    _close(actual, reference, 64 if ten_bit else 1)
    assert torch.equal(actual[18:].cpu(), before)


@pytest.mark.parametrize("is_3d", [False, True])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.float32])
def test_nonidentity_lut_with_domain_and_strides(is_3d, dtype):
    n = 9
    axis = torch.linspace(0, 1, n)
    if is_3d:
        b, g, r = torch.meshgrid(axis, axis, axis, indexing="ij")
        data = torch.stack((g * r, b.square(), 1 - r))
    else:
        data = torch.stack((axis.square(), 1 - axis, axis * 0.7), dim=1)
    lut = CubeLut(n, is_3d, data, (0.1, -0.1, 0.2), (0.9, 0.8, 1.1))
    frame = _random((3, 17, 62))
    if dtype == torch.float32:
        frame = frame.float() / 255
    _close(GpuLutApplier(lut, MPS).apply(frame.to(MPS)[:, :, ::2]),
           GpuLutApplier(lut, CPU).apply(frame[:, :, ::2]), 1 if dtype == torch.uint8 else 2e-6)


@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, 2e-6), (torch.float16, 0.003)])
def test_denoise_odd_size_and_strides(dtype, tolerance):
    frame = (_random((2, 3, 17, 62)).float() / 255).to(dtype)
    _close(spatial_denoise(frame.to(MPS)[:, :, :, ::2], 5, 2.0, 0.12),
           spatial_denoise(frame[:, :, :, ::2], 5, 2.0, 0.12), tolerance)


@pytest.mark.parametrize("ten_bit", [False, True])
@pytest.mark.parametrize("full_range", [False, True])
def test_yuv_range_endpoints_and_nonunit_pixel_strides(ten_bit, full_range):
    dtype = torch.uint16 if ten_bit else torch.uint8
    peak, black, white, center, scale = (1023, 64, 940, 512, 64) if ten_bit else (255, 16, 235, 128, 1)
    for code, expected in [(0 if full_range else black, 0), (peak if full_range else white, 255)]:
        y = torch.full((8, 24), code * scale, dtype=dtype).to(MPS)[:, ::2]
        uv = torch.full((4, 12, 2), center * scale, dtype=dtype).to(MPS)[:, ::2]
        out = torch.empty((3, 8, 24), dtype=torch.uint8, device=MPS)[:, :, ::2]
        converter = YuvToRgbConverter(8, 12, Colorspace.ITU709, full_range, ten_bit, MPS)
        converter.convert_into(y, uv, out)
        assert torch.equal(out.cpu(), torch.full((3, 8, 12), expected, dtype=torch.uint8))


@pytest.mark.parametrize("ten_bit", [False, True])
def test_yuv_rejects_wrong_mps_plane_dtype(ten_bit):
    converter = YuvToRgbConverter(8, 12, Colorspace.ITU709, False, ten_bit, MPS)
    with pytest.raises(TypeError, match="Expected.*MPS planes"):
        converter.convert(torch.zeros((8, 12), device=MPS), torch.zeros((4, 6, 2), device=MPS))


def test_yuv_rejects_mixed_device_and_bad_output():
    converter = YuvToRgbConverter(8, 12, Colorspace.ITU709, False, False, MPS)
    y = torch.zeros((8, 12), dtype=torch.uint8, device=MPS)
    uv = torch.zeros((4, 6, 2), dtype=torch.uint8, device=MPS)
    with pytest.raises(ValueError, match="converter device"):
        converter.convert_into(y, uv.cpu(), torch.empty((3, 8, 12), dtype=torch.uint8, device=MPS))
    with pytest.raises(ValueError, match="RGB destination"):
        converter.convert_into(y, uv, torch.empty((3, 8, 12), device=MPS))


def test_yuv_rejects_raw_cuda_surfaces_on_mps():
    converter = YuvToRgbConverter(8, 12, Colorspace.ITU709, False, False, MPS)
    with pytest.raises(RuntimeError, match="requires a CUDA converter"):
        converter.convert_surface_into(0, 0, 12, torch.empty((3, 8, 12), dtype=torch.uint8, device=MPS))
    with pytest.raises(RuntimeError, match="requires a CUDA converter"):
        converter.convert_frames_into([], torch.empty((0, 3, 8, 12), dtype=torch.uint8, device=MPS))
