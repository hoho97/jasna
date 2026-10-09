"""Portable dispatch regression checks; actual CUDA numerics stay in GPU tests."""
from unittest.mock import Mock

import pytest
import torch
from av.video.reformatter import Colorspace

from jasna.media import cas, lut, resize_normalize, rgb_to_yuv, yuv_to_rgb
from jasna.restorer import denoise


@pytest.mark.parametrize("nvidia", [False, True])
def test_six_fatbins_remain_nvidia_only(monkeypatch, nvidia):
    device = torch.device("cpu")
    for module in (cas, lut, resize_normalize, rgb_to_yuv, yuv_to_rgb, denoise):
        monkeypatch.setattr(module, "is_nvidia_device", lambda _device: nvidia)
    normalizer = resize_normalize.ResizeNormalizer(device=device, dtype=torch.float32,
                                                  mean=(0.,) * 3, std=(1.,) * 3, fill=(0.,) * 3)
    assert normalizer.available is nvidia
    assert rgb_to_yuv.RgbToYuvConverter("nv12_bt709_limited", device=device).uses_kernel is nvidia
    decoder = yuv_to_rgb.YuvToRgbConverter(4, 4, Colorspace.ITU709, False, False, device)
    sharpener = cas.GpuCasSharpener(0.5, ten_bit=False, device=device)
    table = lut.CubeLut(2, False, torch.tensor([[0., 0., 0.], [1., 1., 1.]]), (0.,) * 3, (1.,) * 3)
    applier = lut.GpuLutApplier(table, device)
    for kernel in (decoder._cuda_kernel, sharpener._kernel, applier._kernel):
        assert (kernel is not None) is nvidia
    frames = torch.full((1, 3, 4, 4), 0.5)
    launch = Mock(return_value=frames)
    monkeypatch.setattr(denoise, "_kernel_denoise", launch)
    result = denoise.spatial_denoise(frames, 3, 1., 0.1)
    assert launch.called is nvidia
    assert torch.allclose(result, frames)
