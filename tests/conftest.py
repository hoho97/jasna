"""Load NVIDIA TensorRT DLLs first when the active environment provides them."""
from importlib.util import find_spec

import pytest

_HAS_TENSORRT = find_spec("tensorrt") is not None

if _HAS_TENSORRT and find_spec("tensorrt_libs") is not None:
    import tensorrt_libs

collect_ignore = [] if _HAS_TENSORRT else [
    "test_basicvsrpp_sub_engines.py",
    "test_rtx_superres_restorer.py",
    "test_torch_tensorrt_export.py",
    "test_trt_runner.py",
    "test_trt_utils.py",
    "test_unet4x_secondary_restorer.py",
]


@pytest.fixture
def hidpi(request):
    """Reproduce Windows display scaling on a platform whose DPI factor is always 1.

    CustomTkinter multiplies its detected per-monitor DPI factor by these process-global
    factors, so setting them makes geometry()/minsize() and CTk widget sizes behave exactly
    as on a scaled Windows monitor while winfo_* keeps reporting physical pixels - the
    asymmetry behind issue #241. They are class attributes on ScalingTracker and leak into
    every later test unless reset.
    """
    import customtkinter as ctk

    factor = request.param
    ctk.set_widget_scaling(factor)
    ctk.set_window_scaling(factor)
    try:
        yield factor
    finally:
        ctk.set_widget_scaling(1.0)
        ctk.set_window_scaling(1.0)


@pytest.fixture
def no_gpu_cleanup(monkeypatch):
    """Skip the per-job torch cleanup: it initializes CUDA, which can outlast thread joins in a busy run."""
    monkeypatch.setattr("jasna.gui.processor._cleanup_torch", lambda torch_mod: None)


@pytest.fixture
def nvidia_trt_compiler(monkeypatch):
    # Mock the external TensorRT boundary without requiring its platform wheel.
    # Keep the original compile arguments/assertions, including workspace size.
    import sys
    from types import ModuleType
    from unittest.mock import MagicMock
    import torch

    monkeypatch.setattr(torch.version, "cuda", "test-nvidia")
    monkeypatch.setattr(torch.version, "hip", None)
    module = ModuleType("jasna.trt")
    module.compile_onnx_to_tensorrt_engine = MagicMock()
    monkeypatch.setitem(sys.modules, "jasna.trt", module)


@pytest.fixture
def nvidia_build(monkeypatch):
    """Mock the CUDA build identity, without requiring NVIDIA hardware."""
    import torch
    monkeypatch.setattr(torch.version, "cuda", "test-nvidia")
    monkeypatch.setattr(torch.version, "hip", None)


@pytest.fixture
def nvidia_cli(nvidia_build, nvidia_optional_modules, monkeypatch):
    from contextlib import nullcontext
    monkeypatch.setattr("jasna.accelerator.device_context", lambda _: nullcontext())


@pytest.fixture
def nvidia_optional_modules(monkeypatch):
    """Mock SDK constructors at the boundary used by CLI composition tests."""
    import sys
    from types import ModuleType
    from unittest.mock import MagicMock
    for name, symbols in {
        "jasna.restorer.unet4x_secondary_restorer": ["Unet4xSecondaryRestorer"],
        "jasna.restorer.rtx_superres_secondary_restorer": ["RtxSuperresSecondaryRestorer"],
        "jasna.restorer.basicvsrpp_sub_engines": ["compile_basicvsrpp_engines", "BasicVSRPlusPlusNetSplit"],
    }.items():
        module = ModuleType(name)
        for symbol in symbols:
            setattr(module, symbol, MagicMock())
        monkeypatch.setitem(sys.modules, name, module)
