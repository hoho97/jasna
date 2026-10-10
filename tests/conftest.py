"""Load NVIDIA TensorRT DLLs first when the active environment provides them."""
from importlib.util import find_spec

import pytest

_HAS_TENSORRT = find_spec("tensorrt") is not None

if _HAS_TENSORRT and find_spec("tensorrt_libs") is not None:
    import tensorrt_libs

# These modules import optional native SDKs at module scope. Skip before import,
# without installing fake SDKs globally.
_OPTIONAL_MODULES = {
    "test_basicvsrpp_engine_compilation.py": ("tensorrt",),
    "test_basicvsrpp_sub_engines.py": ("tensorrt",),
    "test_rtx_superres_restorer.py": ("tensorrt_libs",),
    "test_trt_runner.py": ("tensorrt",),
    "test_trt_utils.py": ("tensorrt",),
    "test_unet4x_secondary_restorer.py": ("tensorrt",),
}


def pytest_ignore_collect(collection_path, config):
    """Do not import tests requiring absent native SDKs, even during -m selection.

    Import-time skips become JUnit testcases before pytest applies -m and would
    incorrectly contaminate the strict real-MPS gate with unrelated SDK skips.
    """
    requirements = _OPTIONAL_MODULES.get(collection_path.name, ())
    return True if any(find_spec(name) is None for name in requirements) else None


_REAL_MPS_MODULES = {
    "test_mps_basicvsrpp_inference.py", "test_mps_rfdetr_inference.py",
    "test_mps_e2e.py", "test_mps_pipeline_runtime.py", "test_mps_tensor_fallbacks.py",
}
_MODEL_MPS_MODULES = {
    "test_mps_basicvsrpp_inference.py", "test_mps_rfdetr_inference.py", "test_mps_e2e.py",
}
_NVIDIA_KERNEL_MODULES = {
    "test_cas.py", "test_cas_sharpen_into.py", "test_denoise_kernel.py",
    "test_lut_kernel.py", "test_rgb_to_yuv_kernel.py", "test_resize_normalize.py",
    "test_video_decoder_software.py", "test_video_encoder_mux.py", "test_e2e.py",
    "test_ltx_trial.py", "test_ltx_seed_preview_gpu.py", "test_video_decoder_backends.py",
}
# These tests contain real device operations without a parametrized device fixture.
_REAL_MPS_TESTS = {
    "test_mps_software_encode.py": {"test_parallel_mps_encoders", "test_real_1080p_mps_encode"},
    "test_mps_model_loading.py": {"test_real_rfdetr_cpu_load_then_mps",
                                  "test_real_restoration_cpu_load_then_mps",
                                  "test_real_yolo_cpu_load_then_mps"},
}


# Optional SDK-only cases inside otherwise portable modules. Their bodies mock
# compilation or return an existing engine; they do not execute GPU kernels.
_OPTIONAL_SDK_TESTS = {
    ("test_gui_engine_preflight.py", "test_get_onnx_tensorrt_engine_path_matches_compile_return_when_present"),
    ("test_suppress_noise.py", "test_tensorrt_plugin_experimental_warning_is_muted"),
    ("test_ltx_transformer.py", "test_block_forward_compiles_without_timing_kernels"),
}


def _test_requirements(item):
    """Classify actual runtime requirements, not words in a mocked test's name."""
    filename = item.path.name
    name = getattr(item, "originalname", None) or item.name.split("[")[0]
    requirements = set()
    if filename in _OPTIONAL_MODULES or (filename, name) in _OPTIONAL_SDK_TESTS:
        requirements.add("nvidia")
        if filename == "test_ltx_transformer.py":
            requirements.add("rocm")
    if filename == "test_ltx_trial.py" and name == "test_placeholders_match_the_real_files":
        requirements.add("model_required")
    if filename == "test_crop_buffer.py" and name == "test_output_dtype_follows_requested_precision":
        params = getattr(getattr(item, "callspec", None), "params", {})
        if str(params.get("dtype")) == "torch.float16":
            requirements.update(("nvidia", "rocm"))
    if filename in _REAL_MPS_MODULES or name in _REAL_MPS_TESTS.get(filename, ()):
        requirements.add("mps_real")
    if (filename in _MODEL_MPS_MODULES or "real_weights" in item.fixturenames
            or name == "test_real_checkpoints_threaded_pass_and_memory_plateau"
            or filename in {"test_ltx_seed_preview_gpu.py", "test_tvai_integration.py"}):
        requirements.add("model_required")
    if filename == "test_os_utils.py" and name == "test_freeconsole_dangling_std_handles_break_subprocess_until_redirect":
        requirements.add("windows")
    # Existing CUDA skips identify real operations even in mixed CPU/GPU modules.
    # Keep mocked CUDA/ROCm routing and compilation tests in the CPU suite.
    for mark in item.iter_markers("skipif"):
        reason = mark.kwargs.get("reason", "").lower()
        if "cuda" in reason or "nvidia gpu" in reason or reason.startswith("needs a gpu"):
            requirements.add("nvidia")
            if filename not in _NVIDIA_KERNEL_MODULES:
                requirements.add("rocm")
        if filename == "test_e2e.py" and "weights" in reason:
            requirements.add("model_required")
    # The real video tests intentionally exercise both CPU and accelerator cases.
    if filename in {"test_mps_video_decode.py", "test_mps_software_encode.py", "test_video_decoder_seek.py"}:
        device = getattr(item, "callspec", None)
        device = str(device.params.get("device", "")) if device else ""
        if device == "mps":
            requirements.add("mps_real")
        elif device == "cuda":
            requirements.update(("nvidia", "rocm"))
    return requirements


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    # Run before pytest's -m deselection so these categories are selectable.
    for item in items:
        requirements = _test_requirements(item)
        explicit = {mark.name for mark in item.iter_markers()}
        categories = requirements | (explicit & {"nvidia", "rocm", "windows", "linux", "mps_real", "model_required"})
        for marker in requirements:
            item.add_marker(getattr(pytest.mark, marker))
        if not categories:
            item.add_marker(pytest.mark.platform_independent)


def pytest_runtest_setup(item):
    import sys

    if item.get_closest_marker("windows") and sys.platform != "win32":
        pytest.skip("requires real Windows")
    if item.get_closest_marker("linux") and not sys.platform.startswith("linux"):
        pytest.skip("requires real Linux")
    nvidia = item.get_closest_marker("nvidia") is not None
    rocm = item.get_closest_marker("rocm") is not None
    # SDK unit tests mock their GPU boundary; only TrtRunner allocates real CUDA
    # tensors. A machine with the SDK installed can run those mocked tests on CPU.
    name = getattr(item, "originalname", None) or item.name.split("[")[0]
    sdk_only = ((item.path.name in _OPTIONAL_MODULES and item.path.name != "test_trt_runner.py")
                or (item.path.name, name) in _OPTIONAL_SDK_TESTS)
    if (nvidia or rocm) and not sdk_only:
        import torch

        is_rocm = bool(getattr(torch.version, "hip", None))
        if not torch.cuda.is_available() or (is_rocm and not rocm) or (not is_rocm and not nvidia):
            pytest.skip("requires real " + ("NVIDIA or ROCm" if nvidia and rocm else "ROCm" if rocm else "NVIDIA"))
    if item.get_closest_marker("mps_real"):
        import os
        import torch

        if not torch.backends.mps.is_available():
            # Opting into checkpoint verification must never turn failure into skip.
            assert not os.environ.get("JASNA_TEST_MODEL_WEIGHTS_DIR"), "MPS unavailable after real verification opt-in"
            pytest.skip("requires real Apple MPS")


@pytest.fixture
def license_boundary(monkeypatch):
    """Mock only the private license API consumed by public GUI unit tests."""
    from types import SimpleNamespace
    import jasna.protection as protection

    class ProtectionError(Exception):
        pass

    class LicenseError(ProtectionError):
        pass

    monkeypatch.setattr(protection, "ProtectionError", ProtectionError, raising=False)
    monkeypatch.setattr(protection, "LicenseError", LicenseError, raising=False)
    for name in ("ForgedLicenseError", "RetiredLicenseError", "MalformedLicenseError"):
        monkeypatch.setattr(protection, name, type(name, (LicenseError,), {}), raising=False)
    monkeypatch.setattr(protection, "license_store", SimpleNamespace(
        load_license=lambda: None, set_license=lambda email, key: None,
    ), raising=False)
    return protection


@pytest.fixture
def nvidia_encoding_gui(monkeypatch):
    """Exercise the existing NVIDIA CQ widgets independently of host hardware.

    Apple software-encoder GUI controls belong to the separate GUI port; these
    layout/settings regressions retain the original NVIDIA widget contract.
    """
    from jasna.accelerator import AcceleratorVendor
    monkeypatch.setattr(
        "jasna.gui.settings_sections.encoding.vendor_for_device",
        lambda: AcceleratorVendor.NVIDIA,
    )


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
        # Match Python's import contract for patch("package.module.symbol").
        # A sys.modules entry alone leaves the parent package attribute absent.
        from importlib import import_module
        parent, _, leaf = name.rpartition(".")
        monkeypatch.setattr(import_module(parent), leaf, module, raising=False)
