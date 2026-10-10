"""Keep mocked backend regression coverage selectable on CPU-only hosts."""
from pathlib import Path
from types import SimpleNamespace

import pytest

import conftest


def item(filename, name="test_example", *, fixtures=(), params=None, marks=()):
    return SimpleNamespace(
        path=Path(filename), name=name, originalname=name,
        fixturenames=fixtures,
        callspec=SimpleNamespace(params=params) if params is not None else None,
        iter_markers=lambda kind=None: iter(mark for mark in marks if kind is None or mark.name == kind),
    )


@pytest.mark.parametrize("filename", [
    "test_amd_support.py", "test_mps_accelerator.py", "test_mps_capabilities.py",
    "test_main.py", "test_session_factory.py", "test_pipeline_run.py",
    "test_windows_dll_path_sanitization.py", "test_torch_tensorrt_export.py",
])
def test_mock_backend_regressions_have_no_hardware_requirement(filename):
    assert conftest._test_requirements(item(filename)) == set()


@pytest.mark.parametrize("filename", ["test_mps_video_decode.py", "test_mps_software_encode.py", "test_video_decoder_seek.py"])
def test_mixed_video_cases_classified_per_real_device(filename):
    assert conftest._test_requirements(item(filename, params={"device": "cpu"})) == set()
    assert conftest._test_requirements(item(filename, params={"device": "mps"})) == {"mps_real"}
    assert conftest._test_requirements(item(filename, params={"device": "cuda"})) == {"nvidia", "rocm"}


def test_mixed_model_loading_only_real_tests_require_weights():
    assert conftest._test_requirements(item("test_mps_model_loading.py", "test_cpu_loading_and_schema")) == set()
    assert conftest._test_requirements(item("test_mps_model_loading.py", "test_real_yolo_cpu_load_then_mps", fixtures=("real_weights",))) == {"mps_real", "model_required"}


def test_real_nvidia_kernel_skip_becomes_hardware_category():
    mark = pytest.mark.skipif(True, reason="requires an NVIDIA GPU").mark
    assert conftest._test_requirements(item("test_cas.py", marks=(mark,))) == {"nvidia"}


def test_real_eager_gpu_test_accepts_nvidia_or_rocm():
    mark = pytest.mark.skipif(True, reason="needs a GPU").mark
    assert conftest._test_requirements(item("test_amd_support.py", marks=(mark,))) == {"nvidia", "rocm"}


def test_missing_optional_sdk_skips_before_import(monkeypatch):
    monkeypatch.setattr(conftest, "find_spec", lambda name: None)
    assert conftest.pytest_ignore_collect(Path("test_basicvsrpp_engine_compilation.py"), None) is True
    assert conftest.pytest_ignore_collect(Path("test_torch_tensorrt_export.py"), None) is None
    monkeypatch.setattr(conftest, "find_spec", lambda name: object())
    assert conftest.pytest_ignore_collect(Path("test_basicvsrpp_engine_compilation.py"), None) is None


def test_mps_checkpoint_opt_in_cannot_silently_skip(monkeypatch):
    import torch

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setenv("JASNA_TEST_MODEL_WEIGHTS_DIR", "/trusted/checkpoints")
    selected = SimpleNamespace(name="test_example", path=Path("test_mps_e2e.py"), get_closest_marker=lambda name: object() if name == "mps_real" else None)
    with pytest.raises(AssertionError, match="MPS unavailable after real verification opt-in"):
        conftest.pytest_runtest_setup(selected)


def test_vali_hardware_integration_is_not_in_cpu_suite():
    mark = pytest.mark.skipif(True, reason="needs a GPU, the test clip and the python_vali fork").mark
    assert conftest._test_requirements(item("test_video_decoder_backends.py", "test_vali_backend_matches_pyav_hw_output", marks=(mark,))) == {"nvidia"}


def test_mock_optional_modules_keep_parent_import_contract(nvidia_optional_modules):
    import sys
    import jasna.restorer
    from unittest.mock import patch

    name = "jasna.restorer.unet4x_secondary_restorer"
    assert jasna.restorer.unet4x_secondary_restorer is sys.modules[name]
    with patch(name + ".Unet4xSecondaryRestorer") as constructor:
        assert jasna.restorer.unet4x_secondary_restorer.Unet4xSecondaryRestorer is constructor


@pytest.mark.parametrize("filename,name,categories", [
    ("test_gui_engine_preflight.py", "test_get_onnx_tensorrt_engine_path_matches_compile_return_when_present", {"nvidia"}),
    ("test_suppress_noise.py", "test_tensorrt_plugin_experimental_warning_is_muted", {"nvidia"}),
    ("test_ltx_transformer.py", "test_block_forward_compiles_without_timing_kernels", {"nvidia", "rocm"}),
    ("test_ltx_trial.py", "test_placeholders_match_the_real_files", {"model_required"}),
])
def test_optional_cases_in_mixed_modules_are_not_cpu_tests(filename, name, categories):
    assert conftest._test_requirements(item(filename, name)) == categories
    assert conftest._test_requirements(item(filename, "test_mocked_portable_case")) == set()


def test_crop_precision_classifies_only_real_half_cuda_case():
    import torch

    assert conftest._test_requirements(item("test_crop_buffer.py", "test_output_dtype_follows_requested_precision", params={"dtype": torch.float32})) == set()
    assert conftest._test_requirements(item("test_crop_buffer.py", "test_output_dtype_follows_requested_precision", params={"dtype": torch.float16})) == {"nvidia", "rocm"}


def test_sdk_only_case_does_not_require_real_gpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    selected = SimpleNamespace(path=Path("test_suppress_noise.py"),
        name="test_tensorrt_plugin_experimental_warning_is_muted",
        get_closest_marker=lambda name: object() if name == "nvidia" else None)
    conftest.pytest_runtest_setup(selected)
