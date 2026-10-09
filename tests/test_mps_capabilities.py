from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock
import sys

import pytest
import torch

from jasna import accelerator, os_utils
from jasna.backend_preflight import validate_backend_options
from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled, _subprocess_compile
from jasna.main import build_parser, main, _check_system
from factories import session_config


@pytest.fixture
def mps(monkeypatch):
    backend = SimpleNamespace(is_built=lambda: True, is_available=lambda: True)
    monkeypatch.setattr(torch.backends, "mps", backend)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch, "cuda", MagicMock(side_effect=AssertionError("CUDA on MPS")))
    return backend


def test_mps_preflight_never_queries_cuda_or_driver(mps, monkeypatch):
    monkeypatch.setattr(os_utils, "find_executable", lambda _: pytest.fail("nvidia-smi lookup"))
    assert os_utils.check_supported_gpu("mps") == (True, "Apple MPS")
    ok, info = os_utils.check_gpu_driver_version("mps")
    assert ok and "macOS" in info and "MPS" in info
    import jasna.main as cli
    monkeypatch.setattr(cli, "check_required_executables", lambda: None)
    _check_system(build_parser().parse_args(["--device", "mps"]))
    assert not torch.cuda.mock_calls


@pytest.mark.parametrize("built,available,reason", [(False, False, "mps_not_built"), (True, False, "mps_unavailable")])
def test_mps_failure_diagnostic(mps, built, available, reason):
    mps.is_built = lambda: built
    mps.is_available = lambda: available
    assert os_utils.check_supported_gpu("mps") == (False, reason)
    assert "MPS" in os_utils.gpu_check_error(reason)
    assert os_utils.check_gpu_driver_version("mps")[0] is False


@pytest.mark.parametrize("option,value", [
    ("secondary_restoration", "unet-4x"), ("secondary_restoration", "rtx-super-res"),
    ("secondary_restoration", "tvai"), ("restoration_model_name", "ltx"),
    ("ltx_fast", True), ("advanced_video", True),
])
def test_mps_features_rejected(mps, option, value):
    with pytest.raises(ValueError, match="not supported on mps"):
        validate_backend_options("mps", **{option: value})


def test_default_free_models_accepted(mps):
    assert build_parser().parse_args([]).secondary_restoration == "none"
    validate_backend_options("mps")
    caps = accelerator.capabilities_for_device("mps")
    assert not caps.tensorrt and not caps.secondary_restoration and not caps.ltx


@pytest.mark.parametrize("extra", [
    ["--secondary-restoration", "unet-4x"], ["--secondary-restoration", "tvai"],
    ["--secondary-restoration", "rtx-super-res"], ["--restoration-model-name", "ltx"],
    ["--ltx-fast"], ["--ltx-trial"], ["--stream"], ["--segments", "1-2"],
    ["--vr-mode", "sbs"], ["--benchmark"],
])
def test_cli_rejects_before_system_or_models(mps, monkeypatch, capsys, extra):
    monkeypatch.setattr("jasna.main._check_system", lambda _: pytest.fail("too late"))
    monkeypatch.setattr(sys, "argv", ["jasna", "--device", "mps", *extra])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "not supported on mps" in capsys.readouterr().err


@pytest.mark.parametrize("secondary", ["unet-4x", "rtx-super-res", "tvai"])
def test_session_and_direct_secondary_gate_before_imports(mps, secondary):
    from jasna.session_factory import build_restoration_session, _build_secondary_restorer
    config = session_config(device="mps", secondary_restoration=secondary)
    for call in [lambda: build_restoration_session(config, log_callback=None),
                 lambda: _build_secondary_restorer(config, torch.device("mps"))]:
        with pytest.raises(ValueError, match="not supported on mps"):
            call()


def test_direct_ltx_gate_before_cuda_and_downloads(mps):
    from jasna.session_factory import _ltx_model_files
    with pytest.raises(ValueError, match="LTX.*not supported"):
        _ltx_model_files(session_config(device="mps", ltx_fast=True), torch.device("mps"), log_callback=None)
    assert not torch.cuda.mock_calls


def test_checkpoint_compilation_has_no_subprocess_or_cache(mps, monkeypatch, tmp_path):
    checkpoint = tmp_path / "rfdetr-v6.pt"
    checkpoint.write_bytes(b"path validation only")
    monkeypatch.setattr("jasna.engine_compiler.subprocess.Popen", lambda *a, **k: pytest.fail("compilation"))
    monkeypatch.setattr("jasna.engine_compiler.all_basicvsrpp_sub_engines_exist", lambda *a: pytest.fail("cache"))
    req = EngineCompilationRequest(device="mps", fp16=False, basicvsrpp=True,
        detection=True, detection_model_name="rfdetr-v6", detection_model_path=str(checkpoint))
    assert ensure_engines_compiled(req).use_basicvsrpp_tensorrt is False
    import logging
    try:
        _subprocess_compile(req)
    finally:
        logging.disable(logging.NOTSET)
    with pytest.raises(RuntimeError, match="NVIDIA TensorRT"):
        ensure_engines_compiled(replace(req, unet4x=True))
    try:
        with pytest.raises(RuntimeError, match="NVIDIA TensorRT"):
            _subprocess_compile(replace(req, unet4x=True))
    finally:
        logging.disable(logging.NOTSET)


def test_mps_gui_engine_preflight_has_no_compiled_requirements(mps):
    from jasna.gui.engine_preflight import run_engine_preflight
    from jasna.gui.models import AppSettings
    result = run_engine_preflight(AppSettings(secondary_restoration="none"), device="mps")
    assert result.requirements == ()
    assert not result.should_warn_first_run_slow


@pytest.mark.parametrize("hip,cuda,trt,secondary", [(None, "13.0", True, True), ("7.2", None, False, False)])
def test_nvidia_rocm_capability_contract(monkeypatch, hip, cuda, trt, secondary):
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.version, "cuda", cuda)
    caps = accelerator.capabilities_for_device("cuda:0")
    assert caps.tensorrt is trt and caps.secondary_restoration is secondary
    assert caps.ltx and caps.advanced_video and caps.streams
    validate_backend_options("cuda:0", restoration_model_name="ltx", advanced_video=True)


@pytest.mark.parametrize("backend", ["vali", "pyav-hw"])
def test_mps_rejects_hardware_decode_before_open(mps, monkeypatch, backend):
    from jasna.media.video_decoder import VideoReader, VideoDecodeError
    import jasna.media.video_decoder as decoder
    monkeypatch.setenv("JASNA_DECODE_BACKEND", backend)
    monkeypatch.setattr(decoder.av, "open", lambda *a, **k: pytest.fail("opened video"))
    with pytest.raises(ValueError, match="Hardware decode.*not supported"):
        validate_backend_options("mps", decode_backend=backend)
    reader = VideoReader("not-opened.mp4", 1, torch.device("mps"), MagicMock())
    with pytest.raises(VideoDecodeError, match="Hardware decode.*not supported"):
        reader.__enter__()
    monkeypatch.setattr(sys, "argv", ["jasna", "--device", "mps"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2


@pytest.mark.parametrize("available", [True, False])
def test_rocm_driver_never_looks_for_nvidia(monkeypatch, available):
    monkeypatch.setattr(torch.version, "hip", "7.2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(os_utils, "find_executable", lambda _: pytest.fail("nvidia-smi"))
    ok, info = os_utils.check_gpu_driver_version("cuda:0")
    assert ok is available and "ROCm 7.2" in info


def test_amd_detector_compilation_never_spawns(monkeypatch):
    monkeypatch.setattr(torch.version, "hip", "7.2")
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr("jasna.engine_compiler.subprocess.Popen", lambda *a, **k: pytest.fail("AMD compilation"))
    req = EngineCompilationRequest(device="cuda:0", fp16=True, basicvsrpp=True,
                                   detection=True, detection_model_name="rfdetr-v6", detection_model_path="det.pt")
    assert not ensure_engines_compiled(req).use_basicvsrpp_tensorrt


def test_segment_and_late_model_loading_cannot_bypass_mps_gates(mps):
    from jasna.session_factory import RestorationSession, provide_restoration_models, build_pipeline
    from jasna.segments import SegmentRange
    config = session_config(device="mps")
    session = RestorationSession(torch.device("mps"), restoration_pipeline=None)
    with pytest.raises(ValueError, match="LTX.*not supported"):
        provide_restoration_models(config, session, frozenset({"ltx"}), log_callback=None)
    with pytest.raises(ValueError, match="smart rendering.*not supported"):
        build_pipeline(config, session, None, None, segments=(SegmentRange(1, 2),))
