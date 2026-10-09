import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

import pytest

from jasna.main import build_parser, main, _run_videos, _validate_mps_cli_options


def test_cli_creates_stream_on_chosen_device(tmp_path: Path, nvidia_build) -> None:
    input_path = tmp_path / "in.mp4"
    input_path.touch()
    output_path = tmp_path / "out.mkv"
    model_weights = tmp_path / "model_weights"
    model_weights.mkdir()
    restoration_path = model_weights / "lada_mosaic_restoration_model_generic_v1.2.pth"
    restoration_path.touch()
    detection_path = model_weights / "rfdetr-v3.onnx"
    detection_path.touch()

    device_capture: list = []
    fake_stream = MagicMock()

    class NoOpDeviceContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def record_device(device):
        device_capture.append(device)
        return NoOpDeviceContext()

    pipeline_capture: dict = {}

    def capture_pipeline(**kwargs):
        pipeline_capture.update(kwargs)
        mock = MagicMock()
        return mock

    with (
        patch("jasna.main.check_ascii_install_path", return_value=(True, "C:\\fake")),
        patch(
            "jasna.main.check_supported_gpu",
            return_value=(True, "Fake GPU"),
        ) as check_gpu,
        patch("jasna.main.check_gpu_driver_version", return_value=(True, "610.18")),
        patch("jasna.main.check_required_executables"),        patch("jasna.main.check_windows_nvidia_sysmem_fallback_policy", return_value=(True, "OK")),
        patch("jasna.engine_compiler.ensure_engines_compiled", return_value=MagicMock(use_basicvsrpp_tensorrt=False)),
        patch("jasna.pipeline.Pipeline", side_effect=capture_pipeline),
        patch("jasna.restorer.basicvsrpp_mosaic_restorer.BasicvsrppMosaicRestorer", MagicMock()),
    ):
        import torch

        with patch("torch.cuda.device", side_effect=record_device), patch(
            "torch.cuda.Stream", return_value=fake_stream
        ):
            with patch.object(
                sys,
                "argv",
                [
                    "jasna",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--device",
                    "cuda:1",
                    "--restoration-model-path",
                    str(restoration_path),
                    "--detection-model-path",
                    str(detection_path),
                ],
            ):
                from jasna.main import main

                main()

    assert any(d == torch.device("cuda:1") for d in device_capture)
    check_gpu.assert_called_once_with("cuda:1")
    assert pipeline_capture["session"].device == torch.device("cuda:1")


@pytest.fixture
def apple(monkeypatch):
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)


def test_apple_defaults_and_explicit_device(apple):
    args = build_parser().parse_args([])
    assert (args.device, args.fp16, args.compile_basicvsrpp) == ("mps", False, False)
    assert (args.batch_size, args.max_clip_size, args.temporal_overlap) == (1, 16, 2)
    assert (args.codec, args.vr_mode, args.secondary_restoration, args.detection_model) == ("h264", "off", "none", "rfdetr-v6")
    _validate_mps_cli_options(args)
    cuda = build_parser().parse_args(["--device", "cuda:0"])
    assert (cuda.batch_size, cuda.max_clip_size, cuda.temporal_overlap, cuda.fp16,
            cuda.compile_basicvsrpp, cuda.codec, cuda.vr_mode) == (4, 90, 8, True, True, "hevc", "auto")


@pytest.mark.parametrize("vendor", ["nvidia", "amd"])
def test_vendor_defaults_preserved(monkeypatch, vendor):
    monkeypatch.setattr(torch.version, "cuda", "test" if vendor == "nvidia" else None)
    monkeypatch.setattr(torch.version, "hip", "test" if vendor == "amd" else None)
    args = build_parser().parse_args([])
    assert (args.device, args.batch_size, args.max_clip_size, args.temporal_overlap,
            args.fp16, args.compile_basicvsrpp, args.codec, args.vr_mode) == ("cuda:0", 4, 90, 8, True, True, "hevc", "auto")


def test_explicit_values_survive_device_defaults(apple):
    args = build_parser().parse_args(["--device=mps", "--batch-size", "2", "--max-clip-size", "8", "--temporal-overlap", "1"])
    assert (args.batch_size, args.max_clip_size, args.temporal_overlap) == (2, 8, 1)


@pytest.mark.parametrize("options", [
    ["--fp16"], ["--compile-basicvsrpp"], ["--batch-size", "3"], ["--max-clip-size", "33"],
    ["--codec", "hevc"], ["--cq", "20"], ["--vr-mode", "auto"],
    ["--detection-model", "lada-yolo-v4"], ["--license-key", "test"],
    ["--secondary-restoration", "unet-4x"], ["--restoration-model-name", "ltx"],
    ["--stream"], ["--segments", "1-2"], ["--benchmark"],
])
def test_mps_unsupported_options_fail_before_system_or_models(apple, monkeypatch, options, capsys):
    check = Mock(side_effect=AssertionError("late validation"))
    monkeypatch.setattr("jasna.main._check_system", check)
    monkeypatch.setattr(sys, "argv", ["jasna", "--device", "mps", *options])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "mps" in capsys.readouterr().err.lower()
    check.assert_not_called()


@pytest.mark.parametrize("mode", ["interrupt", "cancel", "failure", "success"])
def test_video_exit_and_post_export_contract(mode):
    command = Mock()
    pipeline = Mock(cancel_requested=mode == "cancel")
    if mode == "interrupt":
        pipeline.run.side_effect = KeyboardInterrupt()
    if mode == "failure":
        pipeline.run.side_effect = RuntimeError("worker failed")
    args = SimpleNamespace(post_export_video_command="echo done")
    original_sigint = signal.getsignal(signal.SIGINT)
    with patch("jasna.post_export_action.run_post_export_video_command", command):
        def run():
            return _run_videos(args, lambda *a: pipeline, [Path("in.mp4")], lambda p: Path("out.mp4"),
                               in_folder=False, first_index=1, total=1)
        if mode in {"interrupt", "cancel"}:
            with pytest.raises(SystemExit) as exc:
                run()
            assert exc.value.code == 130
            if mode == "interrupt":
                pipeline.cancel.assert_called_once()
        elif mode == "failure":
            with pytest.raises(RuntimeError, match="worker failed"):
                run()
        else:
            assert run()
    assert signal.getsignal(signal.SIGINT) is original_sigint
    assert command.call_count == (1 if mode == "success" else 0)
