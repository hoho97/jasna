"""Opt-in real CLI integration: no substituted detector, restorer or encoder.

JASNA_TEST_MODEL_WEIGHTS_DIR points to read-only original checkpoints. The
repository's existing mosaic test clip supplies eight frames; sine audio is
synthetic. Run with -s to capture inference contracts and ffprobe evidence.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import av
import pytest
import torch


@pytest.fixture
def real_mps_weights():
    weights = os.environ.get("JASNA_TEST_MODEL_WEIGHTS_DIR")
    if not weights:
        pytest.skip("set JASNA_TEST_MODEL_WEIGHTS_DIR to opt into real checkpoints")
    assert torch.backends.mps.is_built() and torch.backends.mps.is_available()
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1"
    weights = Path(weights)
    assert (weights / "rfdetr-v6.pt").is_file()
    assert (weights / "lada_mosaic_restoration_model_generic_v1.2.pth").is_file()
    return weights


def test_real_cli_detection_restoration_encode(real_mps_weights, tmp_path, monkeypatch):
    from jasna.main import main
    from jasna.mosaic.rfdetr_torch_runner import RfDetrTorchRunner
    from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel
    from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
    from jasna.session_factory import RestorationSession

    root = Path(os.environ.get("JASNA_TEST_VIDEO_OUTPUT_DIR", tmp_path))
    root.mkdir(parents=True, exist_ok=True)
    source, output = root / "cli-source.mp4", root / "cli-restored.mp4"
    fixture = Path(__file__).resolve().parents[1] / "assets/test_clip1_1080p.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "4", "-i", str(fixture),
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.266667",
                    "-map", "0:v:0", "-map", "1:a:0", "-frames:v", "8", "-c:v", "libx264",
                    "-crf", "12", "-colorspace", "bt709", "-c:a", "aac", str(source)], check=True)
    detections, contracts, detector_contracts, closed = [], [], [], []
    infer = RfDetrTorchRunner.infer
    detect = RfDetrMosaicDetectionModel.__call__
    restore = BasicvsrppMosaicRestorer.raw_process
    close = RestorationSession.close

    def observed_infer(self, inputs):
        result = infer(self, inputs)
        assert set(result) == {"dets", "labels", "masks"}
        assert result["dets"].shape == (1, 200, 4)
        assert result["labels"].shape == (1, 200, 3)
        assert result["masks"].shape == (1, 200, 144, 144)
        for value in result.values():
            assert value.device.type == "mps" and value.dtype == torch.float32
            assert torch.isfinite(value).all().item()
        detector_contracts.append({name: tuple(value.shape) for name, value in result.items()})
        return result

    def observed_detect(self, frames, **kwargs):
        assert frames.device.type == "mps" and self.input_dtype == torch.float32
        result = detect(self, frames, **kwargs)
        detections.extend(len(boxes) for boxes in result.boxes_xyxy)
        return result

    def observed_restore(self, frames):
        result = restore(self, frames)
        assert result.shape == (len(frames), 3, 256, 256)
        assert result.device.type == "mps" and result.dtype == torch.float32
        assert torch.isfinite(result).all().item()
        assert (result - torch.stack(frames).float().div(255)).abs().mean().item() > 1e-4
        contracts.append(tuple(result.shape))
        return result

    def observed_close(self):
        close(self)
        assert self._detection_model is None
        assert self.restoration_pipeline.restorer.model is None
        closed.append(True)

    monkeypatch.setattr(RfDetrTorchRunner, "infer", observed_infer)
    monkeypatch.setattr(RfDetrMosaicDetectionModel, "__call__", observed_detect)
    monkeypatch.setattr(BasicvsrppMosaicRestorer, "raw_process", observed_restore)
    monkeypatch.setattr(RestorationSession, "close", observed_close)
    monkeypatch.setenv("JASNA_MODEL_WEIGHTS_DIR", str(real_mps_weights))
    monkeypatch.delenv("JASNA_MAIN_PID", raising=False)
    # Forbid actual CUDA calls, while retaining all production MPS operations.
    def no_cuda(*a, **kw):
        pytest.fail("MPS CLI called CUDA")
    for name in ("set_device", "current_stream", "mem_get_info", "empty_cache", "synchronize", "ipc_collect"):
        monkeypatch.setattr(torch.cuda, name, no_cuda)
    monkeypatch.setattr(sys, "argv", ["jasna", "--device", "mps", "--input", str(source),
                                     "--output", str(output), "--no-progress", "--log-level", "info"])
    started = time.perf_counter()
    main()
    assert len(detector_contracts) == 8
    assert sum(detections) > 0, "zero-detection passthrough is not E2E success"
    assert contracts and max(shape[0] for shape in contracts) >= 2
    assert closed == [True]
    assert not {t.name for t in threading.enumerate()} & {
        "DecodeDetect", "PrimaryRestore", "SecondaryRestore", "BlendEncode", "VramOffloader"}
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                            "stream=codec_name,codec_type,width,height,nb_read_frames,duration,avg_frame_rate",
                            "-of", "json", str(output)], check=True, text=True, capture_output=True)
    streams = json.loads(probe.stdout)["streams"]
    video = next(s for s in streams if s["codec_type"] == "video")
    audio = next(s for s in streams if s["codec_type"] == "audio")
    assert (video["codec_name"], video["width"], video["height"], int(video["nb_read_frames"])) == ("h264", 1920, 1080, 8)
    assert video["avg_frame_rate"] == "30/1"
    assert float(video["duration"]) == pytest.approx(8 / 30, abs=1e-5)
    assert audio["codec_name"] == "aac" and int(audio["nb_read_frames"]) > 0
    with av.open(str(output)) as container:
        frames = list(container.decode(video=0))
        assert len(frames) == 8
        assert all(a.pts < b.pts for a, b in zip(frames, frames[1:]))
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(output), "-f", "null", "-"], check=True)
    print(f"\nMPS detections={detections}; restoration_shapes={contracts}; detector_shapes={detector_contracts[0]}; FP32 finite MPS; seconds={time.perf_counter()-started:.3f}")
    print(f"ffprobe {output}: {probe.stdout}")


def test_installed_cli_exit_statuses(real_mps_weights, tmp_path):
    """Exercise OS exit codes, including SIGINT while real workers are running."""
    import signal

    cli = Path(sys.executable).with_name("jasna")
    assert cli.is_file(), "install this checkout (pip/uv pip install -e .) before verifying CLI"
    env = dict(os.environ, JASNA_MODEL_WEIGHTS_DIR=str(real_mps_weights))
    env.pop("JASNA_MAIN_PID", None)
    fixture = Path(__file__).resolve().parents[1] / "assets/test_clip1_1080p.mp4"
    marker = tmp_path / "post-export-ran"
    base = [str(cli), "--input", str(fixture), "--output", str(tmp_path / "cancelled.mp4"),
            "--no-progress", "--log-level", "info", "--post-export-action", "command",
            "--post-export-command", f"touch {marker}"]
    # Default device auto-selects MPS on the real Apple host.
    log = tmp_path / "cancel.log"
    with log.open("w") as handle:
        process = subprocess.Popen(base, env=env, stdout=handle, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 60
            while "jasna.pipeline_threads INFO: Processing " not in log.read_text():
                assert process.poll() is None, log.read_text()
                assert time.monotonic() < deadline, "CLI did not start workers in 60 seconds"
                time.sleep(.02)
            process.send_signal(signal.SIGINT)
            assert process.wait(timeout=30) == 130, log.read_text()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    assert "Processing cancelled" in log.read_text()
    assert not marker.exists(), "cancel must not run success actions"
    # A real invalid video reaches media probing and returns an ordinary failure.
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a video")
    failed = subprocess.run([*base, "--input", str(broken)], env=env,
                            capture_output=True, text=True, timeout=60)
    assert failed.returncode == 1, failed.stderr
    assert not marker.exists()
    invalid = subprocess.run([str(cli), "--device", "mps", "--fp16"], env=env,
                             capture_output=True, text=True, timeout=30)
    assert invalid.returncode == 2 and "--no-fp16" in invalid.stderr
    print("\ninstalled CLI: automatic MPS selection; SIGINT=130; invalid media=1; unsupported option=2; no success actions on failure/cancel")
