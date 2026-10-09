"""Core workers on real MPS; weights/video checks are opt-in, never mocked inference."""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import av
import numpy as np
import psutil
import pytest
import torch

from jasna.accelerator import memory_summary
from jasna.blend_buffer import BlendBuffer
from jasna.media.probe import get_video_meta_data
from jasna.media.video_encoder import VideoEncoder
from jasna.mosaic.detections import Detections
from jasna.pipeline import _OfflineFrameWriter
from jasna.pipeline_threads import run_restoration_pass
from jasna.restorer.restoration_pipeline import RestorationPipeline
from jasna.vram_offloader import VramOffloader


@pytest.fixture
def mps(monkeypatch):
    if not torch.backends.mps.is_available():
        assert not os.environ.get("JASNA_TEST_MODEL_WEIGHTS_DIR"), "MPS unavailable after real verification opt-in"
        pytest.skip("requires real MPS")
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1"
    forbidden = Mock(side_effect=AssertionError("core MPS pipeline entered CUDA"))
    for name in ("set_device", "mem_get_info", "get_device_properties", "memory_allocated",
                 "memory_reserved", "empty_cache", "ipc_collect", "current_stream", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    yield torch.device("mps")
    forbidden.assert_not_called()


def pipeline_for(video, device, detector, restorer, *, overlap=0):
    return SimpleNamespace(
        device=device, restoration_pipeline=RestorationPipeline(restorer),
        input_video=video, batch_size=2, max_clip_size=4, temporal_overlap=overlap,
        max_detection_gap=0, min_detection_duration=0, enable_crossfade=True,
        scene_detection=False, vr_resolution=SimpleNamespace(resolved="off"),
        vr_projector=None, job_detection_model=detector,
    )


class TensorRestorer:
    """Deterministic real MPS ops, for scheduling/lifetime rather than model accuracy."""
    input_dtype = torch.float32
    def __init__(self, device):
        self.device = device
        self.calls = 0
    def raw_process(self, images):
        self.calls += 1
        return torch.stack(images).float().div(255).mul(.5).add(.25)


class Writer:
    def __init__(self, cancel=None):
        self.frames = []
        self.cancel = cancel
    def write(self, frame, pts, **kwargs):
        assert frame.device.type == "mps" and frame.dtype == torch.uint8
        self.frames.append((frame.cpu().clone(), pts))
        if self.cancel is not None:
            self.cancel.set()
    def after_write(self, n):
        pass


def no_workers():
    assert not {t.name for t in threading.enumerate()} & {
        "DecodeDetect", "PrimaryRestore", "SecondaryRestore", "BlendEncode", "VramOffloader"}


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=96x64:rate=12:duration=3", "-c:v", "libx264",
                    "-colorspace", "bt709", str(path)], check=True)
    return path


def positive_detector(frames, **kwargs):
    assert frames.device.type == "mps"
    return Detections([np.array([[20, 15, 70, 50]], dtype=np.float32) for _ in frames],
                      [torch.ones(1, 8, 8, device=frames.device, dtype=torch.bool) for _ in frames])


@pytest.mark.parametrize("mode", ["complete", "cancel", "detect-error", "primary-error", "secondary-error", "write-error", "memory-error", "poll-error", "startup-error"])
def test_workers_cleanup_and_error_propagation(mps, source, mode, monkeypatch):
    cancel = threading.Event()
    restorer = TensorRestorer(mps)
    detector = positive_detector
    writer = Writer(cancel if mode == "cancel" else None)
    failure = RuntimeError(mode)
    def fail(*args, **kwargs):
        raise failure
    if mode == "detect-error":
        detector = fail
    if mode == "primary-error":
        restorer.raw_process = fail
    if mode == "write-error":
        writer.write = fail
    if mode == "memory-error":
        monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 10 ** 15)
    poll = fail if mode == "poll-error" else None
    pipeline = pipeline_for(source, mps, detector, restorer, overlap=1)
    if mode == "secondary-error":
        pipeline.restoration_pipeline._run_secondary = fail
    if mode == "startup-error":
        original_start = threading.Thread.start
        def start(thread):
            if thread.name == "PrimaryRestore":
                raise failure
            return original_start(thread)
        monkeypatch.setattr(threading.Thread, "start", start)
    if mode in {"poll-error", "startup-error"}:
        with pytest.raises(RuntimeError, match=mode):
            run_restoration_pass(pipeline, get_video_meta_data(str(source)), writer, cancel,
                                 seek_ts=None, use_async_secondary=False, poll=poll)
    else:
        error = run_restoration_pass(pipeline, get_video_meta_data(str(source)), writer, cancel,
                                     seek_ts=None, use_async_secondary=False)
        if mode.endswith("error"):
            assert isinstance(error, RuntimeError)
            assert mode in str(error) or "safety budget exceeded" in str(error)
        else:
            assert error is None
        if mode == "complete":
            assert len(writer.frames) == 36 and restorer.calls > 1
            assert [pts for _, pts in writer.frames] == sorted({pts for _, pts in writer.frames})
            again = Writer()
            assert run_restoration_pass(pipeline, get_video_meta_data(str(source)), again,
                                        threading.Event(), seek_ts=None, use_async_secondary=False) is None
            assert len(again.frames) == len(writer.frames)
            for (actual, pts), (expected, expected_pts) in zip(again.frames, writer.frames):
                assert pts == expected_pts
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if mode == "cancel":
            assert cancel.is_set() and len(writer.frames) < 36
    no_workers()


def test_memory_monitor_uses_unified_budget_and_never_offloads(mps, monkeypatch, caplog):
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 24 * 1024 ** 3)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(total=32 * 1024 ** 3, available=20 * 1024 ** 3))
    buffer = BlendBuffer(mps)
    buffer.offloadable_results = Mock(side_effect=AssertionError("unsafe tensor mutation"))
    monitor = VramOffloader(mps, buffer, {})
    assert monitor._threshold == 16 * 1024 ** 3 - 750 * 1024 ** 2
    assert monitor._offload_device_type == "mps" and monitor._offload(1) == 0
    assert "budget is not free VRAM" in memory_summary(mps)
    monitor._dump_stall_diagnostics(31)
    assert "MPS unified memory" in caplog.text
    monitor.start()
    time.sleep(.15)
    monitor.stop()
    assert monitor.stats.sample_count > 0 and monitor.stats.offload_count == 0
    assert not monitor.errors


def test_mps_limits_preserve_explicit_clip_semantics(mps, source):
    p = pipeline_for(source, mps, positive_detector, TensorRestorer(mps))
    for field, value in [("batch_size", 3), ("max_clip_size", 33)]:
        old = getattr(p, field)
        setattr(p, field, value)
        with pytest.raises(ValueError, match="MPS pipeline requires"):
            run_restoration_pass(p, None, Writer(), threading.Event(), seek_ts=None, use_async_secondary=False)
        setattr(p, field, old)
    no_workers()


def test_real_checkpoints_threaded_pass_and_memory_plateau(mps, tmp_path):
    weights = os.environ.get("JASNA_TEST_MODEL_WEIGHTS_DIR")
    if not weights:
        pytest.skip("set JASNA_TEST_MODEL_WEIGHTS_DIR for real checkpoints")
    from jasna.mosaic.detection_registry import build_detection_model, rfdetr_model_config
    from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
    source = tmp_path / "positive.mp4"
    fixture = Path(__file__).resolve().parents[1] / "assets/test_clip1_1080p.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "4", "-i", str(fixture),
                    "-frames:v", "8", "-c:v", "libx264", "-crf", "12", "-colorspace", "bt709", str(source)], check=True)
    config = rfdetr_model_config("rfdetr-v6")
    model = build_detection_model("rfdetr-v6", Path(weights) / "rfdetr-v6.pt", batch_size=2,
                                 device=mps, score_threshold=config.score_threshold, fp16=False)
    restorer = BasicvsrppMosaicRestorer(str(Path(weights) / "lada_mosaic_restoration_model_generic_v1.2.pth"), mps, 4, False, False)
    detections, contracts = [], []
    def detect(*args, **kwargs):
        result = model(*args, **kwargs)
        detections.extend(len(b) for b in result.boxes_xyxy)
        return result
    raw = restorer.raw_process
    def restore(images):
        result = raw(images)
        assert result.shape == (len(images), 3, 256, 256)
        assert result.device.type == "mps" and result.dtype == torch.float32
        assert torch.isfinite(result).all().item()
        assert (result - torch.stack(images).float().div(255)).abs().mean().item() > 1e-4
        contracts.append(tuple(result.shape))
        return result
    restorer.raw_process = restore
    p = pipeline_for(source, mps, detect, restorer, overlap=1)
    p.max_detection_gap = 1  # exercise temporal clips across a missed detection
    meta = get_video_meta_data(str(source))
    root = Path(os.environ.get("JASNA_TEST_VIDEO_OUTPUT_DIR", tmp_path))
    root.mkdir(parents=True, exist_ok=True)
    allocated, drivers, rss = [], [], []
    started = time.perf_counter()
    try:
        for repeat in range(5):
            dst = root / f"pipeline-mps-{repeat}.mp4"
            writer = _OfflineFrameWriter(VideoEncoder(str(dst), mps, meta, codec="h264",
                                         encoder_settings={"crf": 18, "preset": "fast"}), [time.monotonic()])
            try:
                error = run_restoration_pass(p, meta, writer, threading.Event(), seek_ts=None, use_async_secondary=False)
                assert error is None
            finally:
                writer.close()
            no_workers()
            with av.open(str(dst)) as c:
                decoded = list(c.decode(video=0))
                assert len(decoded) == 8
                assert all(f.width == 1920 and f.height == 1080 for f in decoded)
                assert all(a.pts < b.pts for a, b in zip(decoded, decoded[1:]))
            probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
                                    "stream=codec_name,width,height,nb_read_frames,duration", "-of", "json", str(dst)],
                                   check=True, capture_output=True, text=True)
            stream = json.loads(probe.stdout)["streams"][0]
            assert stream["codec_name"] == "h264"
            assert (stream["width"], stream["height"], int(stream["nb_read_frames"])) == (1920, 1080, 8)
            assert float(stream["duration"]) == pytest.approx(8 / 30, abs=1e-5)
            print(f"\nffprobe {dst}: {probe.stdout}")
            allocated.append(torch.mps.current_allocated_memory())
            drivers.append(torch.mps.driver_allocated_memory())
            rss.append(psutil.Process().memory_info().rss)
        assert sum(detections) > 0 and contracts, "must restore positive real detections"
        assert max(shape[0] for shape in contracts) >= 2, "real temporal inference required"
        # Warm-up/cache growth has a fixed conservative ceiling, not a timing assertion.
        assert allocated[-1] <= allocated[0] + 64 * 1024 ** 2
        assert drivers[-1] <= drivers[0] + 512 * 1024 ** 2
        assert rss[-1] <= rss[0] + 512 * 1024 ** 2
        print(f"\ndetections={detections}; temporal_shapes={contracts}; allocated={allocated}; driver={drivers}; rss={rss}; seconds={time.perf_counter()-started:.3f}")
    finally:
        model.close()
        restorer.close()
        torch.mps.empty_cache()


def test_crop_owns_storage_before_decoder_reuse(mps):
    from jasna.accelerator import execution_context
    from jasna.crop_buffer import extract_crop
    with execution_context(mps):
        frame = torch.full((3, 64, 96), 27, dtype=torch.uint8, device=mps)
        crop = extract_crop(frame, np.array([20, 15, 70, 50]), 64, 96)
        assert crop.crop.untyped_storage().data_ptr() != frame.untyped_storage().data_ptr()
        frame.fill_(99)
    assert torch.all(crop.crop == 27).item()


def test_execution_context_serializes_mps_workers(mps):
    from jasna.accelerator import execution_context
    entered, release, second = threading.Event(), threading.Event(), threading.Event()
    def first():
        with execution_context(mps):
            torch.ones(8, device=mps).mul_(2)
            entered.set()
            assert release.wait(3)
    def other():
        with execution_context(mps):
            second.set()
            torch.ones(8, device=mps).mul_(3)
    a, b = threading.Thread(target=first), threading.Thread(target=other)
    a.start()
    assert entered.wait(3)
    b.start()
    try:
        assert not second.wait(.05)
    finally:
        release.set()
        a.join(3)
        b.join(3)
    assert second.is_set() and not a.is_alive() and not b.is_alive()
