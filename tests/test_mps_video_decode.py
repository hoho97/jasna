"""Real PyAV decode on CPU/MPS; CUDA is forbidden, not mocked into success."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gc
import os
from pathlib import Path
import subprocess
import threading
import time

import av
import numpy as np
import pytest
import torch
from av.video.reformatter import ColorRange, Colorspace

from jasna.media import video_decoder as module
from jasna.media.probe import get_video_meta_data


@pytest.fixture(params=["cpu", "mps"])
def device(request):
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires real MPS")
    return torch.device(request.param)


@pytest.fixture(autouse=True)
def forbid_cuda_and_pinning(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("CPU/MPS decoder attempted CUDA, private streams or YUV staging")
    for name in ("set_device", "current_stream", "Stream", "stream", "synchronize"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    for name in ("_ValiFrameSource", "_cuda_hwaccel", "_create_blocking_cuda_stream",
                 "new_stream", "YuvToRgbConverter"):
        monkeypatch.setattr(module, name, forbidden)
    real_empty = torch.empty
    def unpinned(*args, **kwargs):
        assert not kwargs.get("pin_memory", False)
        return real_empty(*args, **kwargs)
    monkeypatch.setattr(torch, "empty", unpinned)


@pytest.fixture
def video(tmp_path):
    # Non-aligned width exercises RGB24 row padding; B-frames exercise EOF flush.
    path = tmp_path / "bframes.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=12, options={"bf": "2", "g": "12"})
        stream.width, stream.height, stream.pix_fmt = 100, 50, "yuv420p"
        for i in range(17):
            data = np.empty((50, 100, 3), dtype=np.uint8)
            data[:] = (20 + i * 10, 160 - i * 5, 40 + i * 8)
            frame = av.VideoFrame.from_ndarray(data, format="rgb24")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path, get_video_meta_data(str(path))


def reference(path, metadata):
    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
    full_range = metadata.color_range == ColorRange.JPEG
    pixels = [torch.from_numpy(f.to_ndarray(
        format="rgb24", src_colorspace=metadata.color_space,
        dst_colorspace=metadata.color_space,
        src_color_range=ColorRange.JPEG if full_range else ColorRange.MPEG,
        dst_color_range=ColorRange.JPEG,
    )).permute(2, 0, 1) for f in frames]
    return torch.stack(pixels), [f.pts for f in frames]


@pytest.mark.parametrize("backend", ["auto", "pyav-sw"])
def test_h264_batches_pts_pixels_and_lifetime(video, device, monkeypatch, backend):
    path, metadata = video
    monkeypatch.setenv(module.DECODE_BACKEND_ENV, backend)
    expected, expected_pts = reference(path, metadata)
    retained, pts, snapshots = [], [], []
    with module.VideoReader(str(path), 6, device, metadata) as reader:
        assert reader._software_only and reader._decoder_ctx is None
        if device.type == "mps":
            assert reader.video_stream.codec_context.thread_count == 1
        for batch, group_pts in reader.frames():
            assert batch.device.type == device.type
            assert batch.dtype == torch.uint8 and batch.is_contiguous()
            assert batch.shape == (len(group_pts), 3, 50, 100)
            retained.append(batch)
            snapshots.append(batch.cpu().clone())
            pts.extend(group_pts)
    gc.collect()
    assert [len(b) for b in retained] == [6, 6, 5]
    assert pts == expected_pts
    actual = torch.cat([b.cpu() for b in retained])
    assert torch.equal(actual, torch.cat(snapshots))
    assert torch.equal(actual, expected)
    assert torch.isfinite(actual.float()).all()


@pytest.mark.parametrize("stride", [1, 3])
def test_seek_stride_nonzero_start_and_eof(video, device, monkeypatch, tmp_path, stride):
    path, _ = video
    monkeypatch.setenv(module.DECODE_BACKEND_ENV, "pyav-sw")
    shifted = tmp_path / "shifted.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-c", "copy",
                    "-output_ts_offset", "1.5", str(shifted)], check=True)
    metadata = get_video_meta_data(str(shifted))
    expected, pts = reference(shifted, metadata)
    seek = 0.5
    target = metadata.start_pts + round(seek / metadata.time_base)
    first = next(i for i, p in enumerate(pts) if p >= target)
    with module.VideoReader(str(shifted), 4, device, metadata, frame_stride=stride) as reader:
        assert reader.start_pts == metadata.start_pts > 0
        batches = list(reader.frames(seek_ts=seek))
        assert [p for _, group in batches for p in group] == pts[first::stride]
        assert torch.equal(torch.cat([b.cpu() for b, _ in batches]), expected[first::stride])
        assert list(reader.frames(seek_ts=10.0)) == []


def test_cancel_at_batch_boundary_and_reopen(video, device, monkeypatch):
    path, metadata = video
    monkeypatch.setenv(module.DECODE_BACKEND_ENV, "auto")
    cancel = threading.Event()
    with module.VideoReader(str(path), 4, device, metadata) as reader:
        container = reader.container
        original = reader._read_group
        calls = []
        def read_group(decoded):
            calls.append(True)
            return original(decoded)
        monkeypatch.setattr(reader, "_read_group", read_group)
        gen = reader.frames()
        first, first_pts = next(gen)
        cancel.set()
        if cancel.is_set():
            gen.close()
        assert len(calls) == 1  # No read-ahead after yield or on close.
    with pytest.raises(AssertionError, match="Container is not open"):
        list(container.demux())
    with module.VideoReader(str(path), 4, device, metadata) as reader:
        again, pts = next(reader.frames())
    assert pts == first_pts and torch.equal(again.cpu(), first.cpu())


def test_parallel_readers_handoff_to_consumer_thread(video, device, monkeypatch):
    path, metadata = video
    monkeypatch.setenv(module.DECODE_BACKEND_ENV, "pyav-sw")
    expected, expected_pts = reference(path, metadata)
    def read():
        retained = []
        for _ in range(8):
            with module.VideoReader(str(path), 4, device, metadata) as reader:
                retained.extend(reader.frames())
        return retained
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: read(), range(2)))
    # Workers and containers have exited before the consumer touches the tensors.
    for batches in results:
        assert [p for _, pts in batches for p in pts] == expected_pts * 8
        assert torch.equal(torch.cat([b.cpu() for b, _ in batches]), expected.repeat(8, 1, 1, 1))


@pytest.mark.parametrize("count", [1, module.CORRUPT_PACKET_TOLERANCE + 1])
def test_real_invalid_h264_packets(video, device, monkeypatch, count):
    path, metadata = video
    monkeypatch.setenv(module.DECODE_BACKEND_ENV, "pyav-sw")
    original = module.demux_video
    def corrupt_then_real(container, stream):
        for _ in range(count):
            yield av.Packet(b"not a valid h264 packet")
        yield from original(container, stream)
    monkeypatch.setattr(module, "demux_video", corrupt_then_real)
    with module.VideoReader(str(path), 4, device, metadata) as reader:
        container = reader.container
        # Route injected real packets through the actual FFmpeg H.264 context.
        reader._decoder_ctx = reader.video_stream.codec_context
        if count > module.CORRUPT_PACKET_TOLERANCE:
            with pytest.raises(module.VideoDecodeError, match="too many consecutive corrupt"):
                list(reader.frames())
        else:
            batches = list(reader.frames())
            assert sum(len(pts) for _, pts in batches) == 17
    with pytest.raises(AssertionError, match="Container is not open"):
        list(container.demux())


@pytest.mark.parametrize("transfer", ["smpte2084", "arib-std-b67"])
def test_hdr_metadata_rejected(video, device, monkeypatch, transfer):
    path, metadata = video
    monkeypatch.setenv(module.DECODE_BACKEND_ENV, "pyav-sw")
    with module.VideoReader(str(path), 4, device, replace(metadata, color_transfer=transfer)) as reader:
        with pytest.raises(module.VideoDecodeError, match="8-bit SDR"):
            list(reader.frames())


def test_hdr_frame_tags_rejected_without_metadata_flag(video, device, monkeypatch, tmp_path):
    path, _ = video
    tagged = tmp_path / "pq.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-c", "copy",
                    "-bsf:v", "h264_metadata=transfer_characteristics=16",
                    "-color_trc", "smpte2084", str(tagged)], check=True)
    with av.open(str(tagged)) as container:
        assert next(container.decode(video=0)).color_trc == 16
    metadata = replace(get_video_meta_data(str(tagged)), color_transfer="")
    monkeypatch.setenv(module.DECODE_BACKEND_ENV, "pyav-sw")
    with module.VideoReader(str(tagged), 4, device, metadata) as reader:
        with pytest.raises(module.VideoDecodeError, match="8-bit SDR"):
            list(reader.frames())


def test_real_10bit_rejected_even_without_metadata_flag(tmp_path, device, monkeypatch):
    path = tmp_path / "tenbit.mkv"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=12)
        stream.width, stream.height, stream.pix_fmt = 32, 32, "yuv420p10le"
        frame = av.VideoFrame(32, 32, "yuv420p10le")
        for plane in frame.planes:
            plane.update(bytes(plane.buffer_size))
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    metadata = replace(get_video_meta_data(str(path)), is_10bit=False)
    monkeypatch.setenv(module.DECODE_BACKEND_ENV, "pyav-sw")
    with module.VideoReader(str(path), 4, device, metadata) as reader:
        with pytest.raises(module.VideoDecodeError, match="8-bit SDR"):
            list(reader.frames())


@pytest.mark.parametrize("space", [Colorspace.ITU601, Colorspace.ITU709])
@pytest.mark.parametrize("full_range", [False, True])
def test_color_matrix_and_range(video, device, monkeypatch, space, full_range):
    path, metadata = video
    metadata = replace(metadata, color_space=space,
                       color_range=ColorRange.JPEG if full_range else ColorRange.MPEG)
    monkeypatch.setenv(module.DECODE_BACKEND_ENV, "pyav-sw")
    expected, _ = reference(path, metadata)
    with module.VideoReader(str(path), 4, device, metadata) as reader:
        batches = list(reader.frames())
    assert torch.equal(torch.cat([b.cpu() for b, _ in batches]), expected)


def test_real_1080p_decode_roundtrip(device, monkeypatch, tmp_path):
    """Diagnostic transcode only: does not exercise Jasna's future MPS encoder."""
    fixture = Path(__file__).resolve().parents[1] / "assets/test_clip1_1080p.mp4"
    if not fixture.exists():
        pytest.skip("requires repository video fixture")
    output_dir = Path(os.environ.get("JASNA_TEST_VIDEO_OUTPUT_DIR", tmp_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    source = output_dir / f"source-h264-{device.type}.mp4"
    output = output_dir / f"decode-roundtrip-{device.type}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(fixture), "-t", "1",
                    "-an", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt",
                    "yuv420p", str(source)], check=True)
    metadata = get_video_meta_data(str(source))
    monkeypatch.setenv(module.DECODE_BACKEND_ENV, "auto")
    n, seconds, seen_pts = 0, 0.0, []
    with (module.VideoReader(str(source), 4, device, metadata) as reader,
          av.open(str(source)) as reference_container,
          av.open(str(output), "w") as out):
        reference_frames = reference_container.decode(video=0)
        stream = out.add_stream("libx264", rate=metadata.video_fps_exact,
                                options={"preset": "ultrafast", "crf": "18"})
        stream.width, stream.height, stream.pix_fmt = 1920, 1080, "yuv420p"
        frames = reader.frames()
        while True:
            started = time.perf_counter()
            item = next(frames, None)
            seconds += time.perf_counter() - started
            if item is None:
                break
            batch, pts = item
            assert batch.device.type == device.type and batch.dtype == torch.uint8
            assert batch.shape == (len(pts), 3, 1080, 1920)
            # Real device operation on the returned tensor before host handoff.
            assert torch.isfinite(batch[:, :, ::32, ::32].float().mean()).item()
            host = batch.cpu()
            for pixels, timestamp in zip(host, pts):
                reference_frame = next(reference_frames)
                expected = torch.from_numpy(reference_frame.to_ndarray(
                    format="rgb24", src_colorspace=metadata.color_space,
                    dst_colorspace=metadata.color_space, src_color_range=ColorRange.MPEG,
                    dst_color_range=ColorRange.JPEG)).permute(2, 0, 1)
                assert timestamp == reference_frame.pts
                assert torch.equal(pixels, expected)
                frame = av.VideoFrame.from_ndarray(pixels.permute(1, 2, 0).numpy(), format="rgb24")
                frame.pts, frame.time_base = timestamp, metadata.time_base
                for packet in stream.encode(frame):
                    out.mux(packet)
                seen_pts.append(timestamp)
                n += 1
        assert next(reference_frames, None) is None
        for packet in stream.encode():
            out.mux(packet)
    assert n == metadata.num_frames == 30
    with av.open(str(output)) as container:
        decoded = list(container.decode(video=0))
    assert len(decoded) == n
    assert all((f.width, f.height) == (1920, 1080) for f in decoded)
    assert [f.pts * f.time_base for f in decoded] == [p * metadata.time_base for p in seen_pts]
    print(f"\n{device.type}: H.264 1080p {n} frames, decode/upload {seconds:.3f}s "
          f"({n / seconds:.1f} fps), exact RGB/PTS parity, output={output}")
