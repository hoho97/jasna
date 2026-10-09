"""Real libx264/mux tests on CPU and Apple MPS, without global op fallback."""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import av
import pytest
import torch
from av.video.reformatter import ColorRange

from jasna.accelerator import AcceleratorVendor
from jasna.media.encoder_settings import validate_encoder_settings
from jasna.media.probe import get_video_meta_data
from jasna.media.video_decoder import VideoReader
from jasna.media.video_encoder import VideoEncoder, resolve_encoder_options
from jasna.os_utils import resolve_executable


@pytest.fixture(params=['cpu', 'mps'])
def device(request):
    if request.param == 'mps' and not torch.backends.mps.is_available():
        pytest.skip('requires real Apple MPS')
    return torch.device(request.param)


def source(tmp_path, audio='aac', color_range='tv'):
    path = tmp_path / ('source.mkv' if audio == 'wmav2' else 'source.mp4')
    subprocess.run([
        resolve_executable('ffmpeg'), '-y', '-v', 'error',
        '-f', 'lavfi', '-i', 'testsrc2=size=96x64:rate=12:duration=2',
        '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-c:a', audio,
        '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
        '-color_range', color_range, '-metadata', 'title=encoder-test',
        '-metadata:s:a:0', 'language=eng', str(path),
    ], check=True)
    return path


@pytest.fixture
def forbid_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('software encoder entered a GPU hardware API')
    import jasna.media.video_encoder as module
    monkeypatch.setattr(module, 'HWAccel', forbidden)
    monkeypatch.setattr(module, 'new_stream', forbidden)
    for name in ['set_device', 'current_stream', 'Stream', 'Event', 'synchronize']:
        monkeypatch.setattr(torch.cuda, name, forbidden)
    original_empty = torch.empty
    def unpinned(*args, **kwargs):
        assert not kwargs.get('pin_memory', False)
        return original_empty(*args, **kwargs)
    monkeypatch.setattr(torch, 'empty', unpinned)


class CloseProbe:
    """Delegate all media work to real PyAV while observing resource cleanup."""
    def __init__(self, container):
        self.container = container
        self.close_calls = 0

    def __getattr__(self, name):
        return getattr(self.container, name)

    def close(self):
        self.close_calls += 1
        return self.container.close()


@pytest.mark.parametrize('audio', ['aac', 'wmav2'])
def test_real_roundtrip_mux(tmp_path, device, audio, forbid_cuda):
    src = source(tmp_path, audio)
    meta = get_video_meta_data(str(src))
    root = Path(os.environ.get('JASNA_TEST_VIDEO_OUTPUT_DIR', tmp_path))
    root.mkdir(parents=True, exist_ok=True)
    dst = root / f'encode-{device.type}-{audio}.mp4'
    with av.open(str(src)) as reference:
        source_times = [float(f.pts * f.time_base) for f in reference.decode(video=0)]
    started = time.perf_counter()
    with VideoReader(str(src), batch_size=5, device=device, metadata=meta) as reader, VideoEncoder(
        str(dst), device, meta, codec='h264', encoder_settings={'crf': 18, 'preset': 'fast'}
    ) as enc:
        assert enc.encoder_name == 'libx264'
        assert enc._cuda_ctx is None
        assert enc._host_yuv is None
        assert not enc._packed.is_pinned()
        for frames, timestamps in reader.frames():
            for frame, pts in zip(frames, timestamps):
                assert frame.device.type == device.type
                assert torch.isfinite(frame).all().item()
                enc.encode(frame, pts)
    assert not enc._encode_thread.is_alive()
    assert enc._options_validated
    with av.open(str(dst)) as output:
        video, aud = output.streams.video[0], output.streams.audio[0]
        assert video.codec_context.name == 'h264'
        assert video.codec_context.pix_fmt == 'yuv420p'
        assert (video.width, video.height) == (96, 64)
        assert video.base_rate == Fraction(12)
        assert video.codec_context.color_range == 1
        assert video.codec_context.colorspace == 1
        assert output.metadata['title'] == 'encoder-test'
        assert aud.codec_context.name == 'aac'
        assert aud.metadata['language'] == 'eng'
        assert float(aud.duration * aud.time_base) == pytest.approx(2, abs=.15)
        decoded = list(output.decode(video))
        assert len(decoded) == 24
        assert [float(f.pts * f.time_base) for f in decoded] == pytest.approx(source_times, abs=float(video.time_base))
        assert float(video.duration * video.time_base) == pytest.approx(2, abs=.01)
        assert abs(float((aud.start_time or 0) * aud.time_base) - float((video.start_time or 0) * video.time_base)) < .05
    probe = subprocess.run([resolve_executable('ffprobe'), '-v', 'error', '-count_frames', '-show_streams', '-show_format', '-of', 'json', str(dst)], capture_output=True, text=True, check=True)
    info = json.loads(probe.stdout)
    video = next(s for s in info['streams'] if s['codec_type'] == 'video')
    assert (video['codec_name'], video['width'], video['height'], int(video['nb_read_frames'])) == ('h264', 96, 64, 24)
    dst.with_suffix('.ffprobe.json').write_text(probe.stdout)
    subprocess.run([resolve_executable('ffmpeg'), '-v', 'error', '-xerror', '-i', str(dst), '-f', 'null', '-'], check=True)
    print(f'{device}: {audio}, 24 frames including decode/encode/probe/full decode in {time.perf_counter()-started:.3f}s')


@pytest.mark.parametrize('full_range', [False, True])
def test_reused_frame_vfr_color_and_delayed_flush(tmp_path, device, full_range, forbid_cuda):
    src = source(tmp_path)
    meta = replace(get_video_meta_data(str(src)), color_range=ColorRange.JPEG if full_range else ColorRange.MPEG)
    dst = tmp_path / 'reused.mp4'
    points = [0, 512, 1200, 1600, 2600, 3000, 3500, 4500, 5000, 5600, 6200, 7000]
    frame = torch.empty((3, 64, 192), device=device, dtype=torch.uint8)[:, :, ::2]
    with VideoEncoder(str(dst), device, meta, codec='h264', encoder_settings={'crf': 0}) as enc:
        for i, pts in enumerate(points):
            frame.fill_(i * 20)
            enc.encode(frame, pts)
        frame.fill_(255)
    with av.open(str(dst)) as output:
        v = output.streams.video[0]
        assert int(v.codec_context.color_range) == (2 if full_range else 1)
        frames = list(output.decode(v))
        assert len(frames) == len(points)
        assert [float(f.pts * f.time_base) for f in frames] == pytest.approx([float(p * meta.time_base) for p in points], abs=float(v.time_base))
        for i, f in enumerate(frames):
            # Check actual encoded luma, not just the VUI range tag.
            y = f.to_ndarray(format='yuv420p')[:64]
            expected = i * 20 if full_range else round(16 + i * 20 * 219 / 255)
            assert abs(float(y.mean()) - expected) <= 1


def test_worker_error_and_body_exception_close(tmp_path, device, monkeypatch):
    meta = get_video_meta_data(str(source(tmp_path)))
    enc = VideoEncoder(str(tmp_path/'bad.mp4'), device, meta, codec='h264', encoder_settings={})
    def fail(*args, **kwargs):
        raise RuntimeError('injected encode failure')
    monkeypatch.setattr(enc, '_encode_frame', fail)
    with pytest.raises(RuntimeError, match='injected encode failure'):
        with enc:
            enc.dst = CloseProbe(enc.dst)
            enc._src = CloseProbe(enc._src)
            enc.encode(torch.zeros((3,64,96), device=device, dtype=torch.uint8), 0)
    assert not enc._encode_thread.is_alive()
    assert enc.dst.close_calls == 1
    assert enc._src.close_calls == 1
    enc = VideoEncoder(str(tmp_path/'cancel.mp4'), device, meta, codec='h264', encoder_settings={})
    with pytest.raises(ValueError, match='cancel'):
        with enc:
            enc.dst = CloseProbe(enc.dst)
            enc._src = CloseProbe(enc._src)
            enc.encode(torch.zeros((3,64,96), device=device, dtype=torch.uint8), 0)
            raise ValueError('cancel')
    assert not enc._encode_thread.is_alive()
    assert enc.dst.close_calls == 1
    assert enc._src.close_calls == 1
    with VideoEncoder(str(tmp_path/'empty.mp4'), device, meta, codec='h264', encoder_settings={}):
        pass


@pytest.mark.parametrize('settings', [{'cq': 23}, {'rc': 'vbr'}, {'temporal-aq': 1}, {'qvbr_quality_level': 20}, {'crf': -1}, {'crf': 52}, {'crf': float('nan')}, {'crf': True}, {'preset': 'p5'}])
def test_software_settings_reject_hardware_semantics(settings):
    with pytest.raises(ValueError):
        validate_encoder_settings(settings, codec='h264', vendor=AcceleratorVendor.APPLE)


def test_unsupported_contracts(tmp_path):
    meta = get_video_meta_data(str(source(tmp_path)))
    for changes in [dict(is_10bit=True), dict(color_transfer='smpte2084'), dict(video_width=95)]:
        with pytest.raises(ValueError):
            VideoEncoder(str(tmp_path/'bad.mp4'), torch.device('cpu'), replace(meta, **changes), codec='h264', encoder_settings={})
    for codec, smart in [('hevc', False), ('av1', False), ('h264', True)]:
        with pytest.raises(ValueError):
            resolve_encoder_options(AcceleratorVendor.APPLE, codec, meta, {}, smart_fragment=smart)
    spec, options = resolve_encoder_options(AcceleratorVendor.APPLE, 'h264', meta, {}, smart_fragment=False)
    assert spec.backend == 'software'
    assert options == {'crf': '23', 'preset': 'medium'}


def test_parallel_mps_encoders(tmp_path):
    if not torch.backends.mps.is_available():
        pytest.skip('requires real Apple MPS')
    meta = get_video_meta_data(str(source(tmp_path)))
    def run(index):
        dst = tmp_path/f'parallel-{index}.mp4'
        with VideoEncoder(str(dst), torch.device('mps'), meta, codec='h264', encoder_settings={}) as enc:
            frame = torch.zeros((3,64,96), device='mps', dtype=torch.uint8)
            for i in range(24):
                frame.fill_(i*10)
                enc.encode(frame, i*1024)
        with av.open(str(dst)) as output:
            assert len(list(output.decode(video=0))) == 24
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(run, range(4)))


def test_container_cleanup_on_open_failure(tmp_path, monkeypatch):
    meta = get_video_meta_data(str(source(tmp_path)))
    real_open = av.open
    closed = []
    class SourceProxy:
        def __init__(self):
            self.container = real_open(meta.video_file)
        def __getattr__(self, name):
            return getattr(self.container, name)
        def close(self):
            closed.append(True)
            self.container.close()
    def fail_destination(file, *args, **kwargs):
        if str(file) == meta.video_file:
            return SourceProxy()
        raise OSError('destination cannot open')
    monkeypatch.setattr(av, 'open', fail_destination)
    with pytest.raises(OSError, match='destination cannot open'):
        with VideoEncoder(str(tmp_path/'bad.mp4'), torch.device('cpu'), meta, codec='h264', encoder_settings={}):
            pytest.fail('must fail before entering')
    assert closed == [True]


def test_worker_device_failure_drains_queue(tmp_path, device, monkeypatch):
    meta = get_video_meta_data(str(source(tmp_path)))
    def fail(*args):
        raise RuntimeError('worker device unavailable')
    monkeypatch.setattr('jasna.media.video_encoder.set_device', fail)
    enc = VideoEncoder(str(tmp_path/'bad-device.mp4'), device, meta, codec='h264', encoder_settings={})
    with pytest.raises(RuntimeError, match='worker device unavailable'):
        with enc:
            enc.encode(torch.zeros((3,64,96), dtype=torch.uint8, device=device), 0)
    assert not enc._encode_thread.is_alive()
    assert enc._encode_queue.unfinished_tasks == 0


def test_real_1080p_mps_encode(tmp_path, forbid_cuda):
    if not torch.backends.mps.is_available():
        pytest.skip('requires real Apple MPS')
    fixture = tmp_path/'1080p-source.mp4'
    subprocess.run([resolve_executable('ffmpeg'), '-y', '-v', 'error', '-f', 'lavfi',
                    '-i', 'color=size=1920x1080:rate=30:duration=1', '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', str(fixture)], check=True)
    meta = get_video_meta_data(str(fixture))
    meta = replace(meta, color_range=ColorRange.MPEG)
    root = Path(os.environ.get('JASNA_TEST_VIDEO_OUTPUT_DIR', tmp_path))
    root.mkdir(parents=True, exist_ok=True)
    dst = root/'encode-mps-1080p.mp4'
    started = time.perf_counter()
    with VideoEncoder(str(dst), torch.device('mps'), meta, codec='h264', encoder_settings={'preset': 'fast'}) as enc:
        frame = torch.empty((3,meta.video_height,meta.video_width), dtype=torch.uint8, device='mps')
        for i in range(30):
            frame.fill_(i*8)
            enc.encode(frame, round(i / 30 / meta.time_base))
        assert enc._packed.device.type == 'cpu'
    elapsed = time.perf_counter() - started
    with av.open(str(dst)) as output:
        frames = list(output.decode(video=0))
        assert len(frames) == 30
        assert (frames[0].width, frames[0].height) == (1920,1080)
        for i, f in enumerate(frames):
            expected = round(16 + i*8*219/255)
            assert abs(float(f.to_ndarray(format='yuv420p')[:1080].mean())-expected) <= 1
    subprocess.run([resolve_executable('ffmpeg'), '-v', 'error', '-xerror', '-i', str(dst), '-f', 'null', '-'], check=True)
    probe = subprocess.run([resolve_executable('ffprobe'), '-v', 'error', '-count_frames', '-show_streams', '-show_format', '-of', 'json', str(dst)], capture_output=True, text=True, check=True)
    info = json.loads(probe.stdout)
    video = next(s for s in info['streams'] if s['codec_type'] == 'video')
    assert (video['codec_name'], video['width'], video['height'], int(video['nb_read_frames'])) == ('h264',1920,1080,30)
    assert float(video['duration']) == pytest.approx(1,abs=.01)
    dst.with_suffix('.ffprobe.json').write_text(probe.stdout)
    print(f'MPS 1080p owned snapshots/CPU conversion/libx264: 30 frames in {elapsed:.3f}s, {30/elapsed:.1f} fps')


def test_optional_postprocessing_stays_at_host_boundary(tmp_path, device, forbid_cuda):
    meta = get_video_meta_data(str(source(tmp_path)))
    lut = tmp_path/'invert.cube'
    lut.write_text('LUT_1D_SIZE 2\n1 1 1\n0 0 0\n')
    dst = tmp_path/'lut.mp4'
    frame = torch.full((3,64,96), 40, dtype=torch.uint8, device=device)
    with VideoEncoder(str(dst), device, meta, codec='h264', encoder_settings={'crf': 0}, lut_path=lut, sharpen_strength=.5) as enc:
        assert enc._frame_device.type == 'cpu'
        enc.encode(frame, 0, apply_lut=False)
        enc.encode(frame, 1024, apply_lut=True)
    assert torch.equal(frame.cpu(), torch.full((3,64,96),40,dtype=torch.uint8))
    with av.open(str(dst)) as output:
        frames = list(output.decode(video=0))
        assert len(frames) == 2
        for f, rgb in zip(frames, [40,215]):
            assert abs(float(f.to_ndarray(format='yuv420p')[:64].mean()) - round(16+rgb*219/255)) <= 1


def test_real_codec_open_error_is_forwarded(tmp_path, device):
    meta = get_video_meta_data(str(source(tmp_path)))
    enc = VideoEncoder(str(tmp_path/'invalid-options.mp4'), device, meta, codec='h264', encoder_settings={'g': 'invalid'})
    with pytest.raises(RuntimeError, match='Failed to open h264 encoder .libx264.'):
        with enc:
            enc.encode(torch.zeros((3,64,96),dtype=torch.uint8,device=device),0)
    assert not enc._encode_thread.is_alive()
    assert enc._encode_queue.unfinished_tasks == 0
