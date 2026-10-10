"""VideoToolbox sessions, explicit bitrate semantics, and lossless fallback control."""
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import Mock

import av
import pytest
import torch

from test_mps_e2e import real_mps_weights

from jasna.accelerator import AcceleratorVendor
from jasna.backend_preflight import validate_backend_options
from jasna.main import build_parser, _validate_mps_cli_options, _resolve_cli_encoder_settings
from jasna.media.encoder_settings import validate_encoder_settings
from jasna.media.probe import get_video_meta_data
from jasna.media.video_decoder import VideoReader
from jasna.media.video_encoder import VideoEncoder, resolve_encoder_options
from jasna.os_utils import resolve_executable


@pytest.fixture
def source(tmp_path, request):
    params = getattr(request, "param", {})
    path = tmp_path / 'input.mp4'
    subprocess.run([resolve_executable('ffmpeg'), '-y', '-v', 'error',
                    '-f', 'lavfi', '-i', 'testsrc2=size=320x240:rate=12:duration=2',
                    '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2',
                    '-c:v', params.get('encoder', 'libx264'), '-pix_fmt', params.get('pix_fmt', 'yuv420p'), '-c:a', 'aac',
                    '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709',
                    '-color_range', params.get('range', 'tv'), '-output_ts_offset', str(params.get('offset', 0)), str(path)], check=True)
    return get_video_meta_data(str(path))


@pytest.fixture
def apple(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, 'is_built', lambda: True)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    monkeypatch.setenv('JASNA_ENCODE_BACKEND', 'videotoolbox')


@pytest.mark.parametrize('codec', ['h264', 'hevc'])
def test_policy_and_cli(apple, source, codec):
    settings = {'b': 2_000_000, 'g': 12}
    spec, options = resolve_encoder_options(AcceleratorVendor.APPLE, codec, source, settings, smart_fragment=False)
    assert spec.encoder_name == f'{codec}_videotoolbox'
    assert spec.backend == 'videotoolbox' and not spec.ten_bit
    assert options == {'allow_sw': '0', 'b': '2000000', 'g': '12'}
    assert _resolve_cli_encoder_settings('b=2000000,g=12', cq=None, codec=codec, vendor='apple') == settings
    args = build_parser().parse_args(['--device', 'mps', '--codec', codec])
    _validate_mps_cli_options(args)
    validate_backend_options('mps', decode_backend='pyav-hw')


@pytest.mark.parametrize('settings', [{'cq': 20}, {'crf': 20}, {'preset': 'p5'}, {'rc': 'cqp'},
                                     {'quality': 'quality'}, {'b': '2M'}, {'b': True}, {'b': 0},
                                     {'g': -1}, {'g': 1.5}, {'allow_sw': 1}])
def test_quality_options_are_never_silently_dropped(apple, settings):
    with pytest.raises(ValueError, match='VideoToolbox'):
        validate_encoder_settings(settings, codec='h264', vendor='apple')


def test_non_apple_policy_unchanged(apple, source):
    for vendor, name in [(AcceleratorVendor.NVIDIA, 'h264_nvenc'), (AcceleratorVendor.AMD, 'h264_amf')]:
        spec, options = resolve_encoder_options(vendor, 'h264', source, {}, smart_fragment=False)
        assert spec.encoder_name == name and 'allow_sw' not in options
    spec, options = resolve_encoder_options(AcceleratorVendor.CPU, 'h264', source, {}, smart_fragment=False)
    assert spec.encoder_name == 'libx264' and options == {'crf': '23', 'preset': 'medium'}


def test_contracts(apple, source, tmp_path, monkeypatch):
    for changes in [{'is_10bit': True}, {'color_transfer': 'smpte2084'}, {'color_transfer': 'arib-std-b67'}, {'video_width': 319}]:
        with pytest.raises(ValueError):
            VideoEncoder(str(tmp_path/'bad.mp4'), torch.device('mps'), replace(source, **changes), codec='hevc', encoder_settings={})
    with pytest.raises(ValueError):
        resolve_encoder_options(AcceleratorVendor.APPLE, 'av1', source, {}, smart_fragment=False)
    with pytest.raises(ValueError):
        resolve_encoder_options(AcceleratorVendor.APPLE, 'h264', source, {}, smart_fragment=True)
    monkeypatch.setenv('JASNA_ENCODE_BACKEND', 'typo')
    with pytest.raises(ValueError, match='JASNA_ENCODE_BACKEND'):
        validate_encoder_settings({}, codec='h264', vendor='apple')


def check_output(dst, codec, color_range='tv'):
    probe = json.loads(subprocess.check_output([resolve_executable('ffprobe'), '-v', 'error', '-count_frames',
                                               '-show_streams', '-show_format', '-of', 'json', str(dst)]))
    video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    audio = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
    assert (video['codec_name'], video['width'], video['height'], int(video['nb_read_frames'])) == (codec, 320, 240, 24)
    assert (video['pix_fmt'], video['color_space'], video['color_transfer'], video['color_primaries'], video['color_range']) == ('yuvj420p' if color_range == 'pc' else 'yuv420p', 'bt709', 'bt709', 'bt709', color_range)
    assert audio['codec_name'] == 'aac'
    assert abs(float(video['duration']) - 2) < .09
    assert abs(float(audio['duration']) - 2) < .09
    subprocess.run([resolve_executable('ffmpeg'), '-v', 'error', '-xerror', '-i', str(dst), '-f', 'null', '-'], check=True)
    with av.open(str(dst)) as container:
        times = [float(f.pts * f.time_base) for f in container.decode(video=0)]
    assert len(times) == 24 and all(a < b for a, b in zip(times, times[1:]))
    assert times[0] == 0 and abs(times[-1] - 23/12) < 1e-4
    dst.with_suffix('.ffprobe.json').write_text(json.dumps(probe, indent=2))


@pytest.mark.mps_real
@pytest.mark.parametrize('codec', ['h264', 'hevc'])
@pytest.mark.parametrize('backend', ['videotoolbox', 'auto'])
@pytest.mark.parametrize('source', [{}, {'range': 'pc'}], indirect=True)
def test_real_encode_session_and_mux(source, tmp_path, monkeypatch, codec, backend):
    monkeypatch.setenv('JASNA_ENCODE_BACKEND', backend)
    monkeypatch.setenv('JASNA_DECODE_BACKEND', 'pyav-sw')
    dst = Path(os.environ.get('JASNA_TEST_VIDEO_OUTPUT_DIR', str(tmp_path))) / f'{codec}-{backend}-{source.color_range.name}.mp4'
    dst.parent.mkdir(parents=True, exist_ok=True)
    mps = torch.device('mps')
    with VideoReader(source.video_file, 4, mps, source) as reader, VideoEncoder(
        str(dst), mps, source, codec=codec, encoder_settings={'b': 2_000_000, 'g': 12}
    ) as encoder:
        assert encoder.encoder_name == f'{codec}_videotoolbox'
        assert encoder.out_stream.codec_context.is_open
        assert encoder._frame_device.type == 'cpu'
        for batch, pts in reader.frames():
            assert batch.device.type == 'mps' and batch.dtype == torch.uint8
            for frame, pt in zip(batch, pts):
                encoder.encode(frame, pt)
    check_output(dst, codec, 'pc' if int(source.color_range) == 2 else 'tv')


@pytest.mark.mps_real
@pytest.mark.parametrize('codec', ['h264', 'hevc'])
def test_session_failure_fallback_preserves_frames_and_bitrate(source, tmp_path, monkeypatch, codec, caplog):
    monkeypatch.setenv('JASNA_ENCODE_BACKEND', 'auto')
    monkeypatch.setenv('JASNA_DECODE_BACKEND', 'pyav-sw')
    monkeypatch.setattr(VideoEncoder, '_open_videotoolbox_session', Mock(side_effect=RuntimeError('injected busy session')))
    dst = tmp_path / f'fallback-{codec}.mp4'
    mps = torch.device('mps')
    with VideoReader(source.video_file, 4, mps, source) as reader, VideoEncoder(
        str(dst), mps, source, codec=codec, encoder_settings={'b': 2_000_000, 'g': 12}
    ) as encoder:
        assert encoder.encoder_name == ('libx264' if codec == 'h264' else 'libx265')
        assert encoder.encoder_options == {'b': '2000000', 'g': '12'}
        for batch, pts in reader.frames():
            for frame, pt in zip(batch, pts):
                encoder.encode(frame, pt)
    assert 'injected busy session' in caplog.text and 'unchanged bitrate' in caplog.text
    check_output(dst, codec)


def test_strict_policy_does_not_fallback(apple, source, tmp_path, monkeypatch):
    monkeypatch.setattr(VideoEncoder, '_open_videotoolbox_session', Mock(side_effect=RuntimeError('injected busy session')))
    with pytest.raises(RuntimeError, match='injected busy session'):
        with VideoEncoder(str(tmp_path/'strict.mp4'), torch.device('mps'), source, codec='h264', encoder_settings={}):
            pytest.fail('should not open')


@pytest.mark.mps_real
@pytest.mark.parametrize('source', [{}, {'range': 'pc'}, {'encoder': 'libx265'}, {'encoder': 'libx265', 'range': 'pc'}, {'offset': 5}], indirect=True)
def test_real_decode_pts_seek_and_color(source, monkeypatch):
    mps = torch.device('mps')
    results = {}
    for backend in ['pyav-sw', 'pyav-hw']:
        monkeypatch.setenv('JASNA_DECODE_BACKEND', backend)
        with VideoReader(source.video_file, 4, mps, source) as reader:
            frames, pts = [], []
            for batch, times in reader.frames():
                assert batch.device.type == 'mps' and batch.dtype == torch.uint8
                assert tuple(batch.shape[1:]) == (3, 240, 320)
                frames.append(batch.cpu())
                pts.extend(times)
            results[backend] = (torch.cat(frames), pts)
            if backend == 'pyav-hw':
                assert reader._videotoolbox and reader.video_stream.codec_context.is_hwaccel
        with VideoReader(source.video_file, 3, mps, source, frame_stride=2) as reader:
            selected = [p for _, times in reader.frames(seek_ts=1) for p in times]
            assert selected == pts[12::2]
    sw, hw = results['pyav-sw'], results['pyav-hw']
    assert sw[1] == hw[1] and len(hw[1]) == 24
    # Normalize NV12 layout before RGB so chroma interpolation is identical.
    assert torch.equal(sw[0], hw[0])


@pytest.mark.mps_real
def test_first_frame_decode_failure_restarts_once(source, monkeypatch, caplog):
    monkeypatch.setenv('JASNA_DECODE_BACKEND', 'pyav-hw')
    original = VideoReader._decoded_frames_impl
    def fail_hardware(self, seek_ts):
        if self._videotoolbox:
            raise RuntimeError('injected decode session failure')
        yield from original(self, seek_ts)
    monkeypatch.setattr(VideoReader, '_decoded_frames_impl', fail_hardware)
    with VideoReader(source.video_file, 4, torch.device('mps'), source) as reader:
        pts = [p for _, times in reader.frames() for p in times]
        assert len(pts) == len(set(pts)) == 24 and not reader._videotoolbox
    assert 'injected decode session failure' in caplog.text and 'using software decode' in caplog.text


@pytest.mark.mps_real
def test_midstream_decode_failure_is_visible(source, monkeypatch):
    monkeypatch.setenv('JASNA_DECODE_BACKEND', 'pyav-hw')
    original = VideoReader._decoded_frames_impl
    def fail_after_frame(self, seek_ts):
        yield next(original(self, seek_ts))
        raise RuntimeError('injected midstream failure')
    monkeypatch.setattr(VideoReader, '_decoded_frames_impl', fail_after_frame)
    with VideoReader(source.video_file, 1, torch.device('mps'), source) as reader:
        stream = reader.frames()
        next(stream)
        with pytest.raises(RuntimeError, match='injected midstream failure'):
            next(stream)
        assert reader._videotoolbox


@pytest.mark.mps_real
def test_decoder_device_unavailable_uses_software(source, monkeypatch, caplog):
    import jasna.media.video_decoder as module
    monkeypatch.setenv('JASNA_DECODE_BACKEND', 'pyav-hw')
    monkeypatch.setattr(module, 'HWAccel', Mock(side_effect=RuntimeError('injected absent device')))
    with VideoReader(source.video_file, 4, torch.device('mps'), source) as reader:
        assert not reader._videotoolbox
        assert sum(len(pts) for _, pts in reader.frames()) == 24
    assert 'injected absent device' in caplog.text and 'using software decode' in caplog.text


@pytest.mark.mps_real
def test_missing_encoder_registration_uses_software(source, tmp_path, monkeypatch, caplog):
    import jasna.media.video_encoder as module
    monkeypatch.setenv('JASNA_ENCODE_BACKEND', 'auto')
    original = module.av.Codec
    def missing(name, mode):
        if name.endswith('_videotoolbox'):
            raise ValueError('injected absent codec')
        return original(name, mode)
    monkeypatch.setattr(module.av, 'Codec', missing)
    with VideoEncoder(str(tmp_path/'missing.mp4'), torch.device('mps'), source, codec='h264', encoder_settings={}) as encoder:
        assert encoder.encoder_name == 'libx264'
    assert 'injected absent codec' in caplog.text


@pytest.mark.mps_real
def test_encoder_midstream_failure_propagates(source, tmp_path, monkeypatch):
    monkeypatch.setenv('JASNA_ENCODE_BACKEND', 'auto')
    mps = torch.device('mps')
    with pytest.raises(RuntimeError, match='injected encode failure'):
        with VideoEncoder(str(tmp_path/'midstream.mp4'), mps, source, codec='h264', encoder_settings={}) as encoder:
            monkeypatch.setattr(encoder, '_encode_frame', Mock(side_effect=RuntimeError('injected encode failure')))
            encoder.encode(torch.zeros((3, 240, 320), dtype=torch.uint8, device=mps), 0)
    assert encoder.encoder_name == 'h264_videotoolbox'
    assert not encoder._encode_thread.is_alive()


@pytest.mark.mps_real
@pytest.mark.parametrize('source', [{'encoder': 'libx265', 'pix_fmt': 'yuv420p10le'}], indirect=True)
def test_real_10bit_decode_is_rejected(source, monkeypatch):
    from jasna.media.video_decoder import VideoDecodeError
    assert source.is_10bit
    monkeypatch.setenv('JASNA_DECODE_BACKEND', 'pyav-hw')
    with VideoReader(source.video_file, 1, torch.device('mps'), source) as reader:
        with pytest.raises(VideoDecodeError, match='8-bit SDR'):
            next(reader.frames())


@pytest.mark.mps_real
def test_partial_hardware_decode_closes(source, monkeypatch):
    monkeypatch.setenv('JASNA_DECODE_BACKEND', 'pyav-hw')
    with VideoReader(source.video_file, 1, torch.device('mps'), source) as reader:
        stream = reader.frames()
        batch, pts = next(stream)
        assert len(pts) == 1 and batch.device.type == 'mps'
        stream.close()


@pytest.mark.mps_real
@pytest.mark.model_required
@pytest.mark.parametrize('codec', ['h264', 'hevc'])
def test_real_cli_videotoolbox(real_mps_weights, tmp_path, monkeypatch, codec):
    from test_mps_e2e import verify_cli_detection_restoration_encode
    verify_cli_detection_restoration_encode(real_mps_weights, tmp_path, monkeypatch, 'videotoolbox', codec)
