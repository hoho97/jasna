"""Reproducible MPS E2E benchmark; run from repository root with python -m scripts.profile_mps_pipeline.

Use --observe for stage/lock/transfer observation, omit for control. Both runs
check real model contracts and output decode. Use separate output directories.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
from unittest.mock import patch

import av
import psutil
import torch

from jasna.mps_profiling import WallProfile, observe_pipeline


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--observe', action='store_true')
    parser.add_argument('--detail-transfers', action='store_true', help='Attribute Tensor.to transfers by thread, dtype and shape')
    parser.add_argument('--eager-detector', action='store_true', help='Use original MPS forward as before/control')
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--clip', type=int, default=90)
    parser.add_argument('--overlap', type=int, default=8)
    args = parser.parse_args()
    if args.detail_transfers and not args.observe:
        parser.error('--detail-transfers requires --observe')
    os.environ['JASNA_MPS_RFDETR_EXPORT'] = '0' if args.eager_detector else '1'
    assert torch.backends.mps.is_built() and torch.backends.mps.is_available()
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '1'
    from jasna.main import main as cli
    from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel
    from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
    from jasna.restorer.restoration_pipeline import RestorationPipeline

    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    output = root / 'restored.mp4'
    if output.exists():
        raise FileExistsError(output)
    def save(name, value):
        temporary = root / (name + '.tmp')
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
        temporary.replace(root / name)
    checkpoints = [args.weights / 'rfdetr-v6.pt', args.weights / 'lada_mosaic_restoration_model_generic_v1.2.pth']
    hashes = {p.name: sha256(p) for p in checkpoints}
    command = ['jasna', '--device', 'mps', '--input', str(args.input), '--output', str(output),
               '--batch-size', str(args.batch), '--max-clip-size', str(args.clip),
               '--temporal-overlap', str(args.overlap), '--no-progress', '--log-level', 'info']
    env = dict(python=sys.version, torch=torch.__version__, mps_built=torch.backends.mps.is_built(),
               mps_available=torch.backends.mps.is_available(), platform=platform.platform(),
               machine=platform.machine(), command=command, input_sha256=sha256(args.input),
               weights=hashes, observed=args.observe, detail_transfers=args.detail_transfers,
               versions={p: importlib.metadata.version(p) for p in ('torchvision', 'av', 'rfdetr', 'numpy', 'mmengine')},
               media={k: os.environ.get(k) for k in ('JASNA_ENCODE_BACKEND', 'JASNA_DECODE_BACKEND', 'JASNA_MPS_RFDETR_EXPORT')},
               git=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
    save('environment.json', env)
    counts = dict(detection_calls=0, frames=0, positive_frames=0, detections=0,
                  restoration_calls=0, restoration_frames=0, restoration_kept_frames=0)
    original_detect = RfDetrMosaicDetectionModel.__call__
    original_restore = BasicvsrppMosaicRestorer.raw_process
    def detect(self, frames, **kwargs):
        result = original_detect(self, frames, **kwargs)
        counts['detection_calls'] += 1
        counts['frames'] += len(frames)
        counts['positive_frames'] += sum(len(b) > 0 for b in result.boxes_xyxy)
        counts['detections'] += sum(len(b) for b in result.boxes_xyxy)
        return result
    def restore(self, frames):
        result = original_restore(self, frames)
        assert result.shape == (len(frames), 3, 256, 256)
        assert result.device.type == 'mps' and result.dtype == torch.float32
        assert torch.isfinite(result).all().item()
        counts['restoration_calls'] += 1
        counts['restoration_frames'] += len(frames)
        return result
    original_primary = RestorationPipeline.prepare_and_run_primary
    def primary(*a, **kw):
        result = original_primary(*a, **kw)
        counts['restoration_kept_frames'] += max(0, min(result.frame_count, result.keep_end) - max(0, result.keep_start))
        return result
    profile = WallProfile()
    process = psutil.Process()
    samples, stop = [], threading.Event()
    start = time.perf_counter()
    def monitor():
        while not stop.wait(5):
            samples.append(dict(seconds=time.perf_counter()-start, rss=process.memory_info().rss,
                                host_available=psutil.virtual_memory().available, swap=psutil.swap_memory().used,
                                mps_allocated=torch.mps.current_allocated_memory(), driver=torch.mps.driver_allocated_memory()))
            save('progress.json', dict(**counts, memory=samples[-1], stages=profile.snapshot()))
    thread = threading.Thread(target=monitor, daemon=True)
    os.environ['JASNA_MODEL_WEIGHTS_DIR'] = str(args.weights)
    os.environ.pop('JASNA_MAIN_PID', None)
    thread.start()
    try:
        with patch.object(RfDetrMosaicDetectionModel, '__call__', detect), \
             patch.object(BasicvsrppMosaicRestorer, 'raw_process', restore), \
             patch.object(RestorationPipeline, 'prepare_and_run_primary', primary), \
             (observe_pipeline(profile, detail_transfers=args.detail_transfers) if args.observe else nullcontext()), patch.object(sys, 'argv', command):
            cli()
    finally:
        elapsed = time.perf_counter() - start
        stop.set()
        thread.join()
        save('measurements.json', dict(wall_seconds=elapsed, fps=counts['frames']/elapsed,
                                      counts=counts, stages=profile.snapshot(), memory=samples,
                                      semantics='inclusive/exclusive CPU thread wall; asynchronous submission is not GPU time; lock hold includes existing GPU completion; nested/cross-thread intervals must not be summed'))
    assert counts['positive_frames'] and counts['restoration_calls']
    decoded, previous = 0, None
    with av.open(str(output)) as container:
        for frame in container.decode(video=0):
            assert previous is None or frame.pts > previous
            previous = frame.pts
            decoded += 1
    assert decoded == counts['frames']
    probe = subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(output)], text=True)
    save('ffprobe.json', json.loads(probe))
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(output), '-f', 'null', '-'], check=True)
    assert hashes == {p.name: sha256(p) for p in checkpoints}
    save('validation.json', dict(result='PASS', frames=decoded, finite_fp32_mps=True, weights_unchanged=True))


if __name__ == '__main__':
    main()
