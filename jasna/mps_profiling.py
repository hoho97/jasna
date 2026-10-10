"""Opt-in wall-time observation for the MPS benchmark, never imported by jobs.

Nested intervals have inclusive and exclusive CPU-thread wall times. These are
NOT GPU kernel timings. In E2E only the existing execution lock completion is
used: no new synchronization or scheduling policy is introduced.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, ExitStack, nullcontext
from functools import wraps
import threading
import time
from unittest.mock import patch


class WallProfile:
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.local = threading.local()
        self.guard = threading.Lock()
        self.stats = defaultdict(lambda: dict(calls=0, inclusive_seconds=0., exclusive_seconds=0., units=0))

    @contextmanager
    def measure(self, label, units=0):
        stack = getattr(self.local, 'stack', None)
        if stack is None:
            stack = self.local.stack = []
        entry = [self.clock(), 0.]
        stack.append(entry)
        try:
            yield
        finally:
            elapsed = self.clock() - entry[0]
            stack.pop()
            if stack:
                stack[-1][1] += elapsed
            with self.guard:
                row = self.stats[label]
                row['calls'] += 1
                row['inclusive_seconds'] += elapsed
                row['exclusive_seconds'] += elapsed - entry[1]
                row['units'] += units

    def wrap(self, fn, label):
        @wraps(fn)
        def observed(*args, **kwargs):
            with self.measure(label):
                return fn(*args, **kwargs)
        return observed

    def snapshot(self):
        with self.guard:
            return {name: {**row, 'calls_per_second': row['calls'] / row['inclusive_seconds'] if row['inclusive_seconds'] else None, 'units_per_second': row['units'] / row['inclusive_seconds']
                          if row['inclusive_seconds'] else None}
                    for name, row in sorted(self.stats.items())}


class ObservedRLock:
    """Observe outermost acquisition; reentrant holds must not double count."""
    def __init__(self, lock, profile):
        self.lock, self.profile = lock, profile
        self.local = threading.local()

    def __enter__(self):
        depth = getattr(self.local, 'depth', 0)
        name = threading.current_thread().name
        if depth == 0:
            with self.profile.measure(f'lock.wait/{name}'):
                self.lock.acquire()
            self.local.start = self.profile.clock()
        else:
            self.lock.acquire()
        self.local.depth = depth + 1
        return self

    def __exit__(self, *exc):
        self.local.depth -= 1
        if self.local.depth == 0:
            elapsed = self.profile.clock() - self.local.start
            # Record independently of the method nesting tree.
            with self.profile.guard:
                row = self.profile.stats[f'lock.hold/{threading.current_thread().name}']
                row['calls'] += 1
                row['inclusive_seconds'] += elapsed
                row['exclusive_seconds'] += elapsed
        self.lock.release()


@contextmanager
def observe_pipeline(profile, *, detail_transfers=False):
    """Process-local patches restored even on error; profiling CLI only."""
    import torch
    from jasna import accelerator
    from jasna.media import video_decoder, video_encoder
    from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel
    from jasna.mosaic.rfdetr_torch_runner import RfDetrTorchRunner
    from jasna.models.basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet, SecondOrderDeformableAlignment
    from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
    from jasna.restorer.restoration_pipeline import RestorationPipeline
    from jasna.pipeline_timing import LoopTimer

    methods = [
        (RfDetrMosaicDetectionModel, '_preprocess', 'detector.preprocess'),
        (RfDetrMosaicDetectionModel, '_postprocess', 'detector.postprocess'),
        (RfDetrMosaicDetectionModel, 'scan_scores_masks', 'detector.scan_with_mask_resize'),
        (RfDetrTorchRunner, 'infer', 'detector.forward_submission'),
        (BasicvsrppMosaicRestorer, 'raw_process', 'restorer.forward_submission'),
        (RestorationPipeline, '_prepare_from_raw_crops', 'restorer.crop_preparation'),
        (BasicVSRPlusPlusNet, 'compute_flow', 'restorer.spynet_submission'),
        (BasicVSRPlusPlusNet, 'propagate', 'restorer.propagation_submission'),
        (BasicVSRPlusPlusNet, 'upsample', 'restorer.reconstruction_submission'),
        (SecondOrderDeformableAlignment, 'forward', 'restorer.alignment_submission'),
        (video_decoder.VideoReader, '_decode_packet', 'media.decode_packet'),
        (video_encoder.VideoEncoder, 'encode', 'media.snapshot_and_queue'),
        (video_encoder.VideoEncoder, '_to_yuv', 'media.rgb_yuv'),
        (video_encoder.VideoEncoder, '_encode_frame', 'media.encode_and_pack'),
        (video_encoder.VideoEncoder, '_mux_video', 'media.mux_video'),
        (video_encoder.VideoEncoder, '_pump_source_streams', 'media.audio_copy'),
        (accelerator, 'synchronize', 'pipeline.completion_wait'),
    ]
    with ExitStack() as patches:
        for owner, method, label in methods:
            fn = getattr(owner, method)
            replacement = profile.wrap(fn, label)
            # Preserve the staticmethod descriptor (postprocess).
            if isinstance(vars(owner).get(method), staticmethod):
                replacement = staticmethod(replacement)
            patches.enter_context(patch.object(owner, method, replacement))
        patches.enter_context(patch.object(accelerator, '_MPS_EXECUTION_LOCK',
                                           ObservedRLock(accelerator._MPS_EXECUTION_LOCK, profile)))
        original_measure = LoopTimer.measure
        @contextmanager
        def loop_measure(self, category):
            with profile.measure(f'loop/{self.name}/{category}'), original_measure(self, category):
                yield
        patches.enter_context(patch.object(LoopTimer, 'measure', loop_measure))
        original_to = torch.Tensor.to
        def transfer(tensor, *args, **kwargs):
            dest = kwargs.get('device', args[0] if args else None)
            if isinstance(dest, torch.Tensor):
                dest = dest.device
            if isinstance(dest, (str, torch.device)):
                target = torch.device(dest).type
                source = tensor.device.type
                if target != source:
                    with profile.measure(f'transfer/{source}->{target}', tensor.numel() * tensor.element_size()):
                        detail = (profile.measure(
                            f'transfer_detail/{threading.current_thread().name}/{source}->{target}/{tensor.dtype}/{tuple(tensor.shape)}',
                            tensor.numel() * tensor.element_size()) if detail_transfers else nullcontext())
                        with detail:
                            return original_to(tensor, *args, **kwargs)
            return original_to(tensor, *args, **kwargs)
        patches.enter_context(patch.object(torch.Tensor, 'to', transfer))
        original_cpu = torch.Tensor.cpu
        def cpu(tensor, *args, **kwargs):
            if tensor.device.type == 'mps':
                with profile.measure('transfer/mps->cpu', tensor.numel() * tensor.element_size()):
                    return original_cpu(tensor, *args, **kwargs)
            return original_cpu(tensor, *args, **kwargs)
        patches.enter_context(patch.object(torch.Tensor, 'cpu', cpu))
        original_copy = torch.Tensor.copy_
        def copy(tensor, source, *args, **kwargs):
            if tensor.device.type != source.device.type:
                with profile.measure(f'transfer/{source.device.type}->{tensor.device.type}', source.numel() * source.element_size()):
                    return original_copy(tensor, source, *args, **kwargs)
            return original_copy(tensor, source, *args, **kwargs)
        patches.enter_context(patch.object(torch.Tensor, 'copy_', copy))
        original_reformatter = video_decoder.VideoReformatter
        class ObservedReformatter:
            def __init__(self):
                self.original = original_reformatter()
            def reformat(self, *args, **kwargs):
                with profile.measure('media.decode_colorspace'):
                    return self.original.reformat(*args, **kwargs)
        patches.enter_context(patch.object(video_decoder, 'VideoReformatter', ObservedReformatter))
        yield
