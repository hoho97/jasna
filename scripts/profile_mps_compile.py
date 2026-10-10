"""Opt-in RF-DETR MPS compilation probes; never enables application compilation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import traceback
import time
import warnings
from collections import Counter
from unittest.mock import patch

import av
import torch
from torch._dynamo.utils import counters

from jasna.mosaic.detection_registry import build_detection_model, rfdetr_model_config
from scripts.profile_mps_models import error, timed


def main():
    program_start = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['inductor', 'jit'], default='inductor')
    parser.add_argument('--shape-compat', action='store_true', help='Experiment only: replace private shape op with static CPU tensor construction')
    parser.add_argument('--backbone-only', action='store_true', help='Compile only the backbone through the standard API; avoids transformer private shape op')
    args = parser.parse_args()
    if args.backbone_only and (args.shape_compat or args.backend != 'inductor'):
        parser.error('--backbone-only requires Inductor without --shape-compat')
    assert torch.backends.mps.is_built() and torch.backends.mps.is_available()
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '1'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = args.output.parent / (args.output.stem + '-inductor-cache')
    assert not cache.exists(), 'Use a fresh output/cache for cold compilation'
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(cache)
    os.environ['JASNA_MPS_RFDETR_EXPORT'] = '1'
    torch._dynamo.config.suppress_errors = False
    import torch._inductor.config as inductor_config
    inductor_config.compile_threads = 1
    checkpoint = args.weights / 'rfdetr-v6.pt'
    def checksum():
        with checkpoint.open('rb') as handle:
            return hashlib.file_digest(handle, 'sha256').hexdigest()
    digest = checksum()
    config = rfdetr_model_config('rfdetr-v6')
    model = build_detection_model('rfdetr-v6', checkpoint, batch_size=4,
                                  device=torch.device('mps'), score_threshold=config.score_threshold, fp16=False)
    images = []
    with av.open('assets/test_clip1_1080p.mp4') as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in (0, 120, 121, 122):
                images.append(torch.from_numpy(frame.to_ndarray(format='rgb24')).permute(2, 0, 1))
            if len(images) == 4:
                break
    report = dict(torch=torch.__version__, backend=args.backend, dynamic=False,
                  batch=4, precision='fp32', suppress_errors=False, compile_threads=1, weights_sha256=digest,
                  cache=str(cache), fixture_frames=[0,120,121,122])
    report['shape_compat'] = args.shape_compat
    report['backbone_only'] = args.backbone_only
    shape_patch = patch.object(torch, '_shape_as_tensor', lambda tensor: torch.tensor(tuple(tensor.shape), dtype=torch.int64))
    try:
        with torch.inference_mode():
            inputs = model._preprocess(torch.stack(images).to('mps'))
            reference, report['eager_export'] = timed(lambda: model.runner._core(inputs), 3)
            # A blocking 16-byte copy can inherit outstanding backbone work.
            # Separate that waiting from actual completed-GPU upload latency.
            shape = torch.tensor([[48, 48]], dtype=torch.int64)
            _, completed_upload = timed(lambda: shape.to('mps'), 3)
            queued_upload = []
            for _ in range(3):
                torch.mps.synchronize()
                start = time.perf_counter()
                features = model.runner._core.backbone(inputs)
                submitted = time.perf_counter()
                uploaded_shape = shape.to('mps')
                copied = time.perf_counter()
                torch.mps.synchronize()
                queued_upload.append(dict(backbone_submission_seconds=submitted-start,
                                          blocking_copy_seconds=copied-submitted))
                del features, uploaded_shape
            report['shape_upload_wait_probe'] = dict(bytes=16, completed_gpu=completed_upload,
                after_backbone=queued_upload, semantics='single-thread isolated; blocking upload may include prior GPU completion')
            if args.shape_compat:
                # Single-process experiment only; never installed by application.
                # Dynamic spatial shapes are deliberately not claimed here.
                shape_patch.start()
                compatible = model.runner._core(inputs)
                report['shape_compat_numerical'] = {name:error(a,r) for name,a,r in zip(['dets','labels','masks'],compatible,reference)}
                for a,r in zip(compatible,reference):
                    torch.testing.assert_close(a,r,rtol=0,atol=0)
            if args.backend == 'inductor':
                target = model.runner._core.backbone if args.backbone_only else model.runner._core
                explain_start = time.perf_counter()
                try:
                    explanation = torch._dynamo.explain(target)(inputs)
                    report['explain'] = dict(graph_count=explanation.graph_count,
                        graph_break_count=explanation.graph_break_count, op_count=explanation.op_count,
                        break_reasons=[str(reason) for reason in explanation.break_reasons])
                except Exception:
                    report['explain_error'] = traceback.format_exc()
                finally:
                    report['explain_seconds'] = time.perf_counter() - explain_start
                torch._dynamo.reset()
                counters.clear()
                if args.backbone_only:
                    compiled = copy.deepcopy(model.runner._core)
                    compiled.backbone = torch.compile(compiled.backbone, backend=args.backend, dynamic=False)
                else:
                    compiled = torch.compile(model.runner._core, backend=args.backend, dynamic=False)
            else:
                torch.mps.synchronize()
                started = time.perf_counter()
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter('always')
                    try:
                        compiled = torch.jit.trace(model.runner._core, inputs, check_trace=True, strict=True)
                        torch.mps.synchronize()
                    finally:
                        report['trace_seconds'] = time.perf_counter() - started
                        report['trace_warnings'] = sorted(set(str(w.message) for w in captured))
                report['trace_graph_nodes'] = dict(Counter(node.kind() for node in compiled.inlined_graph.nodes()))
                report['graph_breaks'] = 'not applicable to TorchScript tracing; warnings and partial batches recorded'
            attempt_start = time.perf_counter()
            try:
                actual, report['compiled'] = timed(lambda: compiled(inputs), 3)
            finally:
                report['compile_and_timing_attempt_seconds'] = time.perf_counter() - attempt_start
            report['numerical'] = {name:error(a,r) for name,a,r in zip(['dets','labels','masks'],actual,reference)}
            for a,r in zip(actual,reference):
                assert a.device.type == 'mps' and a.dtype == torch.float32 and a.shape == r.shape
                assert torch.isfinite(a).all().item()
            def selected(outputs):
                return model._postprocess(pred_boxes=outputs[0],pred_logits=outputs[1],pred_masks=outputs[2],
                    target_hw=(1080,1920),score_threshold=config.score_threshold,max_select=16)
            boxes,masks = selected(actual)
            ref_boxes,ref_masks = selected(reference)
            report['detections'] = [len(b) for b in boxes]
            report['reference_detections'] = [len(b) for b in ref_boxes]
            report['selected_box_shapes_match'] = all(b.shape == r.shape for b,r in zip(boxes,ref_boxes))
            report['selected_box_max_pixels'] = max((float(abs(b-r).max()) for b,r in zip(boxes,ref_boxes) if len(b)),default=0.) if report['selected_box_shapes_match'] else None
            report['masks_exact'] = all(m.shape==r.shape and torch.equal(m,r) for m,r in zip(masks,ref_masks))
            if args.backend in ('jit', 'inductor'):
                report['partial_batches'] = []
                for batch in (1, 2, 3):
                    row = dict(batch=batch)
                    try:
                        out, row['timing'] = timed(lambda: compiled(inputs[:batch]), 3)
                        base = model.runner._core(inputs[:batch])
                        for a,r in zip(out,base):
                            assert a.shape == r.shape and a.dtype == torch.float32 and a.device.type == 'mps'
                            assert torch.isfinite(a).all().item()
                        row['numerical'] = {name:error(a,r) for name,a,r in zip(['dets','labels','masks'],out,base)}
                        b,m = selected(out)
                        rb,rm = selected(base)
                        row['detections'] = [len(x) for x in b]
                        row['reference_detections'] = [len(x) for x in rb]
                        row['masks_exact'] = all(x.shape==y.shape and torch.equal(x,y) for x,y in zip(m,rm))
                    except Exception:
                        row['error'] = traceback.format_exc()
                    report['partial_batches'].append(row)
            report['result'] = 'COMPLETED'
    except Exception:
        report['result'], report['error'] = 'FAILED', traceback.format_exc()
    finally:
        shape_patch.stop()
        report['dynamo_counters'] = {str(name):dict(values) for name,values in counters.items()}
        report['total_seconds'] = time.perf_counter() - program_start
        report['weights_unchanged'] = checksum() == digest
        assert report['weights_unchanged']
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(report),flush=True)
        model.close()
        torch.mps.empty_cache()


if __name__ == '__main__':
    main()
