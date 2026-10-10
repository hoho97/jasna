"""Isolated real-checkpoint MPS precision/scaling experiments (no CLI policy changes)."""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
import json
import hashlib
import os
from pathlib import Path
import statistics
import time
from unittest.mock import patch

import av
import torch
import torch.nn.functional as F

from jasna.mps_profiling import WallProfile
from jasna.mosaic.detection_registry import build_detection_model, rfdetr_model_config
from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer


def timed(fn, runs):
    # Single-thread experiments only. These completions are NOT E2E observation.
    torch.mps.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.mps.synchronize()
    cold = time.perf_counter() - start
    durations = []
    for _ in range(runs):
        torch.mps.synchronize()
        start = time.perf_counter()
        result = fn()
        torch.mps.synchronize()
        durations.append(time.perf_counter() - start)
    return result, dict(cold_seconds=cold, warm_seconds=durations, median_seconds=statistics.median(durations))


def error(actual, reference):
    delta = actual.float().cpu() - reference.float().cpu()
    finite = bool(torch.isfinite(actual).all())
    return dict(finite=finite, max_abs=float(delta.abs().max()) if finite else None,
                rmse=float(delta.square().mean().sqrt()) if finite else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--clips', type=int, nargs='+', default=[2, 16, 45, 90])
    args = parser.parse_args()
    if args.runs < 1 or any(length < 1 for length in args.clips):
        parser.error('runs and clip lengths must be positive')
    assert torch.backends.mps.is_built() and torch.backends.mps.is_available()
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '1'
    fixture = Path('assets/test_clip1_1080p.mp4')
    images = []
    with av.open(str(fixture)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in (0, 120, 121, 122):
                images.append(torch.from_numpy(frame.to_ndarray(format='rgb24')).permute(2, 0, 1))
            if len(images) == 4:
                break
    frames = torch.stack(images).to('mps')
    report = dict(torch=torch.__version__, mps_available=True, fixture=str(fixture), frames=[0,120,121,122], detector=[], restorer=[])
    def checksum(path):
        with path.open('rb') as handle:
            return hashlib.file_digest(handle, 'sha256').hexdigest()
    checkpoints = [args.weights / name for name in ['rfdetr-v6.pt', 'lada_mosaic_restoration_model_generic_v1.2.pth', 'lada_mosaic_detection_model_v4_fast.pt'] if (args.weights / name).is_file()]
    report['weights'] = {path.name: checksum(path) for path in checkpoints}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    os.environ['JASNA_MPS_RFDETR_EXPORT'] = '0'
    config = rfdetr_model_config('rfdetr-v6')
    model = build_detection_model('rfdetr-v6', args.weights / 'rfdetr-v6.pt', batch_size=4,
                                  device=torch.device('mps'), score_threshold=config.score_threshold, fp16=False)
    detector_state = {k:v.detach().cpu().clone() for k,v in model.runner._core.state_dict().items()}
    import rfdetr.models.ops.functions.ms_deform_attn_func as sampling
    original_sample = sampling._bilinear_grid_sample
    sampling_inputs = []
    def capture(x, grid, **kw):
        if not sampling_inputs:
            sampling_inputs.append((x.detach().clone(), grid.detach().clone(), kw))
        return original_sample(x, grid, **kw)
    def native(x, grid, padding_mode='zeros', align_corners=False):
        return F.grid_sample(x, grid, mode='bilinear', padding_mode=padding_mode, align_corners=align_corners)
    try:
        with torch.inference_mode():
            for batch in (1, 2, 4):
                x = model._preprocess(frames[:batch])
                model.runner._core.float()
                model.runner._core.load_state_dict(detector_state)
                sampling_inputs.clear()
                with patch.object(sampling, '_bilinear_grid_sample', capture):
                    reference, baseline = timed(lambda: model._infer(x), args.runs)
                sx, sg, skw = sampling_inputs[0]
                gather_result, gather_time = timed(lambda: original_sample(sx,sg,**skw), args.runs)
                native_result, native_time = timed(lambda: native(sx,sg,**skw), args.runs)
                report.setdefault('sampling',[]).append(dict(batch=batch, input_shape=list(sx.shape), grid_shape=list(sg.shape), gather=gather_time, native=native_time, numerical=error(native_result,gather_result)))
                save()
                # Quantify the existing scan-only CPU area fallback separately
                # from detector forward and its outstanding GPU completion.
                per_query = reference['labels'].sigmoid().amax(-1)
                merged = ((reference['masks'] > 0) & (per_query > config.score_threshold)[:, :, None, None]).any(1, keepdim=True).float()
                host_mask, download = timed(lambda: merged.cpu(), args.runs)
                resized, resize = timed(lambda: F.interpolate(host_mask, size=(37,65), mode='area'), args.runs)
                returned, upload = timed(lambda: resized.to('mps'), args.runs)
                assert returned.device.type == 'mps' and torch.isfinite(returned).all().item()
                torch.testing.assert_close(returned.cpu(), resized, rtol=0, atol=0)
                report.setdefault('scan_cpu_mask_resize', []).append(dict(batch=batch, shape=list(merged.shape), target=[37,65], download=download, cpu_area_resize=resize, upload=upload, bytes=merged.numel()*merged.element_size()))
                save()
                ref_boxes, ref_masks = model._postprocess(pred_boxes=reference['dets'], pred_logits=reference['labels'],
                                                          pred_masks=reference['masks'], target_hw=(1080,1920),
                                                          score_threshold=config.score_threshold, max_select=16)
                exported = copy.deepcopy(model.runner._core)
                exported.export()
                exported_output, export_timing = timed(lambda: exported(x), args.runs)
                report.setdefault('export',[]).append(dict(batch=batch, baseline=baseline, timing=export_timing, numerical={k:error(v,reference[k]) for k,v in zip(['dets','labels','masks'],exported_output)}))
                del exported, exported_output
                save()
                for sampler in ('gather', 'native'):
                    for precision in ('fp32', 'autocast_fp16', 'fp16'):
                        row = dict(batch=batch, sampler=sampler, precision=precision, baseline=baseline)
                        try:
                            model.runner._core.float()
                            model.runner._core.load_state_dict(detector_state)
                            if precision == 'fp16':
                                model.runner._core.half()
                            with patch.object(sampling, '_bilinear_grid_sample', native if sampler=='native' else original_sample):
                                def infer():
                                    with torch.autocast('mps', dtype=torch.float16, enabled=precision=='autocast_fp16'):
                                        out = model.runner._core(x.half() if precision=='fp16' else x)
                                    return {key: out[src].float() for key,src in [('dets','pred_boxes'),('labels','pred_logits'),('masks','pred_masks')]}
                                actual, row['timing'] = timed(infer, args.runs)
                            row['numerical'] = {k: error(actual[k], reference[k]) for k in actual}
                            boxes, masks = model._postprocess(pred_boxes=actual['dets'], pred_logits=actual['labels'],
                                                              pred_masks=actual['masks'], target_hw=(1080,1920),
                                                              score_threshold=config.score_threshold, max_select=16)
                            row['detections'] = [len(b) for b in boxes]
                            row['reference_detections'] = [len(b) for b in ref_boxes]
                            row['selected_box_shapes_match'] = all(b.shape == r.shape for b,r in zip(boxes,ref_boxes))
                            row['selected_box_max_pixels'] = max((float(abs(b-r).max()) for b,r in zip(boxes,ref_boxes) if len(b)),default=0.) if row['selected_box_shapes_match'] else None
                            row['masks_exact'] = all(m.shape==r.shape and torch.equal(m,r) for m,r in zip(masks,ref_masks))
                        except Exception as exc:
                            row['error'] = repr(exc)
                        report['detector'].append(row)
                        save()
                        print(json.dumps(row), flush=True)
                model.runner._core.float()
                model.runner._core.load_state_dict(detector_state)
    finally:
        model.close()
        del model, frames
        torch.mps.empty_cache()
    restorer = BasicvsrppMosaicRestorer(str(args.weights/'lada_mosaic_restoration_model_generic_v1.2.pth'),
                                       torch.device('mps'), 90, False, False)
    # Same real positive-frame crop repeated with three adjacent frames, rather
    # than random pixels: isolate temporal/precision cost at production 256x256.
    restoration_state = {k:v.detach().cpu().clone() for k,v in restorer.model.state_dict().items()}
    crops = [F.interpolate(image[:,400:656,800:1056][None].float(), size=(256,256), mode='bilinear', align_corners=False)[0].to('mps')
             for image in images[1:]]
    try:
        with torch.inference_mode():
            for length in args.clips:
                clip = [crops[i%len(crops)] for i in range(length)]
                restorer.model.float()
                restorer.model.load_state_dict(restoration_state)
                restorer.input_dtype = torch.float32
                reference, baseline = timed(lambda: restorer.raw_process(clip), args.runs)
                reference = reference.cpu()
                profile = WallProfile()
                from jasna.models.basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet, SecondOrderDeformableAlignment
                with ExitStack() as patches:
                    for owner, method in [(BasicVSRPlusPlusNet,'compute_flow'),(BasicVSRPlusPlusNet,'propagate'),(BasicVSRPlusPlusNet,'upsample'),(SecondOrderDeformableAlignment,'forward')]:
                        original = getattr(owner,method)
                        def completed(*a, _fn=original, _label=method, **kw):
                            torch.mps.synchronize()
                            with profile.measure(_label):
                                result = _fn(*a,**kw)
                                torch.mps.synchronize()
                                return result
                        patches.enter_context(patch.object(owner,method,completed))
                    _, stage_timing = timed(lambda: restorer.raw_process(clip),1)
                report.setdefault('restorer_stages',[]).append(dict(length=length, timing=stage_timing, stages=profile.snapshot(), semantics='synchronized isolated intervals; inclusive nesting; instrumentation overhead relative to uninstrumented baseline'))
                save()
                for precision in ('fp32','autocast_fp16','fp16'):
                    row = dict(length=length, precision=precision, baseline=baseline)
                    try:
                        restorer.model.float()
                        restorer.model.load_state_dict(restoration_state)
                        restorer.input_dtype = torch.float32
                        if precision == 'fp16':
                            restorer.model.half()
                            restorer.input_dtype = torch.float16
                        def infer():
                            with torch.autocast('mps', dtype=torch.float16, enabled=precision=='autocast_fp16'):
                                return restorer.raw_process(clip)
                        actual, row['timing'] = timed(infer, args.runs)
                        row['shape'], row['dtype'], row['device'] = list(actual.shape), str(actual.dtype), str(actual.device)
                        row['numerical'] = error(actual,reference)
                        row['psnr_db'] = -20 * __import__('math').log10(max(row['numerical']['rmse'],1e-12)) if row['numerical']['finite'] else None
                    except Exception as exc:
                        row['error'] = repr(exc)
                    report['restorer'].append(row)
                    save()
                    print(json.dumps(row), flush=True)
                del actual, reference
                torch.mps.empty_cache()
            # ladamac describes batching equal-length clips. Probe N independently
            # at the existing clip limit; do not alter production scheduling/caps.
            restorer.model.float()
            restorer.model.load_state_dict(restoration_state)
            restorer.input_dtype = torch.float32
            length = min(90, max(args.clips))
            inputs = torch.stack([crops[i % len(crops)] for i in range(length)]).div(255)
            clips = torch.stack([inputs.roll(shifts=i * 8, dims=-1) for i in range(3)])
            references = torch.cat([restorer.model(inputs=clip[None]).cpu() for clip in clips])
            for batch in (1, 2, 3):
                torch.mps.empty_cache()
                memory_before = dict(current=torch.mps.current_allocated_memory(), driver=torch.mps.driver_allocated_memory())
                actual, timing = timed(lambda: restorer.model(inputs=clips[:batch]), args.runs)
                numerical = error(actual, references[:batch])
                assert numerical['finite'] and actual.device.type == 'mps'
                assert actual.dtype == torch.float32 and actual.shape == (batch, length, 3, 256, 256)
                report.setdefault('restorer_batching', []).append(dict(
                    batch=batch, length=length, timing=timing, numerical=numerical,
                    memory_before=memory_before, memory_after=dict(current=torch.mps.current_allocated_memory(), driver=torch.mps.driver_allocated_memory()),
                    frames_per_second=batch * length / timing['median_seconds'],
                    shape=list(actual.shape), dtype=str(actual.dtype), device=str(actual.device)))
                save()
                del actual
            del references, clips, inputs
    finally:
        restorer.close()
        torch.mps.empty_cache()
    # Detector-only decomposition, never a replacement for the RF-DETR MVP.
    yolo_weights = args.weights / 'lada_mosaic_detection_model_v4_fast.pt'
    if yolo_weights.is_file():
        yolo = build_detection_model('lada-yolo-v4', yolo_weights, batch_size=4, device=torch.device('mps'), score_threshold=.25, fp16=False)
        uploaded = torch.stack(images).to('mps')
        try:
            with torch.inference_mode():
                for batch in (1,2,4):
                    preprocessed, _ = yolo._preprocess(uploaded[:batch])
                    raw, forward = timed(lambda: yolo._forward_raw(preprocessed), args.runs)
                    assert raw[0].device.type == raw[1].device.type == 'mps'
                    assert raw[0].dtype == raw[1].dtype == torch.float32
                    assert torch.isfinite(raw[0]).all().item() and torch.isfinite(raw[1]).all().item()
                    detections, public = timed(lambda: yolo(uploaded[:batch], target_hw=(1080,1920)), args.runs)
                    assert len(detections.boxes_xyxy) == len(detections.masks) == batch
                    for boxes, masks in zip(detections.boxes_xyxy, detections.masks):
                        assert boxes.shape == (len(masks), 4) and torch.isfinite(torch.as_tensor(boxes)).all().item()
                        assert masks.device.type == 'mps' and masks.dtype == torch.bool
                    report.setdefault('yolo_decomposition',[]).append(dict(batch=batch, forward=forward, public=public, detections=[len(b) for b in detections.boxes_xyxy], pred_shape=list(raw[0].shape), proto_shape=list(raw[1].shape), precision='fp32', input_resolution=640, quality_equivalent_to_rfdetr=False))
                    save()
        finally:
            yolo.close()
            torch.mps.empty_cache()
    assert report['weights'] == {path.name: checksum(path) for path in checkpoints}
    save()


if __name__ == '__main__':
    main()
