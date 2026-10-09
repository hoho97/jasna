"""Opt-in full RF-DETR inference on Apple Silicon, using trusted read-only weights.

JASNA_TEST_MODEL_WEIGHTS_DIR must contain rfdetr-v6.pt and rfdetr-vr-v1.pt.
No global PYTORCH_ENABLE_MPS_FALLBACK is allowed in this verification.
"""
import os
from pathlib import Path
from time import perf_counter
from unittest.mock import Mock

import av
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from jasna.mosaic.detection_registry import build_detection_model, rfdetr_model_config
from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel


@pytest.fixture(scope="module")
def frames():
    value = os.environ.get("JASNA_TEST_MODEL_WEIGHTS_DIR")
    if not value:
        pytest.skip("set JASNA_TEST_MODEL_WEIGHTS_DIR for real MPS inference")
    assert torch.backends.mps.is_built() and torch.backends.mps.is_available()
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1"
    video = Path(__file__).resolve().parents[1] / "assets/test_clip1_1080p.mp4"
    images = []
    with av.open(str(video)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in (0, 120):
                rgb = frame.to_ndarray(format="rgb24")
                images.append(torch.from_numpy(rgb).permute(2, 0, 1))
            if len(images) == 2:
                break
    assert len(images) == 2
    return torch.stack(images)


@pytest.mark.parametrize("name,classes,mask_size", [
    ("rfdetr-v6", 3, 144), ("rfdetr-vr-v1", 2, 192),
])
def test_real_checkpoint_inference_and_cpu_reference(frames, monkeypatch, name, classes, mask_size):
    import jasna.mosaic.rfdetr as module
    # Fail on forbidden boundaries, while leaving all Torch/model operations real.
    forbidden = Mock(side_effect=AssertionError("CUDA/TensorRT reached on Apple"))
    monkeypatch.setattr(module, "get_onnx_tensorrt_engine_path", forbidden)
    monkeypatch.setattr(module, "TrtRunner", forbidden)
    monkeypatch.setattr(torch.cuda, "current_stream", forbidden)
    monkeypatch.setattr("jasna.media.cuda_kernel.cuda_driver", forbidden)
    config = rfdetr_model_config(name)
    model = build_detection_model(
        name, Path(os.environ["JASNA_TEST_MODEL_WEIGHTS_DIR"]) / f"{name}.pt",
        batch_size=2, device=torch.device("mps"),
        score_threshold=config.score_threshold, fp16=True,
    )
    try:
        assert model.runner.fp16 is False
        assert model._resizer is None
        assert model.input_dtype == torch.float32
        tensors = list(model.runner._core.parameters()) + list(model.runner._core.buffers())
        assert tensors and all(t.device.type == "mps" for t in tensors)
        assert all(t.dtype == torch.float32 for t in tensors if t.is_floating_point())
        uploaded = frames.to("mps")
        x = model._preprocess(uploaded)
        assert x.shape == (2, 3, config.resolution, config.resolution)
        assert x.dtype == torch.float32 and x.device.type == "mps"
        assert torch.isfinite(x).all().item()
        # Verify both dynamic batch sizes on the same loaded model.
        torch.mps.synchronize()
        start = perf_counter()
        outputs = model._infer(x)
        torch.mps.synchronize()
        batch_seconds = perf_counter() - start
        single = model._infer(x[:1])
        expected = {"dets": (2, 200, 4), "labels": (2, 200, classes),
                    "masks": (2, 200, mask_size, mask_size)}
        for key, shape in expected.items():
            assert outputs[key].shape == shape
            assert outputs[key].dtype == torch.float32 and outputs[key].device.type == "mps"
            assert torch.isfinite(outputs[key]).all().item()
            torch.testing.assert_close(single[key][0], outputs[key][0], rtol=2e-3, atol=2e-3)
        boxes, masks = model._postprocess(
            pred_boxes=outputs["dets"], pred_logits=outputs["labels"], pred_masks=outputs["masks"],
            target_hw=tuple(frames.shape[-2:]), score_threshold=config.score_threshold, max_select=16,
        )
        assert len(boxes) == len(masks) == 2
        for b, m in zip(boxes, masks):
            assert b.ndim == 2 and b.shape[1] == 4 and b.dtype == np.float32
            assert np.isfinite(b).all()
            assert m.shape == (len(b), mask_size, mask_size)
            assert m.dtype == torch.bool and m.device.type == "mps"
        # Exercise the public detector and real whole-video scan, too.
        public = model(uploaded, target_hw=tuple(frames.shape[-2:]))
        for expected_box, expected_mask, actual_box, actual_mask in zip(boxes, masks, public.boxes_xyxy, public.masks):
            np.testing.assert_array_equal(actual_box, expected_box)
            torch.testing.assert_close(actual_mask, expected_mask)
        scores, scan = model.scan_scores_masks(uploaded, mask_hw=(37, 65))
        per_query = outputs["labels"].sigmoid().amax(-1)
        merged = ((outputs["masks"] > 0) & (per_query > config.score_threshold)[:, :, None, None]).any(1, keepdim=True).float()
        expected_scan = F.interpolate(merged.cpu(), size=(37, 65), mode="area")[:, 0] > 0
        torch.testing.assert_close(scan.cpu(), expected_scan)
        torch.testing.assert_close(scores, per_query.amax(-1))
        assert scores.device.type == scan.device.type == "mps"
        # Same checkpoint/core, CPU reference (no random second model).
        # Compare the consumer contract: floating point noise can reorder nearly
        # tied background proposals, so raw query indices are not CPU identities.
        cpu_input = x.cpu()
        model.runner._core.to("cpu")
        with torch.no_grad():
            cpu = model.runner._core(cpu_input)
        errors = {}
        for key, source in (("dets", "pred_boxes"), ("labels", "pred_logits"), ("masks", "pred_masks")):
            actual = outputs[key].cpu()
            errors[key] = (actual - cpu[source]).abs().max().item()
        cpu_boxes, cpu_masks = model._postprocess(
            pred_boxes=cpu["pred_boxes"], pred_logits=cpu["pred_logits"], pred_masks=cpu["pred_masks"],
            target_hw=tuple(frames.shape[-2:]), score_threshold=config.score_threshold, max_select=16,
        )
        cpu_scores = cpu["pred_logits"].sigmoid().amax(-1).amax(-1)
        torch.testing.assert_close(scores.cpu(), cpu_scores, rtol=0, atol=1e-4)
        for b, mask, cb, cm in zip(boxes, masks, cpu_boxes, cpu_masks):
            # Subpixel box parity and exact selected binary masks on these frames.
            np.testing.assert_allclose(b, cb, rtol=0, atol=.05)
            torch.testing.assert_close(mask.cpu(), cm, rtol=0, atol=0)
        if name == "rfdetr-v6":
            assert [len(b) for b in boxes] == [0, 1], "fixture must exercise a real positive detection"
            torch.testing.assert_close(outputs["dets"].cpu(), cpu["pred_boxes"], rtol=0, atol=1e-3)
            torch.testing.assert_close(outputs["labels"].sigmoid().cpu(), cpu["pred_logits"].sigmoid(), rtol=0, atol=1e-3)
        print(f"\n{name}: batch2_seconds={batch_seconds:.3f}; CPU max_abs_errors={errors}; detections={[len(b) for b in boxes]}")
        forbidden.assert_not_called()
    finally:
        model.close()
        torch.mps.empty_cache()


def test_mps_preprocess_postprocess_and_scan_match_cpu(frames, monkeypatch):
    # Nonempty and empty detections exercise long gather + boolean indexing.
    cpu_boxes = torch.tensor([[[.5, .5, .4, .2], [.2, .3, .1, .1], [.8, .6, .2, .1]]] * 2)
    cpu_logits = torch.tensor([[[5., -5.], [-5., 4.], [-8., -9.]], [[-8., -9.]] * 3])
    cpu_masks = torch.arange(2 * 3 * 12 * 16).reshape(2, 3, 12, 16).float() % 7 - 3
    outputs = {"dets": cpu_boxes, "labels": cpu_logits, "masks": cpu_masks}
    # The model inference itself is verified above; isolate exact scan/postprocess semantics here.
    detector = RfDetrMosaicDetectionModel.__new__(RfDetrMosaicDetectionModel)
    detector.device, detector.input_dtype = torch.device("mps"), torch.float32
    detector.resolution, detector._resizer, detector._normalization_cache = 576, None, {}
    actual = detector._preprocess(frames.to("mps"))
    detector.device = torch.device("cpu")
    reference = detector._preprocess(frames)
    torch.testing.assert_close(actual.cpu(), reference, atol=2e-6, rtol=2e-6)
    common = dict(target_hw=(1080, 1920), score_threshold=.35, max_select=3)
    cpu = detector._postprocess(pred_boxes=cpu_boxes, pred_logits=cpu_logits, pred_masks=cpu_masks, **common)
    gpu = {k: v.to("mps") for k, v in outputs.items()}
    mps = detector._postprocess(pred_boxes=gpu["dets"], pred_logits=gpu["labels"], pred_masks=gpu["masks"], **common)
    assert [len(b) for b in mps[0]] == [2, 0]
    for cb, cm, gb, gm in zip(cpu[0], cpu[1], mps[0], mps[1]):
        np.testing.assert_allclose(gb, cb, rtol=1e-6, atol=1e-4)
        assert gm.dtype == torch.bool and gm.device.type == "mps"
        torch.testing.assert_close(gm.cpu(), cm)
    detector._preprocess = lambda x: x
    detector._infer = lambda x: gpu
    detector.logits_out, detector.masks_out, detector.score_threshold = "labels", "masks", .35
    interpolation_devices = []
    original = F.interpolate
    def interpolate(x, **kwargs):
        interpolation_devices.append(x.device.type)
        return original(x, **kwargs)
    monkeypatch.setattr(F, "interpolate", interpolate)
    for hw, expected_device in [((5, 7), "cpu"), ((6, 8), "mps"), ((17, 19), "cpu")]:
        score, mask = detector.scan_scores_masks(frames, mask_hw=hw)
        assert interpolation_devices[-1] == expected_device
        per_query = cpu_logits.sigmoid().amax(-1)
        merged = ((cpu_masks > 0) & (per_query > .35)[:, :, None, None]).any(1, keepdim=True).float()
        expected = original(merged, size=hw, mode="area")[:, 0] > 0
        torch.testing.assert_close(score.cpu(), per_query.amax(-1))
        torch.testing.assert_close(mask.cpu(), expected)
        assert score.dtype == torch.float32 and score.device.type == "mps"
        assert mask.dtype == torch.bool and mask.device.type == "mps"


@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("padding_mode", ["zeros", "border"])
def test_external_rfdetr_mps_grid_matches_cpu(frames, align_corners, padding_mode):
    from rfdetr.utilities.tensors import _bilinear_grid_sample
    generator = torch.Generator().manual_seed(8)
    features = torch.randn(2, 32, 12, 16, generator=generator)
    # Include out-of-bounds points as well as long gather indices.
    grid = torch.rand(2, 9, 11, 2, generator=generator) * 3 - 1.5
    options = dict(align_corners=align_corners, padding_mode=padding_mode)
    reference = _bilinear_grid_sample(features, grid, **options)
    actual = _bilinear_grid_sample(features.to("mps"), grid.to("mps"), **options)
    assert actual.device.type == "mps"
    torch.testing.assert_close(actual.cpu(), reference, atol=1e-6, rtol=1e-6)


def test_mps_sdpa_and_layernorm_match_cpu(frames):
    generator = torch.Generator().manual_seed(8)
    q, k, v = [torch.randn(2, 4, 64, 32, generator=generator) for _ in range(3)]
    actual = F.scaled_dot_product_attention(q.to("mps"), k.to("mps"), v.to("mps"))
    reference = F.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(actual.cpu(), reference, rtol=1e-5, atol=1e-6)
    x = torch.randn(2, 64, 256, generator=generator)
    actual = F.layer_norm(x.to("mps"), (256,))
    torch.testing.assert_close(actual.cpu(), F.layer_norm(x, (256,)), rtol=1e-5, atol=1e-6)
