"""Opt-in full Lada YOLO inference on Apple Silicon with release weights.

JASNA_TEST_MODEL_WEIGHTS_DIR must contain lada_mosaic_detection_model_v4_fast.pt.
The test deliberately keeps PYTORCH_ENABLE_MPS_FALLBACK disabled and verifies
that Ultralytics NMS receives MPS tensors rather than silently moving detector
post-processing to CPU.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import Mock

import av
import numpy as np
import pytest
import torch
import torchvision

from jasna.mosaic.detection_registry import build_detection_model
from jasna.tracking.clip_tracker import ClipTracker


pytestmark = [pytest.mark.mps_real, pytest.mark.model_required]


@pytest.fixture(scope="module")
def frames() -> torch.Tensor:
    value = os.environ.get("JASNA_TEST_MODEL_WEIGHTS_DIR")
    if not value:
        pytest.skip("set JASNA_TEST_MODEL_WEIGHTS_DIR for real MPS YOLO inference")
    assert torch.backends.mps.is_built() and torch.backends.mps.is_available()
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1"

    video = Path(__file__).resolve().parents[1] / "assets/test_clip1_1080p.mp4"
    images = []
    with av.open(str(video)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            # The same known fixture pair used by the RF-DETR real-MPS gate:
            # one clean frame and one mosaic-positive frame.
            if index in (0, 120):
                rgb = frame.to_ndarray(format="rgb24")
                images.append(torch.from_numpy(rgb).permute(2, 0, 1))
            if len(images) == 2:
                break
    assert len(images) == 2
    return torch.stack(images)


def _detector_tensors(model) -> list[torch.Tensor]:
    """Return tensors from the real Ultralytics model behind AutoBackend wrappers."""
    backend = getattr(model.model, "backend", None)
    candidates = (getattr(backend, "model", None), backend, model.model)
    for candidate in candidates:
        if isinstance(candidate, torch.nn.Module):
            tensors = list(candidate.parameters()) + list(candidate.buffers())
            if tensors:
                return tensors
    raise AssertionError("Ultralytics AutoBackend does not expose detector parameters or buffers")


def _assert_detector_tensors_on(model, device_type: str) -> None:
    tensors = _detector_tensors(model)
    for value in tensors:
        assert value.device.type == device_type
        if value.is_floating_point():
            assert value.dtype == torch.float32
            assert torch.isfinite(value).all().item()


def _mask_iou(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    actual = actual.cpu()
    expected = expected.cpu()
    intersection = (actual & expected).flatten(1).sum(1).float()
    union = (actual | expected).flatten(1).sum(1).float().clamp_min(1)
    return intersection / union


def test_real_lada_yolo_v4_mps_inference_matches_cpu_and_tracker(frames, monkeypatch):
    import jasna.mosaic.yolo as module

    weights = Path(os.environ["JASNA_TEST_MODEL_WEIGHTS_DIR"]) / "lada_mosaic_detection_model_v4_fast.pt"
    forbidden = Mock(side_effect=AssertionError("CUDA/TensorRT reached on Apple"))
    monkeypatch.setattr(module, "get_yolo_tensorrt_engine_path", forbidden)
    monkeypatch.setattr(module, "TrtRunner", forbidden)
    monkeypatch.setattr(torch.cuda, "current_stream", forbidden)
    monkeypatch.setattr("jasna.media.cuda_kernel.cuda_driver", forbidden)

    # Importing torchvision makes the currently selected Ultralytics NMS path use
    # torchvision.ops.nms. Record its input device to prove native MPS NMS is used.
    original_nms = torchvision.ops.nms
    nms_devices: list[str] = []

    def checked_nms(boxes, scores, iou_threshold):
        nms_devices.append(boxes.device.type)
        assert scores.device == boxes.device
        return original_nms(boxes, scores, iou_threshold)

    monkeypatch.setattr(torchvision.ops, "nms", checked_nms)

    mps_model = build_detection_model(
        "lada-yolo-v4",
        weights,
        batch_size=1,
        device=torch.device("mps"),
        score_threshold=0.25,
        fp16=True,
    )
    cpu_model = None
    try:
        assert mps_model.runner is None
        assert mps_model.fp16 is False
        assert mps_model.input_dtype == torch.float32
        assert mps_model._resizer is None
        _assert_detector_tensors_on(mps_model, "mps")

        uploaded = frames.to("mps")
        mps_results = []
        scan_results = []
        for frame in uploaded:
            one = frame.unsqueeze(0)
            x, _ = mps_model._preprocess(one)
            assert x.shape == (1, 3, 640, 640)
            assert x.dtype == torch.float32 and x.device.type == "mps"
            pred_raw, proto, nc = mps_model._forward_raw(x)
            assert nc > 0
            assert pred_raw.device.type == proto.device.type == "mps"
            assert pred_raw.dtype == proto.dtype == torch.float32
            assert torch.isfinite(pred_raw).all().item()
            assert torch.isfinite(proto).all().item()

            detections = mps_model(one, target_hw=tuple(frames.shape[-2:]))
            assert len(detections.boxes_xyxy) == len(detections.masks) == 1
            assert detections.boxes_xyxy[0].dtype == np.float32
            assert np.isfinite(detections.boxes_xyxy[0]).all()
            assert detections.masks[0].dtype == torch.bool
            assert detections.masks[0].device.type == "mps"
            assert len(detections.boxes_xyxy[0]) == len(detections.masks[0])
            mps_results.append(detections)

            score, mask = mps_model.scan_scores_masks(one, mask_hw=(90, 160))
            assert score.shape == (1,) and score.device.type == "mps"
            assert mask.shape == (1, 90, 160) and mask.device.type == "mps"
            assert mask.dtype == torch.bool
            scan_results.append((score.cpu(), mask.cpu()))

        counts = [len(result.boxes_xyxy[0]) for result in mps_results]
        assert any(count == 0 for count in counts), "fixture must exercise empty detections"
        assert any(count > 0 for count in counts), "fixture must exercise a real mosaic detection"
        assert nms_devices and set(nms_devices) == {"mps"}
        forbidden.assert_not_called()

        # Restore native NMS before creating/running the CPU reference. The CPU
        # model is the same immutable checkpoint and exercises the same public API.
        monkeypatch.setattr(torchvision.ops, "nms", original_nms)
        cpu_model = build_detection_model(
            "lada-yolo-v4",
            weights,
            batch_size=1,
            device=torch.device("cpu"),
            score_threshold=0.25,
            fp16=False,
        )
        _assert_detector_tensors_on(cpu_model, "cpu")

        cpu_results = []
        cpu_scans = []
        for frame in frames:
            one = frame.unsqueeze(0)
            cpu_results.append(cpu_model(one, target_hw=tuple(frames.shape[-2:])))
            score, mask = cpu_model.scan_scores_masks(one, mask_hw=(90, 160))
            cpu_scans.append((score, mask))

        for mps, cpu in zip(mps_results, cpu_results):
            actual_boxes, expected_boxes = mps.boxes_xyxy[0], cpu.boxes_xyxy[0]
            actual_masks, expected_masks = mps.masks[0], cpu.masks[0]
            assert len(actual_boxes) == len(expected_boxes)
            np.testing.assert_allclose(actual_boxes, expected_boxes, rtol=0, atol=1.0)
            if len(actual_masks):
                iou = _mask_iou(actual_masks, expected_masks)
                assert torch.all(iou >= 0.95), f"MPS/CPU mask IoU too low: {iou.tolist()}"

        for (mps_score, mps_mask), (cpu_score, cpu_mask) in zip(scan_results, cpu_scans):
            torch.testing.assert_close(mps_score, cpu_score, rtol=2e-3, atol=2e-3)
            intersection = (mps_mask & cpu_mask).flatten(1).sum(1).float()
            union = (mps_mask | cpu_mask).flatten(1).sum(1).float()
            # Empty scan masks match trivially; positive scans must overlap closely.
            iou = torch.where(union > 0, intersection / union.clamp_min(1), torch.ones_like(union))
            assert torch.all(iou >= 0.95), f"MPS/CPU scan mask IoU too low: {iou.tolist()}"

        # Feed the real detector contract through the tracker on-device. Boxes stay
        # CPU numpy arrays while boolean masks stay on MPS throughout tracking.
        positive = next(result for result in mps_results if len(result.boxes_xyxy[0]) > 0)
        empty = next(result for result in mps_results if len(result.boxes_xyxy[0]) == 0)
        tracker = ClipTracker(max_clip_size=8)
        ended, active = tracker.update(0, positive.boxes_xyxy[0], positive.masks[0])
        assert not ended and active
        assert all(mask.device.type == "mps" for clip in tracker.active_clips.values() for mask in clip.masks)
        ended, active = tracker.update(1, empty.boxes_xyxy[0], empty.masks[0])
        assert ended and not active
    finally:
        if cpu_model is not None:
            cpu_model.close()
        mps_model.close()
        torch.mps.empty_cache()
