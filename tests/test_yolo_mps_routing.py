from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from jasna.mosaic.yolo import YoloMosaicDetectionModel


def test_mps_pt_routes_to_autobackend_fp32_without_tensorrt(monkeypatch, tmp_path: Path):
    weights = tmp_path / "lada_mosaic_detection_model_v4_fast.pt"
    weights.touch()

    backend = MagicMock()
    backend.fp16 = False
    backend.names = {0: "mosaic"}
    backend.end2end = False
    backend.stride = torch.tensor([32.0])
    backend.eval.return_value = backend

    backend_cls = MagicMock(return_value=backend)
    forbidden_engine = MagicMock(side_effect=AssertionError("TensorRT path reached on MPS"))
    monkeypatch.setattr("jasna.mosaic.yolo.get_yolo_tensorrt_engine_path", forbidden_engine)
    monkeypatch.setattr(
        "jasna.mosaic.yolo.ResizeNormalizer",
        lambda **kwargs: SimpleNamespace(available=False),
    )
    monkeypatch.setattr("ultralytics.nn.autobackend.AutoBackend", backend_cls)

    model = YoloMosaicDetectionModel(
        model_path=weights,
        batch_size=1,
        device=torch.device("mps"),
        score_threshold=0.25,
        fp16=True,
    )

    assert model.runner is None
    assert model.fp16 is False
    assert model.input_dtype == torch.float32
    assert model._resizer is None
    backend_cls.assert_called_once_with(
        model=str(weights),
        device=torch.device("mps"),
        fp16=False,
        fuse=True,
        verbose=False,
    )
    backend.eval.assert_called_once()
    forbidden_engine.assert_not_called()
