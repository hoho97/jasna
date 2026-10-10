"""Backend routing and precision policy; no real GPU or checkpoint required."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys

import pytest
import torch

from jasna.mosaic import rfdetr
from jasna.mosaic.rfdetr_torch_runner import RfDetrTorchRunner


def test_mps_routes_to_torch_without_engine_cache(monkeypatch):
    import jasna.mosaic.rfdetr_torch_runner as module
    runner = SimpleNamespace(
        input_names=["input"], input_dtypes={"input": torch.float32},
        output_names=["dets", "labels", "masks"], outputs={
            "dets": torch.empty(1, 200, 4), "labels": torch.empty(1, 200, 3),
            "masks": torch.empty(1, 200, 8, 8),
        }, close=Mock(),
    )
    construct = Mock(return_value=runner)
    forbidden = Mock(side_effect=AssertionError("TensorRT accessed on MPS"))
    monkeypatch.setattr(rfdetr, "is_amd_device", lambda _: False)
    monkeypatch.setattr(rfdetr, "is_apple_device", lambda _: True)
    monkeypatch.setattr(rfdetr, "get_onnx_tensorrt_engine_path", forbidden)
    monkeypatch.setattr(rfdetr, "TrtRunner", forbidden)
    monkeypatch.setattr(rfdetr, "ResizeNormalizer", lambda **_: SimpleNamespace(available=False))
    monkeypatch.setattr(module, "RfDetrTorchRunner", construct)
    model = rfdetr.RfDetrMosaicDetectionModel(
        weights_path=Path("rfdetr-v6.pt"), batch_size=2, device=torch.device("mps"),
        resolution=576, dynamic_batch=True, torch_variant="medium", fp16=True,
    )
    assert model.runner is runner
    assert model.input_dtype == torch.float32
    assert model.engine_path == Path("rfdetr-v6.pt")
    assert construct.call_args.kwargs["device"] == torch.device("mps")
    assert construct.call_args.kwargs["variant"] == "medium"
    forbidden.assert_not_called()
    model.close()
    runner.close.assert_called_once()


def test_mps_rejects_unmapped_legacy_variant(monkeypatch):
    monkeypatch.setattr(rfdetr, "is_amd_device", lambda _: False)
    monkeypatch.setattr(rfdetr, "is_apple_device", lambda _: True)
    with pytest.raises(RuntimeError, match="Legacy checkpoint variants cannot be inferred"):
        rfdetr.RfDetrMosaicDetectionModel(
            weights_path=Path("rfdetr-v5.pt"), batch_size=1, device=torch.device("mps"),
            resolution=768, dynamic_batch=False,
        )


@pytest.mark.parametrize("filename,error,match", [
    ("missing.pt", FileNotFoundError, "Supply an explicit model path"),
    ("model.onnx", ValueError, "requires .pt"),
    ("model.engine", ValueError, "requires .pt"),
])
def test_mps_compile_validates_checkpoint_without_compiling(monkeypatch, tmp_path, filename, error, match):
    monkeypatch.setattr(rfdetr, "is_apple_device", lambda _: True)
    path = tmp_path / filename
    if filename != "missing.pt":
        path.touch()
    with pytest.raises(error, match=match):
        rfdetr.compile_rfdetr_engine(path, torch.device("mps"), batch_size=1,
                                    resolution=576, dynamic_batch=True, fp16=True)
    valid = tmp_path / "model.pt"
    valid.touch()
    assert rfdetr.compile_rfdetr_engine(valid, torch.device("mps"), batch_size=1,
                                       resolution=576, dynamic_batch=True, fp16=True) == valid


@pytest.mark.parametrize("device,requested,expected", [
    ("mps", True, False), ("mps", False, False),
    ("cuda:0", True, True), ("cuda:0", False, False),
])
@pytest.mark.parametrize("eager", [False, True])
def test_runner_precision_policy(monkeypatch, tmp_path, device, requested, expected, eager):
    monkeypatch.setenv("JASNA_MPS_RFDETR_EAGER", "1" if eager else "0")
    import jasna.mosaic.rfdetr_torch_runner as module
    path = tmp_path / "model.pt"
    torch.save({"model": {"class_embed.weight": torch.ones(3, 256)}}, path)
    core = Mock()
    core.to.return_value = core
    core.eval.return_value = core
    optimize = Mock()
    context = SimpleNamespace(model=core, inference_model=core)
    wrapper = Mock(return_value=SimpleNamespace(model=context, optimize_for_inference=optimize))
    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(RFDETRSegMedium=wrapper))
    monkeypatch.setattr(module, "device_name", lambda _: "test")
    runner = RfDetrTorchRunner(path, [(2, 3, 576, 576)], torch.device(device),
                             fp16=requested, resolution=576, variant="medium")
    assert runner.fp16 is expected
    assert wrapper.call_args.kwargs["device"] == ("cpu" if device == "mps" else device)
    core.to.assert_called_once_with(torch.device(device))
    assert runner._exported is (device == 'mps' and not eager)
    if device == 'mps' and not eager:
        optimize.assert_called_once_with(compile=False, dtype=torch.float32, inplace=True)
    else:
        optimize.assert_not_called()


@pytest.mark.parametrize("filename,error,match", [
    ("missing.pt", FileNotFoundError, "Supply an explicit model path"),
    ("model.onnx", ValueError, "requires .pt"),
    ("model.engine", ValueError, "requires .pt"),
])
def test_direct_mps_detector_rejects_missing_or_foreign_weights(monkeypatch, tmp_path, filename, error, match):
    monkeypatch.setattr(rfdetr, "is_amd_device", lambda _: False)
    monkeypatch.setattr(rfdetr, "is_apple_device", lambda _: True)
    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace())
    path = tmp_path / filename
    if filename != "missing.pt":
        path.touch()
    with pytest.raises(error, match=match):
        rfdetr.RfDetrMosaicDetectionModel(
            weights_path=path, batch_size=1, device=torch.device("mps"),
            resolution=576, dynamic_batch=True, torch_variant="medium",
        )


def test_exported_runner_preserves_public_output_contract():
    runner = RfDetrTorchRunner.__new__(RfDetrTorchRunner)
    runner.device, runner.fp16, runner._exported = torch.device('cpu'), False, True
    tensors = (torch.rand(2, 200, 4), torch.rand(2, 200, 3), torch.rand(2, 200, 144, 144))
    runner._core = Mock(return_value=tensors)
    actual = runner.infer({'input': torch.zeros(2, 3, 576, 576)})
    assert set(actual) == {'dets', 'labels', 'masks'}
    for key, value in zip(('dets','labels','masks'), tensors):
        assert actual[key] is value
    runner.close()
    with pytest.raises(RuntimeError, match='closed'):
        runner.infer({'input': torch.empty(0)})
