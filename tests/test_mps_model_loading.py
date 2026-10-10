"""Weights policy tests plus opt-in real Apple Silicon checkpoint loading.

JASNA_TEST_MODEL_WEIGHTS_DIR points to a trusted, read-only release directory.
"""
from pathlib import Path
import os
from unittest.mock import Mock

import pytest
import torch

from jasna import engine_compiler, engine_paths
from jasna.model_weights import load_restoration_state_dict, load_rfdetr_checkpoint, validate_weights_path
from jasna.mosaic import detection_registry as registry


@pytest.fixture
def apple(monkeypatch):
    monkeypatch.setattr(registry, "is_apple_device", lambda: True)
    monkeypatch.setattr(registry, "is_amd_device", lambda: False)


def test_apple_discovery_and_explicit_paths(apple, monkeypatch, tmp_path):
    monkeypatch.setenv("JASNA_MODEL_WEIGHTS_DIR", str(tmp_path))
    for name in ("rfdetr-v6.pt", "rfdetr-v6.onnx", "lada_mosaic_detection_model_v4_fast.pt"):
        (tmp_path / name).touch()
    assert registry.rfdetr_weights_suffix() == ".pt"
    assert registry.discover_available_detection_models() == ["rfdetr-v6", "lada-yolo-v4"]
    assert registry.resolve_detection_model("rfdetr-v6", "", None)[1] == tmp_path / "rfdetr-v6.pt"
    custom = tmp_path / "custom.pt"
    custom.touch()
    assert registry.resolve_detection_model("rfdetr-v6", str(custom), None)[1] == custom
    assert registry.resolve_detection_model("lada-yolo-v4", str(custom), None)[1] == custom
    assert engine_paths.default_restoration_model_path("basicvsrpp") == tmp_path / "lada_mosaic_restoration_model_generic_v1.2.pth"


@pytest.mark.parametrize("filename,match", [("model.onnx", "requires .pt"), ("model.engine", "requires .pt"), ("model.onnx.enc", "encrypted"), ("missing.pt", "not found")])
def test_apple_rejects_incompatible_paths(apple, tmp_path, filename, match):
    path = tmp_path / filename
    if filename != "missing.pt":
        path.touch()
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        registry.resolve_detection_model("rfdetr-v6", str(path), None)


def test_directory_is_not_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        validate_weights_path(tmp_path, suffix=".pt", model="RF-DETR")


def test_cpu_loading_and_schema(tmp_path, monkeypatch):
    rf = tmp_path / "rf.pt"
    restoration = tmp_path / "restore.pth"
    torch.save({"model": {"class_embed.weight": torch.zeros(3, 256)}, "args": {}}, rf)
    torch.save({"weight": torch.ones(2, 3)}, restoration)
    original = torch.load
    calls = []
    def load(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "load", load)
    assert load_rfdetr_checkpoint(rf)["model"]["class_embed.weight"].device.type == "cpu"
    assert load_restoration_state_dict(restoration)["weight"].device.type == "cpu"
    assert calls == [{"map_location": "cpu", "weights_only": False}, {"map_location": "cpu", "weights_only": True}]
    torch.save({"model": torch.nn.Linear(3, 2)}, rf)
    with pytest.raises(ValueError, match="YOLO serialized model"):
        load_rfdetr_checkpoint(rf)
    torch.save({"bad": "not tensors"}, restoration)
    with pytest.raises(ValueError, match="tensor state_dict"):
        load_restoration_state_dict(restoration)
    rf.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="Cannot load checkpoint.*on CPU"):
        load_rfdetr_checkpoint(rf)


def test_mps_never_probes_or_compiles_engines(monkeypatch, tmp_path):
    import jasna.accelerator as accelerator
    monkeypatch.setattr(accelerator, "is_apple_device", lambda device: True)
    monkeypatch.setattr(accelerator, "is_nvidia_device", lambda device: False)
    forbidden = Mock(side_effect=AssertionError("TensorRT cache accessed on Apple"))
    monkeypatch.setattr(engine_paths, "get_onnx_tensorrt_engine_path", forbidden)
    monkeypatch.setattr(engine_compiler, "all_basicvsrpp_sub_engines_exist", forbidden)
    monkeypatch.setattr(engine_compiler.subprocess, "Popen", forbidden)
    path = tmp_path / "rfdetr-v6.pt"
    path.touch()
    result = engine_compiler.ensure_engines_compiled(engine_compiler.EngineCompilationRequest(
        device="mps", fp16=True, basicvsrpp=True, basicvsrpp_model_path="x.pth",
        detection=True, detection_model_name="rfdetr-v6", detection_model_path=str(path),
    ))
    assert result.use_basicvsrpp_tensorrt is False
    forbidden.assert_not_called()
    with pytest.raises(RuntimeError, match="requires the NVIDIA TensorRT build"):
        engine_compiler.ensure_engines_compiled(engine_compiler.EngineCompilationRequest(device="mps", fp16=False, unet4x=True))


@pytest.fixture
def real_weights():
    value = os.environ.get("JASNA_TEST_MODEL_WEIGHTS_DIR")
    if not value:
        pytest.skip("set JASNA_TEST_MODEL_WEIGHTS_DIR for trusted release weights")
    assert torch.backends.mps.is_available(), "Real weights verification requires available MPS"
    return Path(value)


def assert_mps_model(model):
    tensors = list(model.parameters()) + list(model.buffers())
    assert tensors
    for value in tensors:
        assert value.device.type == "mps"
        if value.is_floating_point():
            assert value.dtype == torch.float32
            assert torch.isfinite(value).all().item()


@pytest.mark.parametrize("name,variant,classes", [("rfdetr-v6.pt", "medium", 3), ("rfdetr-vr-v1.pt", "large", 2)])
def test_real_rfdetr_cpu_load_then_mps(real_weights, name, variant, classes):
    from jasna.mosaic.rfdetr_torch_runner import RfDetrTorchRunner
    checkpoint = load_rfdetr_checkpoint(real_weights / name)
    state = checkpoint["model"]
    assert len(state) == 573
    assert state["class_embed.weight"].shape == (classes, 256)
    assert all(v.device.type == "cpu" for v in state.values())
    runner = RfDetrTorchRunner(real_weights / name, [(1, 3, 576, 576)], torch.device("mps"), fp16=False, resolution=576 if variant == "medium" else 768, variant=variant)
    assert_mps_model(runner._core)
    # Real checkpoint operation, scoped to weights loading rather than issue #8 inference.
    with torch.inference_mode():
        result = runner._core.class_embed(torch.ones(1, 256, device="mps"))
    assert result.shape == (1, classes)
    assert result.device.type == "mps" and torch.isfinite(result).all().item()
    runner.close()


def test_real_restoration_cpu_load_then_mps(real_weights):
    from jasna.models.basicvsrpp.inference import load_model
    path = real_weights / "lada_mosaic_restoration_model_generic_v1.2.pth"
    state = load_restoration_state_dict(path)
    assert len(state) == 812
    assert all(v.device.type == "cpu" and v.dtype == torch.float32 for v in state.values())
    model = load_model(None, str(path), torch.device("mps"), False)
    assert_mps_model(model)
    # A loaded feature convolution; temporal inference remains issue #9.
    with torch.inference_mode():
        result = model.generator.feat_extract(torch.ones(1, 3, 64, 64, device="mps"))
    assert result.shape == (1, 64, 16, 16)
    assert result.device.type == "mps" and torch.isfinite(result).all().item()


def test_real_yolo_cpu_load_then_mps(real_weights):
    from ultralytics.nn.tasks import SegmentationModel
    checkpoint = torch.load(real_weights / "lada_mosaic_detection_model_v4_fast.pt", map_location="cpu", weights_only=False)
    model = checkpoint["model"]
    assert isinstance(model, SegmentationModel)
    assert all(v.device.type == "cpu" for v in model.state_dict().values())
    assert any(v.dtype == torch.float16 for v in model.state_dict().values())
    model = model.float().to("mps").eval()
    assert_mps_model(model)
    with torch.inference_mode():
        result = model.model[0](torch.ones(1, 3, 64, 64, device="mps"))
    assert result.shape == (1, 16, 32, 32)
    assert result.device.type == "mps" and torch.isfinite(result).all().item()


@pytest.mark.parametrize("suffix", [".engine", ".onnx", ".onnx.enc"])
def test_direct_yolo_mps_load_rejects_foreign_formats(tmp_path, suffix):
    from jasna.mosaic.yolo import YoloMosaicDetectionModel
    path = tmp_path / ("model" + suffix)
    path.touch()
    with pytest.raises(ValueError, match="requires .pt|encrypted"):
        YoloMosaicDetectionModel(model_path=path, batch_size=1, device=torch.device("mps"))


@pytest.mark.parametrize("exported", [None, False, True])
def test_rfdetr_constructs_on_cpu_before_mps_transfer(monkeypatch, tmp_path, exported):
    if exported is None:
        monkeypatch.delenv("JASNA_MPS_RFDETR_EXPORT", raising=False)
    else:
        monkeypatch.setenv("JASNA_MPS_RFDETR_EXPORT", "1" if exported else "0")
    from types import SimpleNamespace
    import sys
    from jasna.mosaic.rfdetr_torch_runner import RfDetrTorchRunner
    import jasna.mosaic.rfdetr_torch_runner as module
    weights = tmp_path / "model.pt"
    torch.save({"model": {"class_embed.weight": torch.ones(3, 256)}}, weights)
    events = []
    class Core:
        def to(self, device):
            events.append(("transfer", str(device)))
            return self
        def eval(self):
            return self
    class Wrapper:
        def __init__(self, **kwargs):
            events.append(("construct", kwargs["device"]))
            self.model = SimpleNamespace(model=Core())
        def optimize_for_inference(self, **kwargs):
            events.append(("export", kwargs))
            self.model.inference_model = self.model.model
            self.model.model = None
    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(RFDETRSegMedium=Wrapper))
    monkeypatch.setattr(module, "device_name", lambda device: "test MPS")
    runner = RfDetrTorchRunner(weights, [(1, 3, 576, 576)], torch.device("mps"), fp16=False, resolution=576, variant="medium")
    expected = [("construct", "cpu")]
    if exported:
        expected.append(("export", dict(compile=False, dtype=torch.float32, inplace=True)))
    expected.append(("transfer", "mps"))
    assert events == expected
    runner.close()
