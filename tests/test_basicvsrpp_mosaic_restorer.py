import pytest
import torch


class _CaptureIdentityModel:
    def __init__(self) -> None:
        self.captured_inputs: torch.Tensor | None = None

    def __call__(self, *, inputs: torch.Tensor) -> torch.Tensor:
        self.captured_inputs = inputs.detach().clone()
        return inputs


def _make_restorer(monkeypatch, model, *, use_tensorrt=False, fp16=False, config=None):
    import jasna.restorer.basicvsrpp_mosaic_restorer as br

    monkeypatch.setattr(br, "load_model", lambda config, checkpoint_path, device, fp16: model)
    return br.BasicvsrppMosaicRestorer(
        checkpoint_path="unused.pth",
        device=torch.device("cpu"),
        max_clip_size=30,
        use_tensorrt=use_tensorrt,
        fp16=fp16,
        config=config,
    )


def test_raw_process_normalizes_frames_to_unit_range(monkeypatch) -> None:
    model = _CaptureIdentityModel()
    restorer = _make_restorer(monkeypatch, model)

    frame = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)

    out = restorer.raw_process([frame])

    assert out.shape == (1, 3, 256, 256)
    assert model.captured_inputs.shape == (1, 1, 3, 256, 256)
    assert model.captured_inputs.dtype == torch.float32
    assert torch.equal(model.captured_inputs[0, 0], frame.to(torch.float32).div(255.0))

def test_raw_process_stacks_frames_into_one_clip(monkeypatch) -> None:
    model = _CaptureIdentityModel()
    restorer = _make_restorer(monkeypatch, model)

    frames = [torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8) for _ in range(2)]
    restorer.raw_process(frames)

    assert model.captured_inputs.shape[:2] == (1, 2)

def test_raw_process_empty_video_raises(monkeypatch) -> None:
    import pytest

    restorer = _make_restorer(monkeypatch, _CaptureIdentityModel())

    with pytest.raises(RuntimeError):
        restorer.raw_process([])

def test_init_sets_device_dtype_and_loads_model(monkeypatch) -> None:
    import jasna.restorer.basicvsrpp_mosaic_restorer as br

    captured: dict[str, object] = {}

    def fake_load_model(config, checkpoint_path, device, fp16):
        captured["config"] = config
        captured["checkpoint_path"] = checkpoint_path
        captured["device"] = device
        captured["fp16"] = fp16
        return _CaptureIdentityModel()

    monkeypatch.setattr(br, "load_model", fake_load_model)

    restorer = br.BasicvsrppMosaicRestorer(
        checkpoint_path="ckpt.pth",
        device=torch.device("cpu"),
        max_clip_size=30,
        use_tensorrt=False,
        fp16=True,
        config={"x": 1},
    )

    assert restorer.device.type == "cpu"
    assert restorer.input_dtype == torch.float16
    assert isinstance(restorer.model, _CaptureIdentityModel)

    assert captured["checkpoint_path"] == "ckpt.pth"
    assert captured["device"] == torch.device("cpu")
    assert captured["fp16"] is True
    assert captured["config"] == {"x": 1}


def test_raw_process_produces_contiguous_nchw_input(monkeypatch) -> None:
    import jasna.restorer.basicvsrpp_mosaic_restorer as br

    model = _CaptureIdentityModel()
    restorer = _make_restorer(monkeypatch, model)

    frames = [torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8) for _ in range(3)]
    restorer.raw_process(frames)

    assert model.captured_inputs is not None
    inp = model.captured_inputs.squeeze(0)
    assert inp.is_contiguous(), f"model input must be contiguous NCHW, got stride {inp.stride()}"


def test_split_forward_path_used_when_available(monkeypatch) -> None:
    import jasna.restorer.basicvsrpp_mosaic_restorer as br

    captured: list[torch.Tensor] = []

    class _FakeSplit:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            captured.append(x.detach().clone())
            return x

    model = _CaptureIdentityModel()
    restorer = _make_restorer(monkeypatch, model)
    restorer._split_forward = _FakeSplit()

    frame = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
    restorer.raw_process([frame])

    assert len(captured) == 1
    assert captured[0].shape == (1, 1, 3, 256, 256)
    assert model.captured_inputs is None


@pytest.mark.parametrize("fp16", [False, True])
@pytest.mark.parametrize("device,vendor", [("mps", "apple"), ("cuda:0", "amd"), ("cuda:0", "nvidia"), ("cpu", "cpu")])
def test_precision_policy_and_backend_routing(monkeypatch, caplog, fp16, device, vendor):
    import sys
    from types import ModuleType
    from unittest.mock import Mock
    import jasna.restorer.basicvsrpp_mosaic_restorer as br

    model = _CaptureIdentityModel()
    load = Mock(return_value=model)
    split = Mock()
    create = Mock(return_value=split)
    engines = ModuleType("jasna.restorer.basicvsrpp_sub_engines")
    engines.create_split_forward = create
    monkeypatch.setitem(sys.modules, engines.__name__, engines)
    monkeypatch.setattr(br, "load_model", load)
    monkeypatch.setattr(br, "is_nvidia_device", lambda _: vendor == "nvidia")
    restorer = br.BasicvsrppMosaicRestorer("checkpoint.pth", device, 3, True, fp16)
    effective_fp16 = fp16 and vendor != "apple"
    load.assert_called_once_with(None, "checkpoint.pth", torch.device(device), effective_fp16)
    assert restorer.input_dtype == (torch.float16 if effective_fp16 else torch.float32)
    if vendor == "nvidia":
        create.assert_called_once_with(model=model, model_weights_path="checkpoint.pth",
                                       device=torch.device(device), fp16=fp16)
        assert restorer._split_forward is split and restorer.model is None
    else:
        create.assert_not_called()
        assert restorer._split_forward is None and restorer.model is model
    if vendor == "apple" and fp16:
        assert "MPS uses FP32" in caplog.text
    restorer.close()
    if vendor == "nvidia":
        split.close.assert_called_once()
