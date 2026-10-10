"""Portable Apple backend policy/mapping tests: no Metal runtime required."""

import builtins
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import torch
import numpy as np

from jasna.mosaic.rfdetr_apple import apple_rfdetr_backend
from jasna.mosaic.mlx_rfdetr.config import JasnaV6MLXConfig
from jasna.mosaic.mlx_rfdetr.checkpoint import mapped_tensors, validate_mapping
from jasna.mosaic.rfdetr_mlx_runner import RfDetrMlxRunner
from jasna.mosaic.rfdetr_coreml_runner import RfDetrCoreMLRunner, validate_manifest


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("backend", ["mlx", "coreml", "invalid"])
def test_foreign_vendors_ignore_apple_configuration(monkeypatch, device, backend):
    monkeypatch.setenv("JASNA_APPLE_RFDETR_BACKEND", backend)
    assert apple_rfdetr_backend(torch.device(device)) == "torch"


@pytest.mark.parametrize("backend", ["torch", "mlx", "coreml"])
def test_explicit_apple_selection(monkeypatch, backend):
    monkeypatch.setenv("JASNA_APPLE_RFDETR_BACKEND", backend)
    assert apple_rfdetr_backend(torch.device("mps")) == backend


def test_default_and_invalid_selection(monkeypatch):
    monkeypatch.delenv("JASNA_APPLE_RFDETR_BACKEND", raising=False)
    assert apple_rfdetr_backend(torch.device("mps")) == "torch"
    monkeypatch.setenv("JASNA_APPLE_RFDETR_BACKEND", "typo")
    with pytest.raises(ValueError, match="must be"):
        apple_rfdetr_backend(torch.device("mps"))


def test_v6_config_is_not_builtin_seg_medium():
    c = JasnaV6MLXConfig
    assert (c.resolution, c.patch_size, c.num_queries, c.num_classes, c.dec_layers) == (
        576,
        12,
        200,
        2,
        5,
    )
    assert (c.positional_encoding_size, c.num_query_embeddings * c.group_detr) == (
        36,
        2600,
    )
    assert (
        c.hidden_dim,
        c.backbone_dim,
        c.backbone_heads,
        c.sa_nheads,
        c.ca_nheads,
        c.dec_n_points,
    ) == (256, 384, 6, 8, 16, 2)
    assert (
        c.projector_scales == ("P4",)
        and c.mask_downsample_ratio == 4
        and c.num_select == 16
    )


def test_mapping_layout_and_empty_buffer():
    conv = torch.arange(24.0).reshape(2, 3, 2, 2)
    result = mapped_tensors(
        {
            "conv.weight": conv,
            "stages_sampling.weight": conv,
            "linear": torch.ones(3, 2),
            "_kp_active_mask": torch.empty(0, 0),
        }
    )
    torch.testing.assert_close(result["conv.weight"], conv.permute(0, 2, 3, 1))
    torch.testing.assert_close(
        result["stages_sampling.weight"], conv.permute(1, 2, 3, 0)
    )
    assert result["kp_active_mask"].shape == (0, 0)
    with pytest.raises(ValueError, match="empty"):
        mapped_tensors({"_kp_active_mask": torch.ones(1, 1)})
    with pytest.raises(ValueError, match="Non-tensor"):
        mapped_tensors({"bad": 42})


def test_strict_mapping_reports_every_incompatibility():
    actual = {"extra": torch.empty(1), "wrong": torch.empty(3)}
    expected = {"missing": torch.empty(1), "wrong": torch.empty(2)}
    with pytest.raises(ValueError) as error:
        validate_mapping(actual, expected)
    report = json.loads(str(error.value).split(": ", 1)[1])
    assert report == {
        "missing_keys": ["missing"],
        "unexpected_keys": ["extra"],
        "shape_mismatches": {"wrong": {"checkpoint": [3], "model": [2]}},
    }
    assert validate_mapping(expected, expected) == {
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": {},
    }


@pytest.mark.parametrize("runner", [RfDetrMlxRunner, RfDetrCoreMLRunner])
@pytest.mark.parametrize(
    "device,resolution,variant",
    [("cpu", 576, "medium"), ("cuda", 576, "medium"), ("mps", 768, "large")],
)
def test_unsupported_platform_and_quality_never_load_optional_runtime(
    runner, device, resolution, variant
):
    with pytest.raises(ValueError, match="only Jasna v6"):
        runner(
            Path("model.pt"),
            [(4, 3, resolution, resolution)],
            torch.device(device),
            fp16=False,
            resolution=resolution,
            variant=variant,
        )


@pytest.mark.parametrize(
    "runner,blocked,extra",
    [
        (RfDetrMlxRunner, "mlx", "macos-mlx"),
        (RfDetrCoreMLRunner, "coremltools", "macos-coreml"),
    ],
)
def test_optional_dependency_error_is_actionable(monkeypatch, runner, blocked, extra):
    original = builtins.__import__

    def guarded(name, *a, **kw):
        if name.split(".")[0] == blocked:
            raise ImportError("deliberately absent")
        return original(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(RuntimeError, match=extra):
        runner(
            Path("model.pt"),
            [(4, 3, 576, 576)],
            torch.device("mps"),
            fp16=False,
            resolution=576,
            variant="medium",
        )


@pytest.mark.parametrize(
    "backend,module,class_name",
    [
        ("mlx", "rfdetr_mlx_runner", "RfDetrMlxRunner"),
        ("coreml", "rfdetr_coreml_runner", "RfDetrCoreMLRunner"),
    ],
)
def test_native_selection_keeps_tensor_contract_and_avoids_tensorrt(
    monkeypatch, backend, module, class_name
):
    import importlib
    from jasna.mosaic import rfdetr

    monkeypatch.setenv("JASNA_APPLE_RFDETR_BACKEND", backend)
    runner = SimpleNamespace(
        input_names=["input"],
        input_dtypes={"input": torch.float32},
        output_names=["dets", "labels", "masks"],
        outputs={
            "dets": torch.empty(1, 200, 4),
            "labels": torch.empty(1, 200, 3),
            "masks": torch.empty(1, 200, 144, 144),
        },
        detect=Mock(return_value="selected"),
    )
    monkeypatch.setattr(
        importlib.import_module("jasna.mosaic." + module),
        class_name,
        Mock(return_value=runner),
    )
    monkeypatch.setattr(rfdetr, "is_amd_device", lambda _: False)
    monkeypatch.setattr(rfdetr, "is_apple_device", lambda _: True)
    monkeypatch.setattr(
        rfdetr, "TrtRunner", Mock(side_effect=AssertionError("NVIDIA path"))
    )
    monkeypatch.setattr(
        rfdetr, "ResizeNormalizer", lambda **_: SimpleNamespace(available=False)
    )
    model = rfdetr.RfDetrMosaicDetectionModel(
        weights_path=Path("rfdetr-v6.pt"),
        batch_size=4,
        device=torch.device("mps"),
        resolution=576,
        dynamic_batch=True,
        torch_variant="medium",
    )
    monkeypatch.setattr(model, "_preprocess", lambda x: x)
    assert model(torch.empty(1), target_hw=(1080, 1920)) == "selected"
    runner.detect.assert_called_once()


def artifact(tmp_path):
    weights = tmp_path / "v6.pt"
    weights.write_bytes(b"checkpoint")
    for b in (1, 2, 4):
        (tmp_path / f"batch{b}").mkdir(exist_ok=True)
    manifest = {
        "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
        "contract": {
            "resolution": 576,
            "queries": 200,
            "classes": 3,
            "mask_hw": [144, 144],
            "precision": "float32",
        },
        "models": {
            str(b): {
                "path": f"batch{b}",
                "input": "input",
                "outputs": {"dets": "boxes", "labels": "logits", "masks": "masks"},
            }
            for b in (1, 2, 4)
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return weights, manifest


def test_coreml_manifest_is_bound_to_exact_checkpoint_and_quality(tmp_path):
    weights, m = artifact(tmp_path)
    assert validate_manifest(tmp_path, weights) == m
    weights.write_bytes(b"other")
    with pytest.raises(ValueError, match="SHA256"):
        validate_manifest(tmp_path, weights)
    weights.write_bytes(b"checkpoint")
    m["contract"]["precision"] = "float16"
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="contract"):
        validate_manifest(tmp_path, weights)


def test_coreml_rejects_missing_batch_and_external_path(tmp_path):
    weights, m = artifact(tmp_path)
    del m["models"]["2"]
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="static batches"):
        validate_manifest(tmp_path, weights)
    weights, m = artifact(tmp_path)
    m["models"]["4"]["path"] = ".."
    (tmp_path / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="inside artifact"):
        validate_manifest(tmp_path, weights)


def test_coreml_batch3_padding_and_raw_contract():
    runner = RfDetrCoreMLRunner.__new__(RfDetrCoreMLRunner)
    runner.device = torch.device("cpu")
    runner._closed = False
    values = {
        "a": np.zeros((4, 200, 4), np.float32),
        "b": np.zeros((4, 200, 3), np.float32),
        "c": np.zeros((4, 200, 144, 144), np.float32),
    }
    model = Mock()
    model.predict.return_value = values
    runner._models = {4: model}
    runner.manifest = {
        "models": {
            "4": {
                "input": "input",
                "outputs": {"dets": "a", "labels": "b", "masks": "c"},
            }
        }
    }
    result = runner.infer({"input": torch.ones(3, 3, 576, 576)})
    assert result["dets"].shape == (3, 200, 4)
    assert model.predict.call_args.args[0]["input"].shape == (4, 3, 576, 576)
    runner.close()
    with pytest.raises(RuntimeError, match="closed"):
        runner.infer({"input": torch.ones(1, 3, 576, 576)})


def test_native_postprocess_strict_threshold_and_low_resolution_masks():
    from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel

    runner = RfDetrCoreMLRunner.__new__(RfDetrCoreMLRunner)
    runner.device = torch.device("cpu")
    raw = {
        "dets": np.array([[[0.5, 0.5, 0.2, 0.2], [0.3, 0.4, 0.1, 0.1]]], np.float32),
        "labels": np.array([[[0.0, 0.0, 0.0], [-1.0, 1.0, -1.0]]], np.float32),
        "masks": np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2) - 3,
    }
    runner._raw = lambda _: raw
    native = runner.detect(
        torch.empty(0), target_hw=(100, 200), score_threshold=0.5, max_select=16
    )
    rb, rm = RfDetrMosaicDetectionModel._postprocess(
        pred_boxes=torch.from_numpy(raw["dets"]),
        pred_logits=torch.from_numpy(raw["labels"]),
        pred_masks=torch.from_numpy(raw["masks"]),
        target_hw=(100, 200),
        score_threshold=0.5,
        max_select=16,
    )
    assert len(native.boxes_xyxy[0]) == 1  # logits==0 gives exactly .5, excluded.
    np.testing.assert_array_equal(native.boxes_xyxy[0], rb[0])
    assert torch.equal(native.masks[0], rm[0]) and native.masks[0].shape == (1, 2, 2)
