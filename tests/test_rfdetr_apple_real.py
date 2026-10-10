"""Opt-in real Apple-native v6 tests; never count skipped cases as hardware PASS."""

import importlib.util
import os
from pathlib import Path
import pytest
import torch

from jasna.mosaic.detection_registry import build_detection_model
from scripts.benchmark_rfdetr_apple import read_frames, raw_metrics, selected_metrics

pytestmark = [pytest.mark.mps_real, pytest.mark.model_required]


@pytest.fixture(params=["mlx", "coreml"])
def native_backend(request):
    backend = request.param
    if backend not in os.environ.get("JASNA_TEST_APPLE_BACKENDS", "").split(","):
        pytest.skip("set JASNA_TEST_APPLE_BACKENDS=mlx,coreml for native real tests")
    dependency = "mlx" if backend == "mlx" else "coremltools"
    assert importlib.util.find_spec(
        dependency
    ), f"Opted-in backend dependency missing: {dependency}"
    if backend == "coreml":
        assert os.environ.get(
            "JASNA_RFDETR_COREML_DIR"
        ), "Core ML export artifact directory required"
    return backend


@pytest.fixture
def native_weights():
    directory = os.environ.get("JASNA_TEST_MODEL_WEIGHTS_DIR")
    if not directory:
        pytest.skip("set JASNA_TEST_MODEL_WEIGHTS_DIR for readonly real v6 weights")
    return Path(directory) / "rfdetr-v6.pt"


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_real_native_raw_and_selected_parity(
    native_backend, native_weights, batch, monkeypatch
):
    monkeypatch.setenv("JASNA_MPS_RFDETR_EXPORT", "0")
    monkeypatch.setenv("JASNA_APPLE_RFDETR_BACKEND", "torch")
    ref = build_detection_model(
        "rfdetr-v6",
        native_weights,
        batch_size=4,
        device=torch.device("mps"),
        score_threshold=0.35,
        fp16=False,
    )
    monkeypatch.setenv("JASNA_APPLE_RFDETR_BACKEND", native_backend)
    native = build_detection_model(
        "rfdetr-v6",
        native_weights,
        batch_size=4,
        device=torch.device("mps"),
        score_threshold=0.35,
        fp16=False,
    )
    try:
        frames = read_frames(
            Path(__file__).resolve().parents[1] / "assets/test_clip1_1080p.mp4",
            [120, 121, 122, 123],
        )[:batch]
        with torch.inference_mode():
            x = ref._preprocess(frames)
            r = ref._infer(x)
            a = native._infer(x)
            metrics = raw_metrics(a, r)
            assert all(v["finite"] for v in metrics.values())
            assert metrics["dets"]["shape"] == [batch, 200, 4]
            assert metrics["labels"]["shape"] == [batch, 200, 3]
            assert metrics["masks"]["shape"] == [batch, 200, 144, 144]
            # Raw order can change for nearly tied proposals; this fixture's
            # margin is separated. Keep selected parity as the product gate.
            assert metrics["labels"]["max_abs"] < 0.01
            assert metrics["masks"]["rmse"] < 0.005
            repeated = native._infer(x)
            assert all(torch.equal(a[k].cpu(), repeated[k].cpu()) for k in a)
            selected = selected_metrics(
                native(frames, target_hw=(1080, 1920)),
                ref(frames, target_hw=(1080, 1920)),
            )
            assert sum(selected["reference_counts"]) > 0
            assert selected["counts_match"] and selected["positive_decisions_match"]
            assert selected["max_box_pixels"] <= 1 and selected["min_mask_iou"] >= 0.995
            actual_scores = (
                a["labels"].sigmoid().flatten(1).topk(16, dim=1).values.cpu()
            )
            reference_scores = (
                r["labels"].sigmoid().flatten(1).topk(16, dim=1).values.cpu()
            )
            selected["max_score_error"] = float(
                (actual_scores - reference_scores).abs().max()
            )
            assert selected["max_score_error"] <= 0.001
            print(native_backend, batch, metrics, selected)
    finally:
        native.close()
        ref.close()


def test_real_native_cli_detection_restoration_encode(
    native_backend, native_weights, tmp_path, monkeypatch
):
    from test_mps_e2e import verify_cli_detection_restoration_encode
    from jasna.mosaic.rfdetr_mlx_runner import RfDetrMlxRunner
    from jasna.mosaic.rfdetr_coreml_runner import RfDetrCoreMLRunner

    monkeypatch.setenv("JASNA_APPLE_RFDETR_BACKEND", native_backend)
    monkeypatch.setenv("JASNA_MPS_RFDETR_EXPORT", "0")
    verify_cli_detection_restoration_encode(
        native_weights.parent,
        tmp_path,
        monkeypatch,
        "software",
        "h264",
        native_runner=RfDetrMlxRunner
        if native_backend == "mlx"
        else RfDetrCoreMLRunner,
    )
