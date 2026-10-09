"""Run Issue #4's real MPS preflight; no model inference or video export implied.

Run from the repository root: python -m scripts.verify_mps_preflight --weights-dir PATH
"""
from __future__ import annotations

import argparse
import builtins
import json
import os
from pathlib import Path
import subprocess
import sys
import time


_original_import = builtins.__import__


def no_optional_gpu_imports(name, *args, **kwargs):
    # Torch probes optional packages with find_spec; only actual imports are forbidden.
    for blocked in ("tensorrt", "torch_tensorrt", "tensorrt_libs", "nvvfx", "python_vali", "jasna.protection"):
        if name == blocked or name.startswith(blocked + "."):
            raise AssertionError(f"Unexpected optional import: {name}")
    return _original_import(name, *args, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", required=True, type=Path)
    parser.add_argument("--video", type=Path)
    args = parser.parse_args()
    builtins.__import__ = no_optional_gpu_imports
    import torch
    from jasna.accelerator import device_name
    from jasna.engine_compiler import EngineCompilationRequest, ensure_engines_compiled
    from jasna.main import _check_system, build_parser
    from jasna.model_weights import validate_weights_path
    from jasna.os_utils import check_gpu_driver_version, check_supported_gpu
    from jasna.restorer import BasicvsrppMosaicRestorer, RestorationPipeline

    assert torch.backends.mps.is_built() and torch.backends.mps.is_available()
    assert check_supported_gpu("mps")[0]
    assert check_gpu_driver_version("mps")[0]
    _check_system(build_parser().parse_args(["--device", "mps"]))
    weights = args.weights_dir.resolve()
    detector = weights / "rfdetr-v6.pt"
    restorer = weights / "lada_mosaic_restoration_model_generic_v1.2.pth"
    validate_weights_path(restorer, suffix=".pth", model="BasicVSR++")
    result = ensure_engines_compiled(EngineCompilationRequest(
        device="mps", fp16=False, basicvsrpp=True, basicvsrpp_model_path=str(restorer),
        detection=True, detection_model_name="rfdetr-v6", detection_model_path=str(detector),
    ))
    assert not result.use_basicvsrpp_tensorrt
    # Check explicit hardware backend rejection on a real MPS device, before open.
    from jasna.media.video_decoder import VideoReader, VideoDecodeError
    for backend in ("vali", "pyav-hw"):
        old = os.environ.get("JASNA_DECODE_BACKEND")
        os.environ["JASNA_DECODE_BACKEND"] = backend
        try:
            reader = VideoReader("must-not-open.mp4", 1, torch.device("mps"), None)
            try:
                reader.__enter__()
            except VideoDecodeError as exc:
                assert "not supported on Apple/MPS" in str(exc)
            else:
                raise AssertionError("Hardware backend was accepted on MPS")
        finally:
            if old is None:
                os.environ.pop("JASNA_DECODE_BACKEND", None)
            else:
                os.environ["JASNA_DECODE_BACKEND"] = old
    # This operation proves that passing preflight corresponds to a working MPS device.
    torch.manual_seed(4)
    conv = torch.nn.Conv2d(3, 8, 3, padding=1).eval()
    inputs = torch.rand(2, 3, 32, 32)
    expected = conv(inputs)
    conv = conv.to("mps")
    torch.mps.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = conv(inputs.to("mps"))
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    assert output.device.type == "mps" and output.dtype == torch.float32
    assert tuple(output.shape) == (2, 8, 32, 32) and torch.isfinite(output).all().item()
    torch.testing.assert_close(output.cpu(), expected, rtol=1e-4, atol=1e-5)
    # Exercise loaded free-model layers, without claiming temporal/detector inference.
    from jasna.models.basicvsrpp.inference import load_model
    from jasna.mosaic.rfdetr_torch_runner import RfDetrTorchRunner
    model = load_model(None, str(restorer), torch.device("mps"), False)
    with torch.inference_mode():
        restored_features = model.generator.feat_extract(torch.ones(1, 3, 64, 64, device="mps"))
    assert restored_features.shape == (1, 64, 16, 16)
    assert restored_features.device.type == "mps" and restored_features.dtype == torch.float32
    assert torch.isfinite(restored_features).all().item()
    runner = RfDetrTorchRunner(detector, [(1, 3, 576, 576)], torch.device("mps"),
                              fp16=False, resolution=576, variant="medium")
    with torch.inference_mode():
        logits = runner._core.class_embed(torch.ones(1, 256, device="mps"))
    assert logits.shape == (1, 3) and logits.device.type == "mps"
    assert logits.dtype == torch.float32 and torch.isfinite(logits).all().item()
    for module in (model, runner._core):
        for tensor in list(module.parameters()) + list(module.buffers()):
            assert tensor.device.type == "mps"
            if tensor.is_floating_point():
                assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all().item()
    runner.close()
    decoded_frames = None
    if args.video is not None:
        import av
        decoded_frames = 0
        with av.open(str(args.video)) as container:
            for frame in container.decode(video=0):
                tensor = torch.from_numpy(frame.to_ndarray(format="rgb24")).to("mps")
                assert tensor.device.type == "mps" and tuple(tensor.shape) == (64, 64, 3)
                decoded_frames += 1
        assert decoded_frames == 4
    env = dict(os.environ)
    env.pop("JASNA_MAIN_PID", None)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    console = Path(sys.executable).with_name("jasna")
    assert console.is_file(), "Install jasna to verify its console entry point"
    for entry in ([sys.executable, "-m", "jasna"],
                  [str(console)]):
        proc = subprocess.run([*entry, "--device", "mps", "--secondary-restoration", "rtx-super-res"],
                              env=env, capture_output=True, text=True)
        assert proc.returncode == 2 and "not supported on mps" in proc.stderr, proc.stderr
    assert not any(name in sys.modules for name in (
        "tensorrt", "torch_tensorrt", "tensorrt_libs", "nvvfx", "python_vali", "jasna.protection"))
    print(json.dumps({"result": "PASS", "python": sys.version, "torch": torch.__version__,
        "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
        "device": device_name("mps"), "shape": list(output.shape), "dtype": str(output.dtype),
        "finite": True, "cpu_reference_close": True, "operation_seconds": elapsed,
        "global_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "checkpoint_paths_validated": [str(detector), str(restorer)],
        "model_operations": {"basicvsrpp_feat_extract": list(restored_features.shape),
                             "rfdetr_class_embed": list(logits.shape)},
        "model_inference": "Loaded checkpoint layers only; full inference owned by #8/#9",
        "cli_entries": "python -m jasna and installed console script rejected RTX before execution",
        "optional_gpu_imports": "none", "pyav_host_decode_mps_upload_frames": decoded_frames}, indent=2))


if __name__ == "__main__":
    main()
