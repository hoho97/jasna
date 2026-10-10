"""Strict, reproducible real MPS gate (run with python -m scripts.verify_mps_gate).

This orchestrates the production inference/video tests instead of maintaining a
second pipeline. Checkpoints are opened only for hashing; tests load them via
explicit paths. Any failed/skipped/empty selection is a failed gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

CHECKPOINTS = (
    "rfdetr-v6.pt", "rfdetr-vr-v1.pt",
    "lada_mosaic_restoration_model_generic_v1.2.pth",
    "lada_mosaic_detection_model_v4_fast.pt",
)
REQUIRED_TESTS = (
    "test_real_checkpoint_inference_and_cpu_reference",
    "test_real_temporal_restoration_matches_cpu",
    "test_real_cli_detection_restoration_encode",
    "test_installed_cli_exit_statuses",
    "test_real_checkpoints_threaded_pass_and_memory_plateau",
)


def checkpoint_hashes(directory: Path) -> dict:
    result = {}
    for name in CHECKPOINTS:
        path = directory / name
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"Required nonempty checkpoint not found: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result[name] = {"path": str(path), "bytes": path.stat().st_size,
                        "sha256": digest.hexdigest()}
    return result


def require_real_mps(torch, accelerator, environ: dict) -> dict:
    if "PYTORCH_ENABLE_MPS_FALLBACK" in environ:
        raise RuntimeError("Unset PYTORCH_ENABLE_MPS_FALLBACK before running the strict gate")
    built, available = torch.backends.mps.is_built(), torch.backends.mps.is_available()
    if not built or not available:
        raise RuntimeError(f"Real MPS required: built={built}, available={available}")
    explicit = accelerator.vendor_for_device("mps")
    automatic = accelerator.vendor_for_device()
    if explicit != accelerator.AcceleratorVendor.APPLE or automatic != accelerator.AcceleratorVendor.APPLE:
        raise RuntimeError(f"Apple dispatch required: explicit={explicit}, automatic={automatic}")
    value = torch.ones(4, device="mps").mul(2)
    torch.mps.synchronize()
    if value.device.type != "mps" or not torch.equal(value.cpu(), torch.full((4,), 2.0)):
        raise RuntimeError("Real MPS tensor operation contract failed")
    return {"built": built, "available": available, "explicit_vendor": str(explicit),
            "automatic_vendor": str(automatic), "device": str(value.device),
            "device_name": accelerator.device_name("mps"), "dtype": str(value.dtype),
            "operation": "torch.ones(4, device='mps').mul(2)", "values": value.cpu().tolist(),
            "global_fallback": None}


def junit_result(path: Path, returncode: int) -> dict:
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    counts = {tag: sum(case.find(tag) is not None for case in cases)
              for tag in ("failure", "error", "skipped")}
    names = [case.get("name", "") for case in cases]
    missing = [name for name in REQUIRED_TESTS
               if not any(actual == name or actual.startswith(name + "[") for actual in names)]
    passed = bool(cases) and returncode == 0 and not any(counts.values()) and not missing
    return {"result": "PASS" if passed else "FAIL", "returncode": returncode,
            "tests": len(cases), **counts, "missing_required_tests": missing,
            "testcases": [{"classname": c.get("classname"), "name": c.get("name"),
                           "seconds": c.get("time")} for c in cases]}


def run_command(command: list[str], directory: Path, report: dict, *, env=None, cwd=None) -> str:
    """Persist exact argv, shell-readable command, stdout/stderr and exit code."""
    index = len(report["commands"])
    log = directory / f"command-{index:02d}.log"
    record = {"argv": command, "command": shlex.join(command), "cwd": str(cwd or Path.cwd()),
              "log": str(log)}
    report["commands"].append(record)
    started = time.perf_counter()
    with log.open("w") as output:
        process = subprocess.run(command, env=env, cwd=cwd, stdout=output, stderr=subprocess.STDOUT)
    record.update(returncode=process.returncode, seconds=time.perf_counter() - started)
    text = log.read_text(errors="replace")
    if process.returncode:
        raise RuntimeError(f"Command failed ({process.returncode}): {record['command']}; see {log}")
    return text


def verify_outputs(directory: Path, report: dict) -> list[dict]:
    videos = directory / "video"
    # These files are produced by the existing real temporal and positive-detection E2E tests.
    required = [videos / name for name in ("cli-restored.mp4", "basicvsrpp-t2.mp4", "basicvsrpp-t3.mp4")]
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"Missing required real model video output: {path}")
    results = []
    for path in sorted(videos.glob("*.mp4")):
        probe = run_command(["ffprobe", "-v", "error", "-count_frames", "-show_streams",
                             "-show_format", "-of", "json", str(path)], directory, report)
        details = json.loads(probe)
        streams = [s for s in details.get("streams", []) if s.get("codec_type") == "video"]
        if not streams or not all(int(s.get("nb_read_frames", "0")) > 0 for s in streams):
            raise RuntimeError(f"No decoded video frames: {path}")
        run_command(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
                    directory, report)
        results.append({"path": str(path), "ffprobe": details, "full_decode": "PASS"})
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New or empty evidence directory (never overwrite an old gate run)")
    args = parser.parse_args(argv)
    directory = args.output_dir.resolve()
    weights = args.weights_dir.resolve()
    # Prevent generated evidence from touching the trusted model directory.
    if directory == weights or weights in directory.parents:
        parser.error("--output-dir must be outside --weights-dir")
    if directory.exists() and any(directory.iterdir()):
        parser.error("--output-dir must be new or empty")
    directory.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    report = {"result": "FAIL", "started_utc": datetime.now(timezone.utc).isoformat(),
              "python": sys.version, "python_executable": sys.executable,
              "invocation": shlex.join([sys.executable, "-m", "scripts.verify_mps_gate",
                                        *(sys.argv[1:] if argv is None else argv)]),
              "platform": platform.platform(), "machine": platform.machine(), "commands": [],
              "fallback_boundaries": ["software decode/RGB and host encode/mux",
                                      "detector boxes postprocess and scoped scan area resize"],
              "environment": {key: os.environ.get(key) for key in
                              ("PYTORCH_ENABLE_MPS_FALLBACK", "JASNA_MAIN_PID", "PYTEST_ADDOPTS")}}
    started = time.perf_counter()
    try:
        if os.environ.get("PYTEST_ADDOPTS"):
            raise RuntimeError("Unset PYTEST_ADDOPTS so the recorded strict selection cannot be overridden")
        import torch
        import jasna.accelerator as accelerator
        report["mps"] = require_real_mps(torch, accelerator, os.environ)
        report["versions"] = {name: metadata.version(name) for name in
                              ("torch", "torchvision", "rfdetr", "transformers", "av", "numpy", "pytest", "mmengine")}
        for command in ([sys.executable, "--version"],
                        [sys.executable, "-c", "import torch; print(torch.__version__); print(torch.backends.mps.is_built(), torch.backends.mps.is_available())"],
                        ["sw_vers"], ["uname", "-m"],
                        ["sysctl", "-n", "machdep.cpu.brand_string", "hw.memsize"],
                        ["ffmpeg", "-version"], ["ffprobe", "-version"], ["git", "rev-parse", "HEAD"],
                        ["git", "status", "--short"], ["git", "diff", "HEAD"],
                        ["git", "ls-files", "--others", "--exclude-standard"]):
            run_command(command, directory, report, cwd=repo)
        report["checkpoints"] = checkpoint_hashes(weights)
        if not (repo / "assets/test_clip1_1080p.mp4").is_file():
            raise RuntimeError("Repository test_clip1_1080p.mp4 fixture is required")
        video = directory / "video"
        video.mkdir()
        env = dict(os.environ, JASNA_TEST_MODEL_WEIGHTS_DIR=str(weights),
                   JASNA_MODEL_WEIGHTS_DIR=str(weights), JASNA_TEST_VIDEO_OUTPUT_DIR=str(video))
        env.pop("JASNA_MAIN_PID", None)
        report["test_environment"] = {key: env[key] for key in
                                      ("JASNA_TEST_MODEL_WEIGHTS_DIR", "JASNA_MODEL_WEIGHTS_DIR", "JASNA_TEST_VIDEO_OUTPUT_DIR")}
        command = [sys.executable, "-m", "pytest", "-q", "-s", "-m", "mps_real",
                   "--junitxml", str(directory / "pytest.xml"), "tests"]
        print(f"Running strict real MPS gate; evidence: {directory}", flush=True)
        try:
            run_command(command, directory, report, env=env, cwd=repo)
        finally:
            xml = directory / "pytest.xml"
            if xml.is_file():
                report["pytest"] = junit_result(xml, report["commands"][-1]["returncode"])
        if report.get("pytest", {}).get("result") != "PASS":
            raise RuntimeError("Real MPS selection failed: skipped, empty, failed or missing required coverage")
        report["videos"] = verify_outputs(directory, report)
        report["checkpoints_after"] = checkpoint_hashes(weights)
        if report["checkpoints_after"] != report["checkpoints"]:
            raise RuntimeError("Read-only checkpoint integrity changed during verification")
        report["result"] = "PASS"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["seconds"] = time.perf_counter() - started
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"result": report["result"], "report": str(directory / "report.json"),
                      "error": report.get("error")}, indent=2))
    return 0 if report["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
