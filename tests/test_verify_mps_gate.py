"""Gate policy tests: no GPU, model loading or checkpoint writes required."""
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest

from scripts import verify_mps_gate as gate


def write_junit(path, *, names=gate.REQUIRED_TESTS, outcome=None):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for name in names:
        case = ET.SubElement(suite, "testcase", name=name, classname="tests.real")
        if outcome:
            ET.SubElement(case, outcome)
    ET.ElementTree(root).write(path)


@pytest.mark.parametrize("outcome", ["skipped", "failure", "error"])
def test_gate_rejects_nonpassing_cases_even_with_zero_process_exit(tmp_path, outcome):
    xml = tmp_path / "results.xml"
    write_junit(xml, outcome=outcome)
    result = gate.junit_result(xml, 0)
    assert result["result"] == "FAIL"
    assert result[outcome] == len(gate.REQUIRED_TESTS)


@pytest.mark.parametrize("names,returncode", [((), 0), (gate.REQUIRED_TESTS[:1], 0),
                                              (gate.REQUIRED_TESTS, 1), (gate.REQUIRED_TESTS, 5)])
def test_gate_rejects_empty_incomplete_and_failed_pytest(tmp_path, names, returncode):
    xml = tmp_path / "results.xml"
    write_junit(xml, names=names)
    assert gate.junit_result(xml, returncode)["result"] == "FAIL"


def test_gate_accepts_successful_required_parametrized_cases(tmp_path):
    xml = tmp_path / "results.xml"
    write_junit(xml, names=[name + "[real-mps]" for name in gate.REQUIRED_TESTS])
    result = gate.junit_result(xml, 0)
    assert result["result"] == "PASS"
    assert result["tests"] == len(gate.REQUIRED_TESTS)
    assert result["missing_required_tests"] == []


def test_checkpoint_hashing_preserves_originals_and_requires_all_weights(tmp_path):
    for name in gate.CHECKPOINTS:
        (tmp_path / name).write_bytes(b"trusted checkpoint")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    hashes = gate.checkpoint_hashes(tmp_path)
    assert len(hashes) == len(gate.CHECKPOINTS)
    assert all(len(info["sha256"]) == 64 for info in hashes.values())
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before
    (tmp_path / gate.CHECKPOINTS[-1]).unlink()
    with pytest.raises(RuntimeError, match="Required nonempty checkpoint"):
        gate.checkpoint_hashes(tmp_path)


@pytest.mark.parametrize("built,available", [(False, False), (True, False)])
def test_gate_rejects_unavailable_mps_before_tensor_work(built, available):
    torch = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(
        is_built=lambda: built, is_available=lambda: available)))
    with pytest.raises(RuntimeError, match="Real MPS required"):
        gate.require_real_mps(torch, None, {})


@pytest.mark.parametrize("value", ["1", "0", ""])
def test_gate_requires_global_fallback_environment_unset(value):
    with pytest.raises(RuntimeError, match="Unset PYTORCH_ENABLE_MPS_FALLBACK"):
        gate.require_real_mps(None, None, {"PYTORCH_ENABLE_MPS_FALLBACK": value})


def test_gate_rejects_wrong_automatic_vendor_before_tensor_work():
    torch = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(
        is_built=lambda: True, is_available=lambda: True)))
    accelerator = SimpleNamespace(AcceleratorVendor=SimpleNamespace(APPLE="apple"),
                                  vendor_for_device=lambda device=None: "apple" if device else "cpu")
    with pytest.raises(RuntimeError, match="Apple dispatch required"):
        gate.require_real_mps(torch, accelerator, {})


def test_existing_evidence_directory_cannot_be_overwritten(tmp_path):
    evidence = tmp_path / "old.json"
    evidence.write_text("old evidence")
    with pytest.raises(SystemExit) as error:
        gate.main(["--weights-dir", str(tmp_path / "weights"), "--output-dir", str(tmp_path)])
    assert error.value.code == 2
    assert evidence.read_text() == "old evidence"


def test_evidence_cannot_be_written_inside_readonly_weights(tmp_path):
    with pytest.raises(SystemExit) as error:
        gate.main(["--weights-dir", str(tmp_path), "--output-dir", str(tmp_path / "evidence")])
    assert error.value.code == 2
    assert not (tmp_path / "evidence").exists()


def test_command_failure_records_exact_argv_log_and_exit(tmp_path):
    import sys
    report = {"commands": []}
    command = [sys.executable, "-c", "print('actual failure'); raise SystemExit(7)"]
    with pytest.raises(RuntimeError, match="Command failed \\(7\\)"):
        gate.run_command(command, tmp_path, report)
    record = report["commands"][0]
    assert record["argv"] == command
    assert record["returncode"] == 7
    assert Path(record["log"]).read_text() == "actual failure\n"


def test_missing_model_output_cannot_pass(tmp_path):
    (tmp_path / "video").mkdir()
    with pytest.raises(RuntimeError, match="Missing required real model video output"):
        gate.verify_outputs(tmp_path, {"commands": []})


def test_gate_records_preflight_failure_as_fail(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("PYTEST_ADDOPTS", "--ignore tests/test_mps_e2e.py")
    output = tmp_path / "evidence"
    assert gate.main(["--weights-dir", str(tmp_path / "weights"), "--output-dir", str(output)]) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["result"] == "FAIL"
    assert "Unset PYTEST_ADDOPTS" in report["error"]
