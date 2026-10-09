from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _optional_dependencies() -> dict[str, list[str]]:
    with (ROOT / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    return data["project"]["optional-dependencies"]


def test_macos_extra_pins_mps_torch_stack() -> None:
    macos = _optional_dependencies()["macos"]

    assert any(dep.startswith("torch==2.12.0;") for dep in macos)
    assert any(dep.startswith("torchvision==0.27.0;") for dep in macos)
    assert any(dep.startswith("rfdetr==1.8.3;") for dep in macos)
    assert any(dep.startswith("transformers==5.1.0;") for dep in macos)


def test_macos_extra_is_apple_silicon_scoped() -> None:
    macos = _optional_dependencies()["macos"]

    for dep in macos:
        marker = dep.split(";", 1)[1]
        assert "platform_system == 'Darwin'" in marker
        assert "platform_machine == 'arm64'" in marker


def test_macos_extra_does_not_pull_vendor_gpu_runtimes() -> None:
    macos = "\n".join(_optional_dependencies()["macos"]).lower()

    forbidden = (
        "tensorrt",
        "torch-tensorrt",
        "python_vali",
        "nvidia-vfx",
        "rocm",
        "cuda",
    )
    assert all(package not in macos for package in forbidden)
