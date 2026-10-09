from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import jasna.accelerator as accelerator


@pytest.fixture
def available_mps(monkeypatch):
    backend = SimpleNamespace(
        is_built=MagicMock(return_value=True),
        is_available=MagicMock(return_value=True),
        get_name=MagicMock(return_value="Apple M2 Pro"),
    )
    monkeypatch.setattr(accelerator, "_mps_backend", lambda: backend)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "cuda", None)
    return backend


def test_mps_reports_apple_vendor_and_capabilities(available_mps) -> None:
    assert accelerator.vendor_for_device("mps") is accelerator.AcceleratorVendor.APPLE
    assert accelerator.is_apple_device(torch.device("mps"))

    capabilities = accelerator.capabilities_for_device("mps")
    assert capabilities.streams is False
    assert capabilities.events is False
    assert capabilities.ipc_collect is False
    assert capabilities.mem_get_info is False


def test_default_device_detects_available_mps(available_mps) -> None:
    assert accelerator.vendor_for_device() is accelerator.AcceleratorVendor.APPLE


def test_cpu_and_cuda_like_vendor_dispatch_is_preserved(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "hip", "7.2")
    monkeypatch.setattr(torch.version, "cuda", None)
    assert accelerator.vendor_for_device("cpu") is accelerator.AcceleratorVendor.CPU
    assert accelerator.vendor_for_device("cuda:0") is accelerator.AcceleratorVendor.AMD

    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    assert accelerator.vendor_for_device("cuda:0") is accelerator.AcceleratorVendor.NVIDIA


def test_mps_unavailable_has_clear_diagnostic(monkeypatch) -> None:
    backend = SimpleNamespace(
        is_built=MagicMock(return_value=True),
        is_available=MagicMock(return_value=False),
    )
    monkeypatch.setattr(accelerator, "_mps_backend", lambda: backend)

    with pytest.raises(RuntimeError, match="MPS is not available"):
        accelerator.vendor_for_device("mps")


def test_mps_not_built_has_clear_diagnostic(monkeypatch) -> None:
    backend = SimpleNamespace(
        is_built=MagicMock(return_value=False),
        is_available=MagicMock(side_effect=AssertionError("must not probe availability")),
    )
    monkeypatch.setattr(accelerator, "_mps_backend", lambda: backend)

    with pytest.raises(RuntimeError, match="not built with MPS support"):
        accelerator.vendor_for_device("mps")


def test_mps_context_streams_and_events_use_synchronous_fallback(available_mps) -> None:
    context = accelerator.device_context("mps")
    assert isinstance(context, nullcontext)
    with context:
        pass

    assert accelerator.new_stream("mps") is None
    assert accelerator.current_stream("mps") is None
    assert accelerator.new_event("mps") is None
    with accelerator.stream_context(None):
        pass


def test_mps_synchronize_uses_mps_signature_without_device_arg(
    monkeypatch, available_mps
) -> None:
    synchronize = MagicMock()
    monkeypatch.setattr(torch.mps, "synchronize", synchronize)

    accelerator.synchronize("mps")

    synchronize.assert_called_once_with()


def test_mps_empty_cache_uses_mps_backend(monkeypatch, available_mps) -> None:
    empty_cache = MagicMock()
    monkeypatch.setattr(torch.mps, "empty_cache", empty_cache)

    accelerator.empty_cache("mps")

    empty_cache.assert_called_once_with()


def test_mps_device_name_uses_backend_name(available_mps) -> None:
    assert accelerator.device_name("mps") == "Apple M2 Pro"
    available_mps.get_name.assert_called_once_with()


def test_mps_does_not_fake_cuda_style_memory_info(available_mps) -> None:
    with pytest.raises(NotImplementedError, match="unified memory"):
        accelerator.mem_get_info("mps")


def test_mps_noop_helpers_do_not_require_cuda_semantics(available_mps) -> None:
    accelerator.set_device("mps")
    accelerator.ipc_collect("mps")
    accelerator.reset_peak_memory_stats("mps")
