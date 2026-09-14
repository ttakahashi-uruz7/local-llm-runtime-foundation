from __future__ import annotations

import sys

import pytest

from runtime_foundation import HostProfile
import runtime_foundation.host as host_module


def test_windows_development_profile_is_observation_only() -> None:
    profile = HostProfile.mock_windows()
    payload = profile.to_dict()
    assert payload["platform"] == "windows"
    assert payload["architecture"] == "x86_64"
    assert payload["mlx_available"] is False
    assert payload["metal_available"] is False
    assert "runtime_mode" not in payload
    assert "production_eligible" not in payload


def test_host_detection_fallback_does_not_import_mlx_on_windows() -> None:
    profile = HostProfile.detect()
    assert profile.platform
    assert profile.architecture
    assert profile.captured_at
    if profile.platform != "darwin":
        assert profile.mlx_available is False
        assert profile.metal_available is False
        assert profile.unified_memory_bytes is None


def test_process_memory_uses_current_rss_helper(monkeypatch) -> None:
    monkeypatch.setattr(host_module, "_platform_current_rss_bytes", lambda system: 123456)
    assert host_module.process_memory_bytes() == 123456


def test_process_memory_returns_unavailable_instead_of_peak_counter(monkeypatch) -> None:
    monkeypatch.setattr(host_module, "_platform_current_rss_bytes", lambda system: None)
    monkeypatch.setitem(sys.modules, "psutil", None)
    assert host_module.process_memory_bytes() is None


@pytest.mark.parametrize(("machine", "expected"), [("arm64", 16 * 1024**3), ("x86_64", None)])
def test_unified_memory_is_limited_to_apple_silicon(monkeypatch, machine: str, expected: int | None) -> None:
    monkeypatch.setattr(host_module.platform_module, "system", lambda: "Darwin")
    monkeypatch.setattr(host_module.platform_module, "machine", lambda: machine)
    monkeypatch.setattr(host_module.platform_module, "release", lambda: "24.0.0")
    monkeypatch.setattr(host_module.platform_module, "version", lambda: "Darwin test")
    monkeypatch.setattr(host_module, "_physical_memory_bytes", lambda system: 16 * 1024**3)
    monkeypatch.setattr(host_module, "_darwin_sysctl", lambda name: None)
    monkeypatch.setattr(host_module, "_darwin_swap", lambda: (None, None))
    monkeypatch.setattr(host_module, "_darwin_memory_pressure", lambda: None)
    monkeypatch.setattr(host_module, "_mlx_status", lambda system, architecture: (False, False, None, None, "test"))

    profile = HostProfile.detect()

    assert profile.platform == "darwin"
    assert profile.architecture == ("arm64" if machine == "arm64" else "x86_64")
    assert profile.physical_memory_bytes == 16 * 1024**3
    assert profile.unified_memory_bytes == expected
