from __future__ import annotations

import sys

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
