from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from runtime_foundation import HostProfile, LoadMemoryGuard, ModelArtifactBinding, RuntimeCore
from runtime_foundation.errors import HostMemoryInsufficientError, LoadConflictError
from runtime_foundation.memory_guard import DEFAULT_LIMIT_RATIO, LIMIT_RATIO_ENV
from runtime_foundation.service import create_app

GIB = 1 << 30


def _observer(wired_bytes: int, physical_bytes: int = 64 * GIB):
    calls: list[int] = []

    def observe() -> dict[str, int]:
        calls.append(1)
        return {"wired_bytes": wired_bytes, "physical_memory_bytes": physical_bytes}

    observe.calls = calls  # type: ignore[attr-defined]
    return observe


def _artifact(tmp_path: Path, name: str = "fixture", size: int = 1024) -> ModelArtifactBinding:
    model = tmp_path / f"{name}.bin"
    model.write_bytes(b"\0" * size)
    return ModelArtifactBinding(artifact_id=name, local_path=str(model), format="bin")


def _core(guard: LoadMemoryGuard) -> RuntimeCore:
    return RuntimeCore(host_profile=HostProfile.mock_windows(), load_memory_guard=guard)


def test_load_is_refused_when_wired_memory_plus_model_passes_the_limit(tmp_path: Path) -> None:
    # 64 GiB host, 85% limit = 54.4 GiB; 54.4 GiB already wired leaves no room.
    guard = LoadMemoryGuard(observe=_observer(int(64 * GIB * DEFAULT_LIMIT_RATIO)), exempt_engines=())
    core = _core(guard)
    artifact = _artifact(tmp_path)

    with pytest.raises(HostMemoryInsufficientError) as caught:
        core.load(artifact, adapter="mock")

    details = caught.value.details
    assert details["model_bytes"] == 1024
    assert details["wired_limit_bytes"] == int(64 * GIB * DEFAULT_LIMIT_RATIO)
    health = core.health()
    assert health["lifecycle_state"] == "UNLOADED"
    assert health["loaded_artifact_id"] is None
    assert health["last_error"]["code"] == "host_memory_insufficient"


def test_load_passes_and_reports_the_observation_when_memory_is_available(tmp_path: Path) -> None:
    guard = LoadMemoryGuard(observe=_observer(8 * GIB), exempt_engines=())
    core = _core(guard)

    loaded = core.load(_artifact(tmp_path), adapter="mock")

    assert loaded["lifecycle_state"] == "LOADED"
    preflight = loaded["raw"]["host_memory_preflight"]
    assert preflight["status"] == "passed"
    assert preflight["wired_bytes"] == 8 * GIB
    assert preflight["model_bytes"] == 1024


def test_reusing_the_loaded_model_does_not_observe_memory_again(tmp_path: Path) -> None:
    observe = _observer(8 * GIB)
    core = _core(LoadMemoryGuard(observe=observe, exempt_engines=()))
    artifact = _artifact(tmp_path)

    first = core.load(artifact, adapter="mock", consumer_id="novel")
    second = core.load(artifact, adapter="mock", consumer_id="benchmark")

    assert first["reused"] is False
    assert second["reused"] is True
    assert len(observe.calls) == 1  # type: ignore[attr-defined]


def test_a_different_model_is_refused_as_a_conflict_before_memory_is_observed(tmp_path: Path) -> None:
    observe = _observer(8 * GIB)
    core = _core(LoadMemoryGuard(observe=observe, exempt_engines=()))
    core.load(_artifact(tmp_path, "first"), adapter="mock")

    with pytest.raises(LoadConflictError):
        core.load(_artifact(tmp_path, "second"), adapter="mock")
    assert len(observe.calls) == 1  # type: ignore[attr-defined]


def test_mock_engine_is_exempt_by_default(tmp_path: Path) -> None:
    observe = _observer(64 * GIB)
    core = _core(LoadMemoryGuard(observe=observe))

    loaded = core.load(_artifact(tmp_path), adapter="mock")

    assert loaded["raw"]["host_memory_preflight"] == {"status": "skipped", "reason": "engine_exempt", "engine": "mock"}
    assert observe.calls == []  # type: ignore[attr-defined]


def test_unobservable_host_never_blocks(tmp_path: Path) -> None:
    core = _core(LoadMemoryGuard(observe=lambda: None, exempt_engines=()))

    loaded = core.load(_artifact(tmp_path), adapter="mock")

    assert loaded["raw"]["host_memory_preflight"] == {"status": "skipped", "reason": "unobservable"}


def test_directory_artifacts_and_adapters_are_summed(tmp_path: Path) -> None:
    model_dir = tmp_path / "mlx-model"
    (model_dir / "nested").mkdir(parents=True)
    (model_dir / "weights.safetensors").write_bytes(b"\0" * 300)
    (model_dir / "nested" / "config.json").write_bytes(b"\0" * 20)
    adapter = tmp_path / "adapter.safetensors"
    adapter.write_bytes(b"\0" * 5)
    guard = LoadMemoryGuard(observe=_observer(0), exempt_engines=())

    result = guard.check(engine="mlx", artifact_paths=[str(model_dir), adapter, None])

    assert result["model_bytes"] == 325


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("", DEFAULT_LIMIT_RATIO), ("0.7", 0.7), ("1", 1.0), ("off", None), ("OFF", None)],
)
def test_limit_ratio_comes_from_the_environment(monkeypatch: pytest.MonkeyPatch, raw: str, expected: float | None) -> None:
    monkeypatch.setenv(LIMIT_RATIO_ENV, raw)
    assert LoadMemoryGuard.from_environment().limit_ratio == expected


@pytest.mark.parametrize("raw", ["0", "1.5", "-0.1", "lots"])
def test_invalid_limit_ratio_fails_at_startup(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv(LIMIT_RATIO_ENV, raw)
    with pytest.raises(ValueError, match=LIMIT_RATIO_ENV):
        LoadMemoryGuard.from_environment()


def test_disabled_guard_skips(tmp_path: Path) -> None:
    observe = _observer(64 * GIB)
    result = LoadMemoryGuard(limit_ratio=None, observe=observe, exempt_engines=()).check(
        engine="mlx", artifact_paths=[]
    )
    assert result == {"status": "skipped", "reason": "disabled"}
    assert observe.calls == []  # type: ignore[attr-defined]


def test_service_returns_409_with_the_error_code(tmp_path: Path) -> None:
    guard = LoadMemoryGuard(observe=_observer(60 * GIB), exempt_engines=())
    client = TestClient(create_app(_core(guard)))
    artifact = _artifact(tmp_path)

    response = client.post("/models/load", json={"artifact": artifact.to_dict(), "adapter": "mock"})

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "host_memory_insufficient"
