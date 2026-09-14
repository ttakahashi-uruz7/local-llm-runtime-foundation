from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from runtime_foundation import HostProfile, ModelArtifactBinding, RuntimeCore
from runtime_foundation.service import create_app, validate_loopback_host


@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "localhost", "127.0.0.2"])
def test_service_bind_host_accepts_loopback(host: str) -> None:
    assert validate_loopback_host(host) == host


@pytest.mark.parametrize("host", ["0.0.0.0", "192.168.1.10", "8.8.8.8", "example.local"])
def test_service_bind_host_rejects_non_loopback(host: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        validate_loopback_host(host)


def test_service_api_exposes_contract_version_and_raw_runtime_boundary(tmp_path: Path) -> None:
    model = tmp_path / "service-model.bin"
    model.write_bytes(b"service fixture")
    core = RuntimeCore(host_profile=HostProfile.mock_windows())
    client = TestClient(create_app(core))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["contract_version"] == "runtime-foundation.contract.v1"
    assert health.json()["lifecycle_state"] == "UNLOADED"

    host = client.get("/host")
    assert host.status_code == 200
    assert host.json()["platform"] == "windows"

    engines = client.get("/engines")
    assert engines.status_code == 200
    assert {item["engine"] for item in engines.json()["engines"]} == {"mock", "mlx", "llama.cpp"}

    artifact = ModelArtifactBinding("service", str(model), "bin")
    load = client.post("/models/load", json={"artifact": artifact.to_dict(), "engine": "mock", "consumer_id": "benchmark"})
    assert load.status_code == 200
    lease_id = load.json()["lease_id"]

    generated = client.post(
        "/generate",
        json={
            "model_artifact_id": "service",
            "consumer_id": "benchmark",
            "lease_id": lease_id,
            "messages": [{"role": "user", "content": "HTTP contract"}],
        },
    )
    assert generated.status_code == 200
    result = generated.json()
    assert result["engine"] == "mock"
    assert result["trace"]["status"] == "completed"

    execution = client.get(f"/executions/{result['execution_id']}")
    assert execution.status_code == 200
    assert execution.json()["execution_id"] == result["execution_id"]

    streamed = client.post(
        "/generate/stream",
        json={
            "model_artifact_id": "service",
            "consumer_id": "benchmark",
            "lease_id": lease_id,
            "messages": [{"role": "user", "content": "stream HTTP"}],
        },
    )
    assert streamed.status_code == 200
    events = [json.loads(line) for line in streamed.text.splitlines()]
    assert events[0]["type"] == "started"
    assert events[-1]["type"] == "completed"
    assert events[-1]["result"]["execution_id"]

    bad = client.post(
        "/generate",
        json={
            "model_artifact_id": "service",
            "consumer_id": "benchmark",
            "lease_id": lease_id,
            "messages": [{"role": "user", "content": "bad"}],
            "runtime_options": {"acceleration": {"backend": "metal"}},
        },
    )
    assert bad.status_code == 400
    assert bad.json()["error"]["code"] == "unsupported_runtime_option"

    unload = client.post("/models/unload", json={"artifact_id": "service", "consumer_id": "benchmark", "lease_id": lease_id})
    assert unload.status_code == 200
    assert unload.json()["unloaded"] is True


def test_service_missing_artifact_has_stable_error_shape(tmp_path: Path) -> None:
    client = TestClient(create_app(RuntimeCore(host_profile=HostProfile.mock_windows())))
    response = client.post(
        "/models/load",
        json={
            "artifact": {"artifact_id": "missing", "local_path": str(tmp_path / "missing"), "format": "bin"},
            "engine": "mock",
        },
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "artifact_not_found"
