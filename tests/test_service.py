from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from runtime_foundation import (
    ArtifactBindingV2,
    ArtifactLocator,
    ContentIdentity,
    HostProfile,
    ModelArtifactBinding,
    RuntimeCore,
)
from runtime_foundation.adapters.mock import MockAdapter
from runtime_foundation.client import LocalRuntimeClient, RemoteRuntimeError
from runtime_foundation.errors import RequestCancelledError
from runtime_foundation.service import create_app, validate_loopback_host


class ServiceTimeoutCancellationAdapter(MockAdapter):
    def generate(self, request, cancel_event):
        del request
        assert cancel_event.wait(1.0), "Core timeout did not request adapter cancellation"
        raise RequestCancelledError("adapter observed Core timeout cancellation")


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
    load = client.post(
        "/models/load", json={"artifact": artifact.to_dict(), "engine": "mock", "consumer_id": "benchmark"}
    )
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

    unload = client.post(
        "/models/unload", json={"artifact_id": "service", "consumer_id": "benchmark", "lease_id": lease_id}
    )
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


def test_service_exposes_core_timeout_as_http_504(tmp_path: Path) -> None:
    model = tmp_path / "service-timeout.bin"
    model.write_bytes(b"service timeout fixture")
    artifact = ModelArtifactBinding("service-timeout", str(model), "bin")
    core = RuntimeCore(
        host_profile=HostProfile.mock_windows(),
        adapters={"mock": ServiceTimeoutCancellationAdapter()},
    )
    client = TestClient(create_app(core))
    loaded = client.post("/models/load", json={"artifact": artifact.to_dict(), "engine": "mock"})
    assert loaded.status_code == 200

    response = client.post(
        "/generate",
        json={
            "model_artifact_id": artifact.artifact_id,
            "lease_id": loaded.json()["lease_id"],
            "messages": [{"role": "user", "content": "service timeout"}],
            "timeout_ms": 5,
        },
    )

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "runtime_timeout"
    assert response.json()["error"]["details"]["execution_id"]
    assert client.get("/health").json()["last_error"]["code"] == "runtime_timeout"


def test_client_service_round_trip_preserves_retryable_timeout(tmp_path: Path) -> None:
    model = tmp_path / "client-service-timeout.bin"
    model.write_bytes(b"client service timeout fixture")
    artifact = ModelArtifactBinding("client-service-timeout", str(model), "bin")
    core = RuntimeCore(
        host_profile=HostProfile.mock_windows(),
        adapters={"mock": ServiceTimeoutCancellationAdapter()},
    )
    service_client = TestClient(create_app(core))

    with LocalRuntimeClient(http_client=service_client, base_url="http://127.0.0.1") as client:
        loaded = client.load(artifact, engine="mock")
        with pytest.raises(RemoteRuntimeError) as caught:
            client.generate(
                {
                    "model_artifact_id": artifact.artifact_id,
                    "lease_id": loaded["lease_id"],
                    "messages": [{"role": "user", "content": "client service timeout"}],
                    "timeout_ms": 5,
                }
            )

    error = caught.value
    assert error.status_code == 504
    assert error.code == "runtime_timeout"
    assert error.retryable is True
    assert error.details["timeout_ms"] == 5
    assert error.details["timeout_semantics"] == "cooperative"
    assert error.details["execution_id"]


def test_client_load_v2_artifact_reaches_service_and_succeeds(tmp_path: Path) -> None:
    model = tmp_path / "client-v2-artifact.bin"
    model.write_bytes(b"client v2 artifact fixture")
    artifact = ArtifactBindingV2(
        "client-v2-artifact",
        ArtifactLocator("filesystem", str(model)),
        ContentIdentity.from_file(model),
        format="bin",
    )
    core = RuntimeCore(host_profile=HostProfile.mock_windows())
    service_client = TestClient(create_app(core))

    with LocalRuntimeClient(http_client=service_client, base_url="http://127.0.0.1") as client:
        loaded = client.load(artifact, engine="mock", consumer_id="v2-client")

    assert loaded["lifecycle_state"] == "LOADED"
    assert loaded["artifact"]["artifact_id"] == artifact.artifact_id
    assert loaded["consumer_id"] == "v2-client"
