from __future__ import annotations

import json

import httpx
import pytest

from runtime_foundation import (
    CONTRACT_V2_VERSION,
    ArtifactBindingV2,
    ArtifactLocator,
    GenerationRequestV2,
    ModelArtifactBinding,
    ThinkingIntent,
    ThinkingMode,
)
from runtime_foundation.client import LocalRuntimeClient, RemoteRuntimeError


@pytest.mark.parametrize("base_url", ["http://0.0.0.0:8765", "http://192.168.1.10:8765", "http://example.com"])
def test_client_rejects_non_loopback_base_url(base_url: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        LocalRuntimeClient(base_url=base_url)


@pytest.mark.parametrize("base_url", ["http://localhost:8765", "http://[::1]:8765"])
def test_client_accepts_loopback_base_url(base_url: str) -> None:
    with LocalRuntimeClient(
        http_client=httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"ok": True}))),
        base_url=base_url,
    ):
        pass


def test_client_preserves_http_contract_and_stream_events() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"contract_version": "runtime-foundation.contract.v1", "status": "ready"})
        if request.url.path == "/generate":
            return httpx.Response(200, json={"engine": "mock", "text": "ok"})
        if request.url.path == "/generate/stream":
            body = '{"type":"started"}\n{"type":"completed","result":{"text":"ok"}}\n'
            return httpx.Response(200, content=body.encode(), headers={"content-type": "application/x-ndjson"})
        if request.url.path == "/runtime/metrics":
            return httpx.Response(200, json={"contract_version": "runtime-foundation.contract.v1", "adapters": []})
        return httpx.Response(404, json={"error": {"code": "not_found", "message": "missing", "details": {}}})

    transport = httpx.MockTransport(handler)
    with LocalRuntimeClient(http_client=httpx.Client(transport=transport), base_url="http://127.0.0.1") as client:
        assert client.health()["status"] == "ready"
        assert client.generate({"messages": []})["engine"] == "mock"
        assert [event["type"] for event in client.stream({"messages": []})] == ["started", "completed"]
        assert client.runtime_metrics()["adapters"] == []


def test_client_load_serializes_v1_v2_and_raw_artifacts() -> None:
    received: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/models/load":
            received.append(json.loads(request.content)["artifact"])
            return httpx.Response(200, json={"loaded": True})
        return httpx.Response(404, json={"error": {"code": "not_found", "message": "missing", "details": {}}})

    v1 = ModelArtifactBinding("v1", "C:/models/v1.bin", "bin")
    v2 = ArtifactBindingV2("v2", ArtifactLocator("filesystem", "C:/models/v2.bin"), format="bin")
    raw = {"artifact_id": "raw", "local_path": "C:/models/raw.bin", "format": "bin"}
    with LocalRuntimeClient(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)), base_url="http://127.0.0.1"
    ) as client:
        assert client.load(v1) == {"loaded": True}
        assert client.load(v2) == {"loaded": True}
        assert client.load(raw) == {"loaded": True}

    assert received == [v1.to_dict(), v2.to_dict(), raw]


def test_client_generate_serializes_generation_request_v2() -> None:
    received: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(json.loads(request.content))
        return httpx.Response(200, json={"ok": True})

    request = GenerationRequestV2(
        model_artifact_id="v2-model",
        messages=[{"role": "user", "content": "hello"}],
        thinking_intent=ThinkingIntent(mode=ThinkingMode.OFF),
    )
    with LocalRuntimeClient(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)), base_url="http://127.0.0.1"
    ) as client:
        assert client.generate(request) == {"ok": True}

    assert received[0]["contract_version"] == CONTRACT_V2_VERSION
    assert received[0]["schema_version"] == "runtime-foundation.generation-request.v2"
    assert received[0]["thinking_intent"] == {"mode": "OFF", "effort": None, "budget_tokens": None}


def test_client_stream_serializes_generation_request_v2() -> None:
    received: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append(json.loads(request.content))
        body = '{"type":"started"}\n{"type":"completed"}\n'
        return httpx.Response(200, content=body.encode(), headers={"content-type": "application/x-ndjson"})

    request = GenerationRequestV2(
        model_artifact_id="v2-model",
        messages=[{"role": "user", "content": "hello"}],
        thinking_intent=ThinkingIntent(mode=ThinkingMode.OFF),
    )
    with LocalRuntimeClient(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)), base_url="http://127.0.0.1"
    ) as client:
        assert [event["type"] for event in client.stream(request)] == ["started", "completed"]

    assert received[0]["contract_version"] == CONTRACT_V2_VERSION
    assert received[0]["schema_version"] == "runtime-foundation.generation-request.v2"


def test_client_exposes_remote_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503, json={"error": {"code": "engine_unavailable", "message": "MLX unavailable", "details": {}}}
        )

    with LocalRuntimeClient(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)), base_url="http://127.0.0.1"
    ) as client:
        try:
            client.health()
        except RemoteRuntimeError as exc:
            assert exc.status_code == 503
            assert exc.code == "engine_unavailable"
            assert exc.retryable is False
        else:  # pragma: no cover
            raise AssertionError("expected RemoteRuntimeError")


def test_client_preserves_retryable_runtime_timeout_error() -> None:
    details = {"timeout_ms": 25, "timeout_semantics": "cooperative", "execution_id": "execution-timeout"}

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            504,
            json={
                "error": {
                    "code": "runtime_timeout",
                    "message": "generation exceeded the runtime deadline",
                    "retryable": True,
                    "details": details,
                }
            },
        )

    with (
        LocalRuntimeClient(
            http_client=httpx.Client(transport=httpx.MockTransport(handler)), base_url="http://127.0.0.1"
        ) as client,
        pytest.raises(RemoteRuntimeError) as caught,
    ):
        client.generate({"messages": []})

    error = caught.value
    assert error.status_code == 504
    assert error.code == "runtime_timeout"
    assert error.message == "generation exceeded the runtime deadline"
    assert error.retryable is True
    assert error.details == details


@pytest.mark.parametrize("wire_value", ["true", 1, None])
def test_client_does_not_coerce_malformed_retryable(wire_value: object) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            504,
            json={
                "error": {
                    "code": "runtime_timeout",
                    "message": "timeout",
                    "retryable": wire_value,
                    "details": {"execution_id": "execution-malformed"},
                }
            },
        )

    with (
        LocalRuntimeClient(
            http_client=httpx.Client(transport=httpx.MockTransport(handler)), base_url="http://127.0.0.1"
        ) as client,
        pytest.raises(RemoteRuntimeError) as caught,
    ):
        client.generate({"messages": []})

    assert caught.value.retryable is False


def test_client_preserves_non_retryable_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": "context_length_exceeded",
                    "message": "context budget exceeded",
                    "retryable": False,
                    "details": {"execution_id": "execution-context"},
                }
            },
        )

    with (
        LocalRuntimeClient(
            http_client=httpx.Client(transport=httpx.MockTransport(handler)), base_url="http://127.0.0.1"
        ) as client,
        pytest.raises(RemoteRuntimeError) as caught,
    ):
        client.generate({"messages": []})

    assert caught.value.code == "context_length_exceeded"
    assert caught.value.retryable is False
