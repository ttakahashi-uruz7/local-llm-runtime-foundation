from __future__ import annotations

import httpx
import pytest

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


def test_client_exposes_remote_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"code": "engine_unavailable", "message": "MLX unavailable", "details": {}}})

    with LocalRuntimeClient(http_client=httpx.Client(transport=httpx.MockTransport(handler)), base_url="http://127.0.0.1") as client:
        try:
            client.health()
        except RemoteRuntimeError as exc:
            assert exc.status_code == 503
            assert exc.code == "engine_unavailable"
        else:  # pragma: no cover
            raise AssertionError("expected RemoteRuntimeError")
