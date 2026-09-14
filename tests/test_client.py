from __future__ import annotations

import httpx

from runtime_foundation.client import LocalRuntimeClient, RemoteRuntimeError


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
    with LocalRuntimeClient(http_client=httpx.Client(transport=transport), base_url="http://test") as client:
        assert client.health()["status"] == "ready"
        assert client.generate({"messages": []})["engine"] == "mock"
        assert [event["type"] for event in client.stream({"messages": []})] == ["started", "completed"]
        assert client.runtime_metrics()["adapters"] == []


def test_client_exposes_remote_error() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": {"code": "engine_unavailable", "message": "MLX unavailable", "details": {}}})

    with LocalRuntimeClient(http_client=httpx.Client(transport=httpx.MockTransport(handler)), base_url="http://test") as client:
        try:
            client.health()
        except RemoteRuntimeError as exc:
            assert exc.status_code == 503
            assert exc.code == "engine_unavailable"
        else:  # pragma: no cover
            raise AssertionError("expected RemoteRuntimeError")
