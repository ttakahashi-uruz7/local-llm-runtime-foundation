"""Small Python client for the local Foundation service."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from .contracts import GenerationRequest, ModelArtifactBinding
from .network import validate_loopback_url


class RemoteRuntimeError(Exception):
    """Wire error returned by a Foundation service."""

    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        error = payload.get("error") or payload.get("detail") or payload
        if not isinstance(error, dict):
            error = {"code": "remote_runtime_error", "message": str(error), "details": {}}
        self.status_code = status_code
        self.code = str(error.get("code", "remote_runtime_error"))
        self.message = str(error.get("message", "remote runtime request failed"))
        retryable = error.get("retryable", False)
        self.retryable = retryable if isinstance(retryable, bool) else False
        self.details = dict(error.get("details") or {})
        super().__init__(self.message)


class LocalRuntimeClient:
    """Contract client; it never falls back to cloud or provider APIs."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8765",
        *,
        timeout: float | None = 60.0,
        http_client: httpx.Client | None = None,
    ) -> None:
        base_url = validate_loopback_url(base_url)
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        if http_client is not None:
            self._client.base_url = httpx.URL(base_url.rstrip("/"))

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "LocalRuntimeClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _decode(self, response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError:
            payload = {"error": {"code": "invalid_remote_response", "message": response.text, "details": {}}}
        if response.status_code >= 400:
            raise RemoteRuntimeError(response.status_code, payload)
        if not isinstance(payload, dict):
            raise RemoteRuntimeError(response.status_code, {"error": {"code": "invalid_remote_response", "message": "response was not an object"}})
        return payload

    def health(self) -> dict[str, Any]:
        return self._decode(self._client.get("/health"))

    def host(self) -> dict[str, Any]:
        return self._decode(self._client.get("/host"))

    def engines(self) -> dict[str, Any]:
        return self._decode(self._client.get("/engines"))

    def capability(self, engine: str) -> dict[str, Any]:
        return self._decode(self._client.get(f"/engines/{engine}/capability"))

    def load(
        self,
        artifact: ModelArtifactBinding | dict[str, Any],
        *,
        engine: str | None = None,
        consumer_id: str | None = None,
    ) -> dict[str, Any]:
        binding = artifact.to_dict() if isinstance(artifact, ModelArtifactBinding) else artifact
        body: dict[str, Any] = {"artifact": binding}
        if engine is not None:
            body["engine"] = engine
        if consumer_id is not None:
            body["consumer_id"] = consumer_id
        return self._decode(self._client.post("/models/load", json=body))

    def unload(self, artifact_id: str | None = None, *, consumer_id: str | None = None, lease_id: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if artifact_id is not None:
            body["artifact_id"] = artifact_id
        if consumer_id is not None:
            body["consumer_id"] = consumer_id
        if lease_id is not None:
            body["lease_id"] = lease_id
        return self._decode(self._client.post("/models/unload", json=body))

    def generate(self, request: GenerationRequest | dict[str, Any]) -> dict[str, Any]:
        payload = request.to_dict() if isinstance(request, GenerationRequest) else request
        return self._decode(self._client.post("/generate", json=payload))

    def stream(self, request: GenerationRequest | dict[str, Any]) -> Iterator[dict[str, Any]]:
        payload = request.to_dict() if isinstance(request, GenerationRequest) else request
        with self._client.stream("POST", "/generate/stream", json=payload) as response:
            if response.status_code >= 400:
                raise RemoteRuntimeError(response.status_code, response.json())
            for line in response.iter_lines():
                if not line:
                    continue
                yield json.loads(line)

    def cancel(self, request_id: str) -> dict[str, Any]:
        return self._decode(self._client.post(f"/requests/{request_id}/cancel"))

    def runtime_metrics(self) -> dict[str, Any]:
        return self._decode(self._client.get("/runtime/metrics"))

    def execution(self, execution_id: str) -> dict[str, Any]:
        return self._decode(self._client.get(f"/executions/{execution_id}"))
