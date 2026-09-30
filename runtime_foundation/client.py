"""Small Python client for the local Foundation service."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any, Self

import httpx

from .contracts import GenerationRequest, ModelArtifactBinding
from .contracts_v2 import ArtifactBindingV2, ExecutionInputV1, GenerationRequestV2
from .contracts_v3 import CONTRACT_V3_VERSION, GenerationRequestV3, LoadOptions
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

    def __enter__(self) -> Self:
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
            raise RemoteRuntimeError(
                response.status_code,
                {"error": {"code": "invalid_remote_response", "message": "response was not an object"}},
            )
        return payload

    def health(self) -> dict[str, Any]:
        return self._decode(self._client.get("/health"))

    def health_v3(self) -> dict[str, Any]:
        return self._decode(self._client.get("/v3/health"))

    def host(self) -> dict[str, Any]:
        return self._decode(self._client.get("/host"))

    def host_v3(self) -> dict[str, Any]:
        return self._decode(self._client.get("/v3/host"))

    def engines(self) -> dict[str, Any]:
        return self._decode(self._client.get("/engines"))

    def engines_v3(self) -> dict[str, Any]:
        return self._decode(self._client.get("/v3/engines"))

    def capability(self, engine: str) -> dict[str, Any]:
        return self._decode(self._client.get(f"/engines/{engine}/capability"))

    def capability_v3(self, engine: str) -> dict[str, Any]:
        return self._decode(self._client.get(f"/v3/engines/{engine}/capability"))

    def load(
        self,
        artifact: ModelArtifactBinding | ArtifactBindingV2 | ExecutionInputV1 | dict[str, Any] | None = None,
        *,
        execution_input: ExecutionInputV1 | dict[str, Any] | None = None,
        engine: str | None = None,
        consumer_id: str | None = None,
        load_options: LoadOptions | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if artifact is not None and execution_input is not None:
            raise ValueError("provide either artifact or execution_input, not both")
        candidate = execution_input if execution_input is not None else artifact
        if candidate is None:
            raise ValueError("artifact or execution_input is required")
        if isinstance(candidate, ExecutionInputV1):
            body: dict[str, Any] = {"execution_input": candidate.to_dict()}
        elif isinstance(candidate, dict) and candidate.get("schema_version") == "runtime-foundation.execution-input.v1":
            body = {"execution_input": candidate}
        else:
            binding = candidate.to_dict() if isinstance(candidate, (ModelArtifactBinding, ArtifactBindingV2)) else candidate
            body = {"artifact": binding}
        if engine is not None:
            body["engine"] = engine
        if consumer_id is not None:
            body["consumer_id"] = consumer_id
        if load_options is not None:
            body["load_options"] = load_options.to_dict() if isinstance(load_options, LoadOptions) else load_options
        path = "/v3/models/load" if load_options is not None else "/models/load"
        return self._decode(self._client.post(path, json=body))

    def unload(
        self, artifact_id: str | None = None, *, consumer_id: str | None = None, lease_id: str | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if artifact_id is not None:
            body["artifact_id"] = artifact_id
        if consumer_id is not None:
            body["consumer_id"] = consumer_id
        if lease_id is not None:
            body["lease_id"] = lease_id
        return self._decode(self._client.post("/models/unload", json=body))

    def generate(
        self, request: GenerationRequest | GenerationRequestV2 | GenerationRequestV3 | dict[str, Any]
    ) -> dict[str, Any]:
        payload = request.to_dict() if isinstance(request, (GenerationRequest, GenerationRequestV2, GenerationRequestV3)) else request
        is_v3 = isinstance(request, GenerationRequestV3) or (
            isinstance(payload, dict) and payload.get("contract_version") == CONTRACT_V3_VERSION
        )
        return self._decode(self._client.post("/v3/generate" if is_v3 else "/generate", json=payload))

    def stream(
        self, request: GenerationRequest | GenerationRequestV2 | GenerationRequestV3 | dict[str, Any]
    ) -> Iterator[dict[str, Any]]:
        payload = request.to_dict() if isinstance(request, (GenerationRequest, GenerationRequestV2, GenerationRequestV3)) else request
        is_v3 = isinstance(request, GenerationRequestV3) or (
            isinstance(payload, dict) and payload.get("contract_version") == CONTRACT_V3_VERSION
        )
        path = "/v3/generate/stream" if is_v3 else "/generate/stream"
        with self._client.stream("POST", path, json=payload) as response:
            if response.status_code >= 400:
                raise RemoteRuntimeError(response.status_code, response.json())
            for line in response.iter_lines():
                if not line:
                    continue
                yield json.loads(line)

    def cancel(self, request_id: str) -> dict[str, Any]:
        return self._decode(self._client.post(f"/requests/{request_id}/cancel"))

    def cancel_v3(self, request_id: str) -> dict[str, Any]:
        return self._decode(self._client.post(f"/v3/requests/{request_id}/cancel"))

    def runtime_metrics(self) -> dict[str, Any]:
        return self._decode(self._client.get("/runtime/metrics"))

    def execution(self, execution_id: str) -> dict[str, Any]:
        return self._decode(self._client.get(f"/executions/{execution_id}"))

    def execution_v3(self, execution_id: str) -> dict[str, Any]:
        return self._decode(self._client.get(f"/v3/executions/{execution_id}"))
