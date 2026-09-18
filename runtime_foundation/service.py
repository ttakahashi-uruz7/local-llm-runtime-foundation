"""Loopback HTTP facade for the Foundation Core."""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .contracts import CONTRACT_VERSION, GenerationRequest, ModelArtifactBinding
from .contracts_v2 import CONTRACT_V2_VERSION, ArtifactBindingV2, GenerationRequestV2
from .core import RuntimeCore
from .errors import InvalidRequestError, RuntimeFoundationError
from .network import validate_loopback_host
from .version import FOUNDATION_VERSION


def _artifact_from_body(body: dict[str, Any]) -> ModelArtifactBinding:
    candidate = body.get("artifact")
    if candidate is None:
        candidate = {
            key: body[key]
            for key in (
                "artifact_id",
                "local_path",
                "format",
                "quantization",
                "artifact_hash",
                "revision",
                "metadata",
                "contract_version",
                "registry_identity",
                "content_identity",
                "locator",
            )
            if key in body
        }
    try:
        if isinstance(candidate, dict) and (
            candidate.get("contract_version") == CONTRACT_V2_VERSION
            or "registry_identity" in candidate
            or "content_identity" in candidate
            or "locator" in candidate
        ):
            return ArtifactBindingV2.from_payload(candidate).to_legacy()
        return ModelArtifactBinding.from_payload(candidate)
    except (RuntimeFoundationError, ValueError) as exc:
        if isinstance(exc, RuntimeFoundationError):
            raise
        raise InvalidRequestError(str(exc)) from exc


def _generation_request(body: dict[str, Any]) -> GenerationRequest:
    try:
        if (
            body.get("contract_version") == CONTRACT_V2_VERSION
            or body.get("schema_version") == "runtime-foundation.generation-request.v2"
            or "thinking_intent" in body
            or "execution_guard" in body
        ):
            return GenerationRequestV2.from_payload(body)
        return GenerationRequest.from_payload(body)
    except (RuntimeFoundationError, ValueError) as exc:
        if isinstance(exc, RuntimeFoundationError):
            raise
        raise InvalidRequestError(str(exc)) from exc


def create_app(core: RuntimeCore | None = None) -> FastAPI:
    runtime = core or RuntimeCore()
    app = FastAPI(
        title="Local LLM Runtime Foundation",
        version=FOUNDATION_VERSION,
        description="Policy-free local execution service for MLX and future compatible engines.",
    )
    app.state.core = runtime

    @app.exception_handler(RuntimeFoundationError)
    async def foundation_error_handler(_: Request, exc: RuntimeFoundationError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status,
            content={"contract_version": CONTRACT_VERSION, "error": exc.as_dict()},
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return runtime.health()

    @app.get("/host")
    def host() -> dict[str, Any]:
        return runtime.host()

    @app.get("/engines")
    def engines() -> dict[str, Any]:
        return {"contract_version": CONTRACT_VERSION, "engines": runtime.capabilities()}

    @app.get("/engines/{engine}/capability")
    def capability(engine: str) -> dict[str, Any]:
        return runtime.capability(engine)

    @app.post("/models/load")
    def load_model(body: dict[str, Any]) -> dict[str, Any]:
        artifact = _artifact_from_body(body)
        return runtime.load(
            artifact,
            adapter=body.get("engine") or body.get("adapter"),
            consumer_id=body.get("consumer_id"),
        )

    @app.post("/models/unload")
    def unload_model(body: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = body or {}
        return runtime.unload(
            payload.get("artifact_id") or payload.get("model_artifact_id"),
            consumer_id=payload.get("consumer_id"),
            lease_id=payload.get("lease_id"),
        )

    @app.post("/generate")
    def generate(body: dict[str, Any]) -> dict[str, Any]:
        return runtime.generate(_generation_request(body)).to_dict()

    @app.post("/generate/stream")
    def generate_stream(body: dict[str, Any]) -> StreamingResponse:
        request = _generation_request(body)
        events = runtime.stream(request)

        def lines():
            for event in events:
                yield json.dumps(event.to_dict(), ensure_ascii=False) + "\n"

        return StreamingResponse(lines(), media_type="application/x-ndjson")

    @app.post("/requests/{request_id}/cancel")
    def cancel(request_id: str) -> dict[str, Any]:
        return runtime.cancel(request_id)

    @app.get("/runtime/metrics")
    def metrics() -> dict[str, Any]:
        return runtime.runtime_metrics()

    @app.get("/executions/{execution_id}")
    def execution(execution_id: str) -> dict[str, Any]:
        value = runtime.get_execution(execution_id)
        if value is None:
            raise InvalidRequestError("execution was not found", details={"execution_id": execution_id})
        return value

    return app


app = create_app()


def main() -> None:
    import uvicorn

    host = validate_loopback_host(os.environ.get("RUNTIME_FOUNDATION_HOST", "127.0.0.1"))
    uvicorn.run(
        app,
        host=host,
        port=int(os.environ.get("RUNTIME_FOUNDATION_PORT", "8765")),
    )


if __name__ == "__main__":
    main()
