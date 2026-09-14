"""Loopback HTTP facade for the Foundation Core."""

from __future__ import annotations

import ipaddress
import json
import os
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .contracts import CONTRACT_VERSION, GenerationRequest, ModelArtifactBinding
from .core import RuntimeCore
from .errors import InvalidRequestError, RuntimeFoundationError
from .version import FOUNDATION_VERSION


def validate_loopback_host(value: str | None) -> str:
    """Allow only loopback bind addresses for the unauthenticated v1 service."""

    host = (value or "127.0.0.1").strip()
    if host.lower() == "localhost":
        return host
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError(
            "RUNTIME_FOUNDATION_HOST must be localhost, 127.0.0.1, ::1, or another loopback address"
        ) from exc
    if not address.is_loopback:
        raise ValueError(
            "RUNTIME_FOUNDATION_HOST must be a loopback address; remote/LAN binding is not supported in Foundation v1"
        )
    return host


def _artifact_from_body(body: dict[str, Any]) -> ModelArtifactBinding:
    candidate = body.get("artifact")
    if candidate is None:
        candidate = {key: body[key] for key in ("artifact_id", "local_path", "format", "quantization", "artifact_hash", "revision", "metadata") if key in body}
    try:
        return ModelArtifactBinding.from_payload(candidate)
    except (RuntimeFoundationError, ValueError) as exc:
        if isinstance(exc, RuntimeFoundationError):
            raise
        raise InvalidRequestError(str(exc)) from exc


def _generation_request(body: dict[str, Any]) -> GenerationRequest:
    try:
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
