"""Loopback HTTP facade for the Foundation Core."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from typing import Any

import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .contracts import CONTRACT_VERSION, GenerationRequest, ModelArtifactBinding
from .contracts_v2 import CONTRACT_V2_VERSION, ArtifactBindingV2, ExecutionInputV1, GenerationRequestV2
from .contracts_v3 import CONTRACT_V3_VERSION, GenerationRequestV3, LoadOptions
from .core import RuntimeCore
from .errors import InvalidRequestError, RuntimeFoundationError
from .instance_lock import InstanceAlreadyRunningError, acquire_instance_lock, lock_path_from_environment
from .network import validate_loopback_host
from .version import FOUNDATION_VERSION


class _EventStreamResponse(StreamingResponse):
    """NDJSON stream that always closes its event stream when the response ends.

    Starlette leaves a synchronous iterator suspended when the client
    disconnects, and does not iterate it at all when the client left while
    the request was queued. Closing it here releases the generation slot
    and stops the engine instead of relying on garbage collection.
    """

    def __init__(self, events: Iterable[Any]) -> None:
        self._events = events
        super().__init__(self._lines(events), media_type="application/x-ndjson")

    @staticmethod
    def _lines(events: Iterable[Any]) -> Iterator[str]:
        for event in events:
            yield json.dumps(event.to_dict(), ensure_ascii=False) + "\n"

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            close = getattr(self._events, "close", None)
            if callable(close):
                # Requests waiting for the generation slot hold workers from the
                # default thread limiter; the cleanup that frees the slot must
                # not wait behind them.
                with anyio.CancelScope(shield=True):
                    await anyio.to_thread.run_sync(close, limiter=anyio.CapacityLimiter(1))


def _artifact_from_body(body: dict[str, Any]) -> ModelArtifactBinding | ArtifactBindingV2 | ExecutionInputV1:
    if "execution_input" in body:
        if "artifact" in body or "artifact_id" in body or "local_path" in body:
            raise InvalidRequestError("load request must contain either artifact or execution_input, not both")
        try:
            return ExecutionInputV1.from_payload(body["execution_input"])
        except (RuntimeFoundationError, ValueError) as exc:
            if isinstance(exc, RuntimeFoundationError):
                raise
            raise InvalidRequestError(str(exc)) from exc
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
            return ArtifactBindingV2.from_payload(candidate)
        return ModelArtifactBinding.from_payload(candidate)
    except (RuntimeFoundationError, ValueError) as exc:
        if isinstance(exc, RuntimeFoundationError):
            raise
        raise InvalidRequestError(str(exc)) from exc


def _generation_request(body: dict[str, Any]) -> GenerationRequest | GenerationRequestV2 | GenerationRequestV3:
    try:
        if (
            body.get("contract_version") == CONTRACT_V3_VERSION
            or body.get("schema_version") == "runtime-foundation.generation-request.v3"
        ):
            return GenerationRequestV3.from_payload(body)
        if (
            body.get("contract_version") == CONTRACT_V2_VERSION
            or body.get("schema_version") == "runtime-foundation.generation-request.v2"
            or "thinking_intent" in body
            or "execution_guard" in body
            or "execution_input" in body
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
            content={
                "contract_version": CONTRACT_V3_VERSION
                if _.url.path.startswith("/v3/")
                else CONTRACT_VERSION,
                "error": exc.as_dict(),
            },
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return runtime.health()

    @app.get("/v3/health")
    def health_v3() -> dict[str, Any]:
        return runtime.health_v3()

    @app.get("/host")
    def host() -> dict[str, Any]:
        return runtime.host()

    @app.get("/v3/host")
    def host_v3() -> dict[str, Any]:
        return runtime.host_v3()

    @app.get("/engines")
    def engines() -> dict[str, Any]:
        return {"contract_version": CONTRACT_VERSION, "engines": runtime.capabilities()}

    @app.get("/v3/engines")
    def engines_v3() -> dict[str, Any]:
        return {"contract_version": "runtime-foundation.contract.v3", "engines": runtime.capabilities_v3()}

    @app.get("/engines/{engine}/capability")
    def capability(engine: str) -> dict[str, Any]:
        return runtime.capability(engine)

    @app.get("/v3/engines/{engine}/capability")
    def capability_v3(engine: str) -> dict[str, Any]:
        return runtime.capability_v3(engine)

    @app.post("/models/load")
    @app.post("/v3/models/load")
    def load_model(body: dict[str, Any], request: Request) -> dict[str, Any]:
        if request.url.path.startswith("/v3/") and body.get("load_options") is None:
            body = {**body, "load_options": LoadOptions().to_dict()}
        artifact = _artifact_from_body(body)
        if isinstance(artifact, ExecutionInputV1):
            result = runtime.load(
                execution_input=artifact,
                adapter=body.get("engine") or body.get("adapter"),
                consumer_id=body.get("consumer_id"),
                load_options=body.get("load_options"),
            )
        else:
            result = runtime.load(
                artifact,
                adapter=body.get("engine") or body.get("adapter"),
                consumer_id=body.get("consumer_id"),
                load_options=body.get("load_options"),
            )
        if request.url.path.startswith("/v3/"):
            resolution = result.get("raw", {}).get("load_options_resolution")
            result["contract_version"] = CONTRACT_V3_VERSION
            result["schema_version"] = "runtime-foundation.load-result.v3"
            result["load_identity"] = result.get("raw", {}).get("load_identity")
            result["load_options_evidence"] = (
                {
                    "requested": resolution.get("requested"),
                    "resolved": resolution.get("resolved"),
                    "effective": resolution.get("effective"),
                    "resolutions": resolution.get("resolutions", []),
                }
                if resolution
                else None
            )
        return result

    @app.post("/models/unload")
    @app.post("/v3/models/unload")
    def unload_model(body: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = body or {}
        return runtime.unload(
            payload.get("artifact_id") or payload.get("model_artifact_id"),
            consumer_id=payload.get("consumer_id"),
            lease_id=payload.get("lease_id"),
        )

    @app.post("/generate")
    def generate(body: dict[str, Any]) -> dict[str, Any]:
        request = _generation_request(body)
        if isinstance(request, GenerationRequestV3):
            return runtime.generate_v3(request).to_dict()
        return runtime.generate(request).to_dict()

    @app.post("/v3/generate")
    def generate_v3(body: dict[str, Any]) -> dict[str, Any]:
        request = _generation_request(body)
        if not isinstance(request, GenerationRequestV3):
            raise InvalidRequestError("/v3/generate requires a v3 GenerationRequest")
        return runtime.generate_v3(request).to_dict()

    @app.post("/generate/stream")
    def generate_stream(body: dict[str, Any]) -> StreamingResponse:
        request = _generation_request(body)
        if isinstance(request, GenerationRequestV3):
            return _EventStreamResponse(runtime.stream_v3(request))
        return _EventStreamResponse(runtime.stream(request))

    @app.post("/v3/generate/stream")
    def generate_stream_v3(body: dict[str, Any]) -> StreamingResponse:
        request = _generation_request(body)
        if not isinstance(request, GenerationRequestV3):
            raise InvalidRequestError("/v3/generate/stream requires a v3 GenerationRequest")
        return _EventStreamResponse(runtime.stream_v3(request))

    @app.post("/requests/{request_id}/cancel")
    def cancel(request_id: str) -> dict[str, Any]:
        return runtime.cancel(request_id)

    @app.post("/v3/requests/{request_id}/cancel")
    def cancel_v3(request_id: str) -> dict[str, Any]:
        return {**runtime.cancel(request_id), "contract_version": CONTRACT_V3_VERSION}

    @app.get("/runtime/metrics")
    def metrics() -> dict[str, Any]:
        return runtime.runtime_metrics()

    @app.get("/executions/{execution_id}")
    def execution(execution_id: str) -> dict[str, Any]:
        value = runtime.get_execution(execution_id)
        if value is None:
            raise InvalidRequestError("execution was not found", details={"execution_id": execution_id})
        return value

    @app.get("/v3/executions/{execution_id}")
    def execution_v3(execution_id: str) -> dict[str, Any]:
        value = runtime.get_execution_v3(execution_id)
        if value is None:
            raise InvalidRequestError("execution was not found", details={"execution_id": execution_id})
        return value

    return app


app = create_app()


def main() -> None:
    import uvicorn

    host = validate_loopback_host(os.environ.get("RUNTIME_FOUNDATION_HOST", "127.0.0.1"))
    port = int(os.environ.get("RUNTIME_FOUNDATION_PORT", "8765"))
    lock_path = lock_path_from_environment()
    lock_handle = None
    if lock_path is not None:
        try:
            lock_handle = acquire_instance_lock(lock_path, description=f"pid={os.getpid()} url=http://{host}:{port}")
        except InstanceAlreadyRunningError as exc:
            raise SystemExit(str(exc)) from exc
    try:
        uvicorn.run(app, host=host, port=port)
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    main()
