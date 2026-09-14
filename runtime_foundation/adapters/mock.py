"""Deterministic development adapter for Windows and CI."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

from ..contracts import (
    AccelerationSettings,
    EngineCapability,
    EngineIdentity,
    GenerationRequest,
    GenerationResult,
    KVCacheSettings,
    ModelArtifactBinding,
    PrefillSettings,
    RUNTIME_OPTION_PATHS,
    RuntimeMetrics,
    RuntimeOptions,
    RuntimeSettingsResolution,
    StreamEvent,
    TokenUsage,
    utc_now,
)
from ..errors import (
    ArtifactNotFoundError,
    ContextLengthExceededError,
    EngineRuntimeError,
    InvalidRuntimeOptionError,
    RequestCancelledError,
    RuntimeTimeoutError,
    UnsupportedRuntimeOptionError,
)
from .base import EngineAdapter


class MockAdapter(EngineAdapter):
    """A contract test double; its timing and memory values are not production observations."""

    name = "mock"

    _engine_options = {
        "mock.output_prefix": {"type": "string", "default": "Mock response"},
        "mock.chunk_delay_ms": {"type": "integer", "default": 1, "minimum": 0, "maximum": 1000},
    }

    def __init__(self) -> None:
        self._loaded: ModelArtifactBinding | None = None
        self._lock = threading.RLock()
        self._active: dict[str, threading.Event] = {}
        self._last_metrics: dict[str, Any] = {}

    def identity(self) -> EngineIdentity:
        return EngineIdentity(engine=self.name, version="0.1-mock", build="foundation-mock")

    def discover_capability(self) -> EngineCapability:
        runtime_options = {
            path: {"status": "supported", "mode": "simulated", "reason": "contract test double"}
            for path in RUNTIME_OPTION_PATHS
        }
        runtime_options["acceleration.backend"] = {
            "status": "supported",
            "mode": "simulated",
            "supported_values": ["auto", "cpu"],
        }
        for path in (
            "context.sliding_window",
            "kv_cache.cache_limit_bytes",
            "prompt_cache.enabled",
            "prompt_cache.max_entries",
            "prompt_cache.max_size_tokens",
            "acceleration.device",
            "acceleration.threads",
        ):
            runtime_options[path] = {"status": "unsupported", "reason": "Mock does not simulate this behavior"}
        runtime_options["engine_options"] = {"status": "supported", "options": self._engine_options}
        generation_options = {
            name: {"status": "supported", "mode": "simulated"}
            for name in ("max_tokens", "temperature", "top_p", "thinking_enabled")
        }
        return EngineCapability(
            identity=self.identity(),
            available=True,
            streaming=True,
            cancellation=True,
            load_unload=True,
            chat_template="simulated",
            thinking_flag="pass-through (simulated)",
            artifact_formats=["bin", "safetensors", "mlx", "gguf"],
            runtime_options=runtime_options,
            generation_options=generation_options,
            reason="deterministic development adapter",
        )

    def load(self, artifact: ModelArtifactBinding) -> dict[str, Any]:
        path = Path(artifact.local_path)
        if not path.exists():
            raise ArtifactNotFoundError(
                "model artifact path does not exist",
                details={"artifact_id": artifact.artifact_id, "local_path": artifact.local_path},
            )
        with self._lock:
            self._loaded = artifact
            self._last_metrics = {}
        return {"loaded": True, "artifact_id": artifact.artifact_id, "path_kind": "directory" if path.is_dir() else "file"}

    def unload(self, artifact_id: str) -> dict[str, Any]:
        with self._lock:
            loaded = self._loaded
            if loaded is None or loaded.artifact_id != artifact_id:
                return {"unloaded": False, "noop": True, "artifact_id": artifact_id}
            self._loaded = None
            self._last_metrics = {}
        return {"unloaded": True, "cleanup_status": "clean", "artifact_id": artifact_id}

    def health(self) -> dict[str, Any]:
        with self._lock:
            loaded = self._loaded
            active = list(self._active)
        return {
            "status": "loaded" if loaded else "ready",
            "engine": self.name,
            "loaded_artifact_id": loaded.artifact_id if loaded else None,
            "active_request_ids": active,
        }

    def resolve_runtime_options(self, options: RuntimeOptions) -> RuntimeSettingsResolution:
        requested = RuntimeOptions.from_payload(options.to_dict())
        if requested.acceleration.backend not in {"auto", "cpu"}:
            raise UnsupportedRuntimeOptionError(
                "Mock supports only auto or cpu acceleration",
                details={"path": "acceleration.backend", "value": requested.acceleration.backend},
            )
        unsupported_values = {
            "context.sliding_window": requested.context.sliding_window,
            "kv_cache.cache_limit_bytes": requested.kv_cache.cache_limit_bytes,
            "prompt_cache.enabled": requested.prompt_cache.enabled,
            "prompt_cache.max_entries": requested.prompt_cache.max_entries,
            "prompt_cache.max_size_tokens": requested.prompt_cache.max_size_tokens,
            "acceleration.device": requested.acceleration.device,
            "acceleration.threads": requested.acceleration.threads,
        }
        for path, value in unsupported_values.items():
            if value not in {None, False}:
                raise UnsupportedRuntimeOptionError(
                    f"Mock does not simulate {path}",
                    details={"path": path, "value": value},
                )
        if requested.prefill.batch_size not in {None, 1}:
            raise UnsupportedRuntimeOptionError(
                "Mock simulates only prefill.batch_size=1",
                details={"path": "prefill.batch_size", "value": requested.prefill.batch_size},
            )
        unknown = sorted(set(requested.engine_options) - set(self._engine_options))
        if unknown:
            raise UnsupportedRuntimeOptionError(
                "engine-specific option is not supported by Mock",
                details={"path": "engine_options", "keys": unknown},
            )
        for key, value in requested.engine_options.items():
            spec = self._engine_options[key]
            valid = isinstance(value, str) if spec["type"] == "string" else isinstance(value, int) and not isinstance(value, bool)
            if not valid:
                raise InvalidRuntimeOptionError(
                    "engine-specific option has an invalid type",
                    details={"path": f"engine_options.{key}", "expected": spec["type"]},
                )
            if isinstance(value, int) and not spec["minimum"] <= value <= spec["maximum"]:
                raise InvalidRuntimeOptionError(
                    "engine-specific option is outside its supported range",
                    details={"path": f"engine_options.{key}", "minimum": spec["minimum"], "maximum": spec["maximum"]},
                )
        kv = requested.kv_cache
        quant_fields = (kv.bits, kv.group_size, kv.quantization_start)
        if kv.mode == "auto" and any(value is not None for value in quant_fields):
            raise InvalidRuntimeOptionError(
                "KV quantization fields require an explicit kv_cache.mode",
                details={"path": "kv_cache.mode", "mode": kv.mode},
            )
        if kv.mode == "full_precision" and any(value is not None for value in quant_fields):
            raise InvalidRuntimeOptionError(
                "KV quantization fields require kv_cache.mode=quantized",
                details={"path": "kv_cache", "mode": kv.mode},
            )

        effective = RuntimeOptions.from_payload(requested.to_dict())
        warnings: list[str] = []
        if effective.context.context_length is None:
            effective = replace(effective, context=replace(effective.context, context_length=4096))
            warnings.append("Mock defaulted context.context_length to 4096")
        if effective.kv_cache.mode == "auto":
            effective = replace(
                effective,
                kv_cache=replace(effective.kv_cache, mode="full_precision", precision=effective.kv_cache.precision or "fp16"),
            )
            warnings.append("Mock resolved kv_cache.mode auto to full_precision")
        elif effective.kv_cache.mode == "quantized":
            effective = replace(
                effective,
                kv_cache=replace(
                    effective.kv_cache,
                    precision=effective.kv_cache.precision or "int8",
                    bits=effective.kv_cache.bits or 8,
                    group_size=effective.kv_cache.group_size or 64,
                    quantization_start=effective.kv_cache.quantization_start or 0,
                ),
            )
        elif effective.kv_cache.precision is None:
            effective = replace(effective, kv_cache=replace(effective.kv_cache, precision="fp16"))
        if effective.prefill.chunk_size is None or effective.prefill.batch_size is None:
            effective = replace(
                effective,
                prefill=replace(
                    effective.prefill,
                    chunk_size=effective.prefill.chunk_size or 128,
                    batch_size=effective.prefill.batch_size or 1,
                ),
            )
            warnings.append("Mock defaulted prefill settings")
        if effective.acceleration.backend == "auto":
            effective = replace(effective, acceleration=replace(effective.acceleration, backend="cpu"))
            warnings.append("Mock resolved acceleration.backend auto to cpu")
        capability = self.discover_capability()
        return RuntimeSettingsResolution(
            requested=requested,
            effective=effective,
            option_status={
                path: str(capability.runtime_options.get(path, {}).get("status", "unsupported"))
                for path in RUNTIME_OPTION_PATHS
            },
            warnings=warnings,
        )

    def _ensure_loaded(self, request: GenerationRequest) -> None:
        with self._lock:
            loaded = self._loaded
        if loaded is None or loaded.artifact_id != request.model_artifact_id:
            raise EngineRuntimeError(
                "requested artifact is not loaded in Mock",
                details={"artifact_id": request.model_artifact_id},
            )

    @staticmethod
    def _prompt_text(request: GenerationRequest) -> str:
        for message in reversed(request.messages):
            if message["role"] == "user":
                return message["content"]
        return request.messages[-1]["content"]

    def _render(self, request: GenerationRequest) -> str:
        prefix = request.runtime_options.engine_options.get("mock.output_prefix", "Mock response")
        prompt = self._prompt_text(request).strip()
        words = f"{prefix}: {prompt}".split()
        return " ".join(words[: request.max_tokens])

    @staticmethod
    def _prompt_tokens(request: GenerationRequest) -> int:
        return max(1, sum(len(message["content"].split()) for message in request.messages))

    @staticmethod
    def _enforce_context_budget(request: GenerationRequest, prompt_tokens: int) -> None:
        context_length = request.runtime_options.context.context_length
        if context_length is None:
            return
        requested_total = prompt_tokens + request.max_tokens
        if requested_total > context_length:
            raise ContextLengthExceededError(
                "prompt tokens plus generation budget exceed context.context_length",
                details={
                    "context_length": context_length,
                    "prompt_tokens": prompt_tokens,
                    "max_tokens": request.max_tokens,
                    "requested_total_tokens": requested_total,
                    "enforcement": "preflight",
                },
            )

    @staticmethod
    def _from_result_payload(payload: dict[str, Any]) -> GenerationResult:
        metrics_payload = dict(payload["metrics"])
        metrics_payload.pop("contract_version", None)
        metrics = RuntimeMetrics(**{key: value for key, value in metrics_payload.items() if key in RuntimeMetrics.__dataclass_fields__})
        usage_payload = dict(payload["usage"])
        usage_payload.pop("contract_version", None)
        usage = TokenUsage(**{key: value for key, value in usage_payload.items() if key in TokenUsage.__dataclass_fields__})
        return GenerationResult(
            request_id=payload["request_id"],
            model_artifact_id=payload["model_artifact_id"],
            engine=payload["engine"],
            text=payload["text"],
            finish_reason=payload["finish_reason"],
            usage=usage,
            metrics=metrics,
            requested_runtime_settings=payload.get("requested_runtime_settings"),
            effective_runtime_settings=payload.get("effective_runtime_settings"),
        )

    def stream(self, request: GenerationRequest, cancel_event: threading.Event) -> Iterator[StreamEvent]:
        self._ensure_loaded(request)
        with self._lock:
            self._active[request.request_id] = cancel_event
        started_monotonic = time.perf_counter()
        started_at = utc_now()
        prompt_tokens = self._prompt_tokens(request)
        self._enforce_context_budget(request, prompt_tokens)
        timeout_deadline = (
            time.monotonic() + request.timeout_ms / 1000 if request.timeout_ms is not None else None
        )
        metrics = RuntimeMetrics(
            started_at=started_at,
            measurement_kind="mock",
            measurement_provenance="simulated",
            context_length=request.runtime_options.context.context_length,
            prefill_tokens=prompt_tokens,
        )
        text = self._render(request)
        chunks = text.split(" ") if text else []
        emitted: list[str] = []
        try:
            yield StreamEvent(type="started", request_id=request.request_id, sequence=0)
            for index, word in enumerate(chunks, start=1):
                if timeout_deadline is not None and time.monotonic() >= timeout_deadline:
                    metrics.timeout = True
                    metrics.finished_at = utc_now()
                    metrics.finish_reason = "timeout"
                    self._last_metrics = metrics.to_dict()
                    yield StreamEvent(
                        type="error",
                        request_id=request.request_id,
                        sequence=index,
                        done=True,
                        error=RuntimeTimeoutError(
                            "generation exceeded timeout_ms",
                            details={"timeout_ms": request.timeout_ms, "timeout_semantics": "cooperative"},
                        ).as_dict(),
                    )
                    return
                if cancel_event.is_set():
                    metrics.cancellation = True
                    metrics.finished_at = utc_now()
                    metrics.finish_reason = "cancelled"
                    self._last_metrics = metrics.to_dict()
                    yield StreamEvent(
                        type="error",
                        request_id=request.request_id,
                        sequence=index,
                        done=True,
                        error=RequestCancelledError("generation was cancelled").as_dict(),
                    )
                    return
                if not emitted:
                    metrics.cold_ttft_ms = (time.perf_counter() - started_monotonic) * 1000
                delta = word if index == 1 else f" {word}"
                emitted.append(word)
                yield StreamEvent(type="delta", request_id=request.request_id, sequence=index, delta=delta)
                delay_ms = request.runtime_options.engine_options.get("mock.chunk_delay_ms", 1)
                if delay_ms:
                    time.sleep(float(delay_ms) / 1000)
            if timeout_deadline is not None and time.monotonic() >= timeout_deadline:
                metrics.timeout = True
                metrics.finished_at = utc_now()
                metrics.finish_reason = "timeout"
                self._last_metrics = metrics.to_dict()
                yield StreamEvent(
                    type="error",
                    request_id=request.request_id,
                    sequence=len(emitted) + 1,
                    done=True,
                    error=RuntimeTimeoutError(
                        "generation exceeded timeout_ms",
                        details={"timeout_ms": request.timeout_ms, "timeout_semantics": "cooperative"},
                    ).as_dict(),
                )
                return
            finished_at = utc_now()
            elapsed_ms = (time.perf_counter() - started_monotonic) * 1000
            metrics.finished_at = finished_at
            metrics.prefill_duration_ms = max(0.01, metrics.cold_ttft_ms or elapsed_ms * 0.25)
            metrics.completion_tokens = len(emitted)
            metrics.generation_duration_ms = max(0.01, elapsed_ms - metrics.prefill_duration_ms)
            metrics.prefill_tokens_per_second = metrics.prefill_tokens / (metrics.prefill_duration_ms / 1000)
            metrics.generation_tokens_per_second = len(emitted) / (metrics.generation_duration_ms / 1000)
            metrics.finish_reason = "stop"
            result = GenerationResult(
                request_id=request.request_id,
                model_artifact_id=request.model_artifact_id,
                engine=self.name,
                text=" ".join(emitted),
                finish_reason="stop",
                usage=TokenUsage(
                    prompt_tokens=metrics.prefill_tokens,
                    completion_tokens=len(emitted),
                    total_tokens=metrics.prefill_tokens + len(emitted),
                ),
                metrics=metrics,
            )
            self._last_metrics = metrics.to_dict()
            yield StreamEvent(
                type="completed",
                request_id=request.request_id,
                sequence=len(emitted) + 1,
                done=True,
                result=result.to_dict(),
            )
        finally:
            with self._lock:
                self._active.pop(request.request_id, None)

    def generate(self, request: GenerationRequest, cancel_event: threading.Event) -> GenerationResult:
        completed: dict[str, Any] | None = None
        for event in self.stream(request, cancel_event):
            if event.type == "completed":
                completed = event.result
            elif event.type == "error":
                code = (event.error or {}).get("code")
                if code == "cancelled":
                    raise RequestCancelledError("generation was cancelled")
                if code == "runtime_timeout":
                    timeout_details = dict((event.error or {}).get("details") or {})
                    timeout_details.setdefault("timeout_semantics", "cooperative")
                    raise RuntimeTimeoutError(
                        "generation exceeded timeout_ms",
                        details=timeout_details,
                    )
                raise EngineRuntimeError("Mock generation failed", details=event.error or {})
        if completed is None:
            raise EngineRuntimeError("Mock generation ended without a result")
        return self._from_result_payload(completed)

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            event = self._active.get(request_id)
        if event is None:
            return False
        event.set()
        return True

    def runtime_metrics(self) -> dict[str, Any]:
        with self._lock:
            active = list(self._active)
        return {"engine": self.name, "active_request_ids": active, "last_metrics": dict(self._last_metrics)}
