"""Policy-free runtime lifecycle and execution core."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, ClassVar, cast

from .adapters import EngineAdapter, LlamaCppAdapter, MLXAdapter, MockAdapter
from .contracts import (
    CONTRACT_VERSION,
    ExecutionTrace,
    GenerationRequest,
    GenerationResult,
    HealthResult,
    LifecycleState,
    LoadResult,
    ModelArtifactBinding,
    RuntimeSettingsResolution,
    StreamEvent,
    UnloadResult,
    new_id,
    utc_now,
)
from .contracts_v2 import (
    CONTRACT_V2_VERSION,
    SUPPORTED_CONTRACT_VERSIONS,
    ArtifactBindingV2,
    ExecutionTraceV2,
    GenerationRequestV2,
    GenerationResultV2,
    ThinkingIntent,
    ThinkingResolution,
    build_execution_binding,
)
from .errors import (
    ArtifactNotFoundError,
    EngineNotFoundError,
    EngineRuntimeError,
    EngineUnavailableError,
    LoadConflictError,
    ModelNotLoadedError,
    RuntimeBusyError,
    RuntimeFoundationError,
    RuntimeTimeoutError,
    UnloadConflictError,
    UnsupportedGenerationSettingError,
    error_payload,
)
from .host import HostProfile
from .version import FOUNDATION_VERSION

DEFAULT_CONSUMER_ID = "anonymous"


@dataclass
class _ActiveRequest:
    request: GenerationRequest
    cancel_event: threading.Event
    timeout_event: threading.Event
    adapter: EngineAdapter
    trace: ExecutionTrace
    timeout_timer: threading.Timer | None = None


class RuntimeCore:
    """Own one loaded model process and expose engine-neutral operations."""

    _ARTIFACT_ENGINE_MAP: ClassVar[dict[str, str]] = {
        "mlx": "mlx",
        "safetensors": "mlx",
        "gguf": "llama.cpp",
    }

    def __init__(
        self,
        *,
        host_profile: HostProfile | None = None,
        adapters: dict[str, EngineAdapter] | None = None,
        foundation_version: str = FOUNDATION_VERSION,
        foundation_build_identity: str | None = None,
    ) -> None:
        self.foundation_version = foundation_version
        # Unknown build identities stay null; a package version is not a build
        # identity and must not be copied into this field.
        self.foundation_build_identity = foundation_build_identity
        self.host_profile = host_profile or HostProfile.detect()
        self.adapters: dict[str, EngineAdapter] = adapters or {
            "mock": MockAdapter(),
            "mlx": MLXAdapter(),
            "llama.cpp": LlamaCppAdapter(),
        }
        self._lock = threading.RLock()
        self._state = LifecycleState.UNLOADED
        self._loaded_artifact: ModelArtifactBinding | None = None
        self._loaded_adapter: EngineAdapter | None = None
        self._leases: dict[str, str] = {}
        self._active: dict[str, _ActiveRequest] = {}
        self._traces: dict[str, ExecutionTrace] = {}
        self._last_error: dict[str, Any] | None = None
        self._last_load_duration_ms: float | None = None
        self._last_unload_duration_ms: float | None = None

    @property
    def lifecycle_state(self) -> LifecycleState:
        with self._lock:
            return self._state

    @staticmethod
    def _consumer_id(value: str | None) -> str:
        return value.strip() if isinstance(value, str) and value.strip() else DEFAULT_CONSUMER_ID

    @staticmethod
    def _artifact(value: ModelArtifactBinding | ArtifactBindingV2 | dict[str, Any]) -> ModelArtifactBinding:
        if isinstance(value, ModelArtifactBinding):
            return value
        if isinstance(value, ArtifactBindingV2):
            return value.to_legacy()
        if isinstance(value, dict) and (
            value.get("contract_version") == CONTRACT_V2_VERSION
            or "registry_identity" in value
            or "content_identity" in value
            or "locator" in value
        ):
            return ArtifactBindingV2.from_payload(value).to_legacy()
        return ModelArtifactBinding.from_payload(value)

    def _select_adapter(self, artifact: ModelArtifactBinding, requested: str | None) -> EngineAdapter:
        engine_name = requested.strip() if isinstance(requested, str) and requested.strip() else None
        if engine_name:
            try:
                selected = self.adapters[engine_name]
            except KeyError as exc:
                raise EngineNotFoundError(
                    "requested engine is not registered", details={"engine": engine_name}
                ) from exc
        else:
            engine_name = self._ARTIFACT_ENGINE_MAP.get(artifact.format)
            if engine_name is None:
                raise EngineNotFoundError(
                    "no engine is registered for the artifact format; specify a supported engine explicitly",
                    details={"format": artifact.format, "artifact_id": artifact.artifact_id},
                )
            try:
                selected = self.adapters[engine_name]
            except KeyError as exc:
                raise EngineNotFoundError(
                    "the artifact format maps to an engine that is not registered",
                    details={"format": artifact.format, "engine": engine_name},
                ) from exc

        capability = selected.discover_capability()
        if not capability.available:
            raise EngineUnavailableError(
                "the requested engine is registered but unavailable",
                details={
                    "engine": selected.name,
                    "format": artifact.format,
                    "reason": capability.reason,
                },
            )
        return selected

    def capabilities(self) -> list[dict[str, Any]]:
        return [self.adapters[name].discover_capability().to_dict() for name in sorted(self.adapters)]

    def capability(self, engine: str) -> dict[str, Any]:
        try:
            adapter = self.adapters[engine]
        except KeyError as exc:
            raise EngineNotFoundError("requested engine is not registered", details={"engine": engine}) from exc
        return adapter.discover_capability().to_dict()

    def host(self) -> dict[str, Any]:
        return self.host_profile.to_dict()

    def load(
        self,
        artifact: ModelArtifactBinding | dict[str, Any],
        *,
        adapter: str | None = None,
        consumer_id: str | None = None,
    ) -> dict[str, Any]:
        binding = self._artifact(artifact)
        if not Path(binding.local_path).exists():
            raise ArtifactNotFoundError(
                "model artifact path does not exist",
                details={"artifact_id": binding.artifact_id, "local_path": binding.local_path},
            )
        owner = self._consumer_id(consumer_id)
        try:
            selected = self._select_adapter(binding, adapter)
        except RuntimeFoundationError as exc:
            with self._lock:
                self._state = LifecycleState.ERROR
                self._last_error = exc.as_dict()
            raise
        with self._lock:
            if self._loaded_artifact is not None:
                if (
                    self._loaded_artifact.execution_identity() == binding.execution_identity()
                    and self._loaded_adapter is selected
                ):
                    lease_id = self._leases.setdefault(owner, new_id("lease"))
                    result = LoadResult(
                        artifact=self._loaded_artifact,
                        engine=selected.identity(),
                        lifecycle_state=self._state,
                        lease_id=lease_id,
                        consumer_id=owner,
                        reused=True,
                        raw={
                            "loaded_once": True,
                            "loaded_artifact_identity": self._loaded_artifact.execution_identity(),
                        },
                    )
                    return result.to_dict()
                raise LoadConflictError(
                    "a different artifact is already loaded; release all existing leases before switching",
                    details={
                        "loaded_artifact": self._loaded_artifact.to_dict(),
                        "requested_artifact": binding.to_dict(),
                        "active_request_ids": list(self._active),
                        "lease_owners": sorted(self._leases),
                    },
                )
            if self._state in {LifecycleState.LOADING, LifecycleState.UNLOADING, LifecycleState.GENERATING}:
                raise RuntimeBusyError("runtime lifecycle is busy", details={"lifecycle_state": self._state.value})
            self._state = LifecycleState.LOADING
            started = time.perf_counter()
            try:
                raw = selected.load(binding)
            except RuntimeFoundationError as exc:
                self._state = LifecycleState.ERROR
                self._last_error = exc.as_dict()
                raise
            except Exception as exc:
                wrapped = EngineRuntimeError("engine load failed", details={"engine": selected.name})
                self._state = LifecycleState.ERROR
                self._last_error = wrapped.as_dict()
                raise wrapped from exc
            self._last_load_duration_ms = (time.perf_counter() - started) * 1000
            self._loaded_artifact = binding
            self._loaded_adapter = selected
            self._leases[owner] = new_id("lease")
            self._state = LifecycleState.LOADED
            self._last_error = None
            result = LoadResult(
                artifact=binding,
                engine=selected.identity(),
                lifecycle_state=self._state,
                lease_id=self._leases[owner],
                consumer_id=owner,
                reused=False,
                raw={**raw, "load_duration_ms": self._last_load_duration_ms},
            )
            return result.to_dict()

    def unload(
        self,
        artifact_id: str | None = None,
        *,
        consumer_id: str | None = None,
        lease_id: str | None = None,
    ) -> dict[str, Any]:
        owner = self._consumer_id(consumer_id)
        with self._lock:
            loaded = self._loaded_artifact
            selected = self._loaded_adapter
            if loaded is None or selected is None:
                if self._state == LifecycleState.ERROR:
                    self._state = LifecycleState.UNLOADED
                return UnloadResult(
                    artifact_id=artifact_id,
                    lifecycle_state=self._state,
                    released=False,
                    unloaded=False,
                    noop=True,
                ).to_dict()
            if artifact_id is not None and artifact_id != loaded.artifact_id:
                raise UnloadConflictError(
                    "requested artifact is not the loaded artifact",
                    details={"loaded_artifact_id": loaded.artifact_id, "requested_artifact_id": artifact_id},
                )
            current_lease = self._leases.get(owner)
            if current_lease is None or (lease_id is not None and lease_id != current_lease):
                raise UnloadConflictError(
                    "consumer does not own a lease for the loaded artifact",
                    details={"consumer_id": owner, "lease_id": lease_id, "lease_owners": sorted(self._leases)},
                )
            if self._active:
                raise RuntimeBusyError(
                    "cannot unload a model while a request is active",
                    details={"active_request_ids": list(self._active)},
                )
            if len(self._leases) > 1:
                del self._leases[owner]
                return UnloadResult(
                    artifact_id=loaded.artifact_id,
                    lifecycle_state=LifecycleState.LOADED,
                    released=True,
                    unloaded=False,
                    raw={"remaining_lease_owners": sorted(self._leases)},
                ).to_dict()
            self._state = LifecycleState.UNLOADING
            started = time.perf_counter()
            try:
                raw = selected.unload(loaded.artifact_id)
            except RuntimeFoundationError as exc:
                self._state = LifecycleState.ERROR
                self._last_error = exc.as_dict()
                raise
            except Exception as exc:
                wrapped = EngineRuntimeError("engine unload failed", details={"engine": selected.name})
                self._state = LifecycleState.ERROR
                self._last_error = wrapped.as_dict()
                raise wrapped from exc
            self._last_unload_duration_ms = (time.perf_counter() - started) * 1000
            del self._leases[owner]
            self._loaded_artifact = None
            self._loaded_adapter = None
            self._state = LifecycleState.UNLOADED
            return UnloadResult(
                artifact_id=loaded.artifact_id,
                lifecycle_state=self._state,
                released=True,
                unloaded=True,
                raw={**raw, "unload_duration_ms": self._last_unload_duration_ms},
            ).to_dict()

    def _authorize_request(self, request: GenerationRequest) -> tuple[ModelArtifactBinding, EngineAdapter, str]:
        loaded, selected = self._loaded_artifact, self._loaded_adapter
        if loaded is None or selected is None:
            raise ModelNotLoadedError("no model artifact is loaded")
        if loaded.artifact_id != request.model_artifact_id:
            raise ModelNotLoadedError(
                "requested artifact is not the loaded artifact",
                details={"loaded_artifact_id": loaded.artifact_id, "requested_artifact_id": request.model_artifact_id},
            )
        owner = self._consumer_id(request.consumer_id)
        lease = self._leases.get(owner)
        if lease is None or (request.lease_id is not None and request.lease_id != lease):
            raise UnloadConflictError(
                "consumer does not own a lease for generation",
                details={"consumer_id": owner, "lease_id": request.lease_id, "lease_owners": sorted(self._leases)},
            )
        return loaded, selected, owner

    def _begin(
        self, request: GenerationRequest
    ) -> tuple[ModelArtifactBinding, EngineAdapter, threading.Event, threading.Event, ExecutionTrace]:
        with self._lock:
            loaded, selected, _ = self._authorize_request(request)
            if self._active:
                raise RuntimeBusyError(
                    "Foundation v1 supports one active generation request per service process",
                    details={"active_request_ids": list(self._active)},
                )
            self._state = LifecycleState.GENERATING
            execution_id = new_id("execution")
            trace = ExecutionTrace(
                execution_id=execution_id,
                request_id=request.request_id,
                engine=selected.identity(),
                artifact=loaded,
                requested_runtime_settings=request.runtime_options.to_dict(),
                effective_runtime_settings=None,
                host_observation=self.host_profile.to_dict(),
                started_at=utc_now(),
            )
            trace_v2 = self._make_v2_trace(request, loaded, selected, execution_id, trace.started_at)
            trace.trace_v2 = trace_v2
            cancel_event = threading.Event()
            timeout_event = threading.Event()
            active = _ActiveRequest(request, cancel_event, timeout_event, selected, trace)
            self._active[request.request_id] = active
            if request.timeout_ms is not None:
                timer = threading.Timer(
                    request.timeout_ms / 1000,
                    self._request_timeout,
                    args=(request.request_id,),
                )
                timer.daemon = True
                active.timeout_timer = timer
                timer.start()
            return loaded, selected, cancel_event, timeout_event, trace

    def _request_timeout(self, request_id: str) -> None:
        with self._lock:
            active = self._active.get(request_id)
        if active is None:
            return
        active.timeout_event.set()
        active.cancel_event.set()
        try:
            active.adapter.cancel(request_id)
        except Exception:  # noqa: BLE001, S110 - cancellation must not mask timeout recording
            # The adapter's cooperative cancellation hook must not prevent the
            # Core from recording the timeout.
            pass

    @staticmethod
    def _timeout_error(request: GenerationRequest, trace: ExecutionTrace) -> RuntimeTimeoutError:
        return RuntimeTimeoutError(
            "generation exceeded timeout_ms",
            details={
                "request_id": request.request_id,
                "timeout_ms": request.timeout_ms,
                "execution_id": trace.execution_id,
                "timeout_semantics": "cooperative",
            },
        )

    @staticmethod
    def _raise_if_timed_out(request: GenerationRequest, timeout_event: threading.Event, trace: ExecutionTrace) -> None:
        if timeout_event.is_set():
            raise RuntimeCore._timeout_error(request, trace)

    def _make_v2_trace(
        self,
        request: GenerationRequest,
        artifact: ModelArtifactBinding,
        selected: EngineAdapter,
        execution_id: str,
        started_at: str,
    ) -> ExecutionTraceV2:
        is_v2 = isinstance(request, GenerationRequestV2)
        v2_request = cast(GenerationRequestV2, request) if is_v2 else None
        thinking = (
            v2_request.thinking_intent
            if v2_request is not None
            else ThinkingIntent.from_legacy(request.thinking_enabled)
        )
        studio_resolved = v2_request.studio_resolved_thinking if v2_request is not None else None
        resolution = ThinkingResolution(
            requested=thinking,
            studio_resolved=studio_resolved,
            foundation_effective=None,
            status="requested" if is_v2 else "legacy_compatibility",
            reason=None if is_v2 else "v1 thinking_enabled mapped for v2 evidence",
        )
        binding = build_execution_binding(
            artifact=artifact,
            engine=selected.identity(),
            adapter_id=selected.name,
            foundation_version=self.foundation_version,
            foundation_build_identity=self.foundation_build_identity,
            effective_runtime_options=None,
        )
        return ExecutionTraceV2(
            execution_id=execution_id,
            request_id=request.request_id,
            request_contract_version=CONTRACT_V2_VERSION if is_v2 else CONTRACT_VERSION,
            execution_binding=binding,
            thinking_resolution=resolution,
            started_at=started_at,
            guard=v2_request.execution_guard if v2_request is not None else None,
        )

    @staticmethod
    def _prepare_effective_request(
        request: GenerationRequest,
        selected: EngineAdapter,
        resolution: RuntimeSettingsResolution,
        trace_v2: ExecutionTraceV2,
    ) -> GenerationRequest:
        """Map v2 thinking to the adapter without silently downgrading it."""

        if not isinstance(request, GenerationRequestV2):
            intent = ThinkingIntent.from_legacy(request.thinking_enabled)
            trace_v2.thinking_resolution = ThinkingResolution(
                requested=intent,
                studio_resolved=None,
                foundation_effective=intent,
                status="legacy_compatibility",
                reason="v1 thinking_enabled was preserved and mapped to v2 evidence",
            )
            return replace(request, runtime_options=resolution.effective)

        resolved = request.studio_resolved_thinking or request.thinking_intent
        generation_options = selected.discover_capability().generation_options
        capability = generation_options.get("thinking_enabled", {})
        supports_thinking = capability.get("status") == "supported"
        effort_capability = generation_options.get("thinking_effort", {})
        budget_capability = generation_options.get("thinking_budget_tokens", {})
        effort_supported = effort_capability.get("status") == "supported"
        budget_supported = budget_capability.get("status") == "supported"
        if (
            (resolved.effort is not None and not effort_supported)
            or (resolved.budget_tokens is not None and not budget_supported)
            or not supports_thinking
        ):
            error = UnsupportedGenerationSettingError(
                "the selected adapter cannot represent the requested thinking intent",
                details={
                    "field": "thinking_intent",
                    "requested": request.thinking_intent.to_dict(),
                    "studio_resolved": request.studio_resolved_thinking.to_dict()
                    if request.studio_resolved_thinking
                    else None,
                    "adapter": selected.name,
                    "resolution": "explicit_failure_required",
                    "capability": {
                        "thinking_enabled": capability,
                        "thinking_effort": effort_capability,
                        "thinking_budget_tokens": budget_capability,
                    },
                },
            )
            trace_v2.thinking_resolution = ThinkingResolution(
                requested=request.thinking_intent,
                studio_resolved=request.studio_resolved_thinking,
                foundation_effective=None,
                status="unsupported",
                reason="adapter capability does not expose the requested effort/budget mechanism",
                error=error.as_dict(),
            )
            raise error
        trace_v2.thinking_resolution = ThinkingResolution(
            requested=request.thinking_intent,
            studio_resolved=request.studio_resolved_thinking,
            foundation_effective=resolved,
            status="resolved",
            reason="Foundation mapped the Studio-resolved intent to the adapter boundary",
        )
        return cast(
            GenerationRequest,
            replace(
                cast(Any, request),
                runtime_options=resolution.effective,
                thinking_enabled=resolved.to_legacy_enabled(),
            ),
        )

    def _end(self, request_id: str) -> None:
        with self._lock:
            active = self._active.pop(request_id, None)
            if active is not None and active.timeout_timer is not None:
                active.timeout_timer.cancel()
            if self._state == LifecycleState.GENERATING and self._loaded_artifact is not None:
                self._state = LifecycleState.LOADED

    def _save_trace(self, trace: ExecutionTrace) -> None:
        with self._lock:
            self._traces[trace.execution_id] = trace
            if len(self._traces) > 256:
                oldest = next(iter(self._traces))
                self._traces.pop(oldest, None)

    def _finish_trace(self, trace: ExecutionTrace) -> None:
        trace.finished_at = trace.finished_at or utc_now()
        self._save_trace(trace)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        _, selected, cancel_event, timeout_event, trace = self._begin(request)
        try:
            resolution: RuntimeSettingsResolution = selected.resolve_runtime_options(request.runtime_options)
            self._raise_if_timed_out(request, timeout_event, trace)
            trace.runtime_settings_resolution = resolution
            trace.effective_runtime_settings = resolution.effective.to_dict()
            trace_v2: ExecutionTraceV2 = cast(ExecutionTraceV2, trace.trace_v2)
            trace_v2.with_runtime_settings(resolution.effective)
            effective_request = self._prepare_effective_request(request, selected, resolution, trace_v2)
            result = selected.generate(effective_request, cancel_event)
            self._raise_if_timed_out(request, timeout_event, trace)
            if result.metrics.load_duration_ms is None:
                result.metrics.load_duration_ms = self._last_load_duration_ms
            if result.metrics.unload_duration_ms is None:
                result.metrics.unload_duration_ms = self._last_unload_duration_ms
            result.execution_id = trace.execution_id
            result.requested_runtime_settings = resolution.requested.to_dict()
            result.effective_runtime_settings = resolution.effective.to_dict()
            result.runtime_settings_resolution = resolution
            trace.status = "completed"
            trace.metrics = result.metrics.to_dict()
            trace.finish_reason = result.finish_reason
            trace.finished_at = utc_now()
            trace_v2.status = "completed"
            trace_v2.metrics = result.metrics.to_dict()
            trace_v2.finish_reason = result.finish_reason
            trace_v2.finished_at = trace.finished_at
            result.trace = trace
            result.generation_v2 = GenerationResultV2.from_result(result, trace_v2)
            self._last_error = None
            return result
        except RuntimeFoundationError as exc:
            timeout_cause: RuntimeFoundationError | None = None
            if timeout_event.is_set() and exc.code != "runtime_timeout":
                timeout_cause = exc
                exc = self._timeout_error(request, trace)
            trace.status = "cancelled" if exc.code == "cancelled" else "error"
            exc.details.setdefault("execution_id", trace.execution_id)
            trace.error = error_payload(exc, execution_id=trace.execution_id)
            trace.finished_at = utc_now()
            trace_v2_error = trace.trace_v2
            if trace_v2_error is not None:
                trace_v2_error.status = trace.status
                trace_v2_error.finished_at = trace.finished_at
                trace_v2_error.raw_execution_error = trace.error
            self._last_error = trace.error
            if timeout_cause is not None:
                raise exc from timeout_cause
            raise
        except Exception as exc:
            wrapped = EngineRuntimeError(
                "engine generation failed", details={"engine": selected.name, "execution_id": trace.execution_id}
            )
            trace.status = "error"
            trace.error = wrapped.as_dict()
            trace.finished_at = utc_now()
            trace_v2_error = trace.trace_v2
            if trace_v2_error is not None:
                trace_v2_error.status = trace.status
                trace_v2_error.finished_at = trace.finished_at
                trace_v2_error.raw_execution_error = trace.error
            self._last_error = trace.error
            raise wrapped from exc
        finally:
            self._finish_trace(trace)
            self._end(request.request_id)

    def stream(self, request: GenerationRequest) -> Iterator[StreamEvent]:
        _, selected, cancel_event, timeout_event, trace = self._begin(request)

        def iterator() -> Iterator[StreamEvent]:
            sequence = 0
            completed = False
            terminal_error = False
            try:
                yield StreamEvent(type="started", request_id=request.request_id, sequence=sequence)
                resolution = selected.resolve_runtime_options(request.runtime_options)
                self._raise_if_timed_out(request, timeout_event, trace)
                trace.runtime_settings_resolution = resolution
                trace.effective_runtime_settings = resolution.effective.to_dict()
                trace_v2: ExecutionTraceV2 = cast(ExecutionTraceV2, trace.trace_v2)
                trace_v2.with_runtime_settings(resolution.effective)
                effective_request = self._prepare_effective_request(request, selected, resolution, trace_v2)
                for event in selected.stream(effective_request, cancel_event):
                    if event.type == "started":
                        continue
                    if event.type == "completed" and timeout_event.is_set():
                        raise self._timeout_error(request, trace)
                    sequence += 1
                    if event.type == "completed" and event.result is not None:
                        result_payload = dict(event.result)
                        metrics_payload = dict(result_payload.get("metrics") or {})
                        metrics_payload.setdefault("load_duration_ms", self._last_load_duration_ms)
                        result_payload["metrics"] = metrics_payload
                        result_payload["execution_id"] = trace.execution_id
                        result_payload["requested_runtime_settings"] = resolution.requested.to_dict()
                        result_payload["effective_runtime_settings"] = resolution.effective.to_dict()
                        result_payload["runtime_settings_resolution"] = resolution.to_dict()
                        trace_v2.status = "completed"
                        trace_v2.metrics = metrics_payload
                        trace_v2.finish_reason = result_payload.get("finish_reason")
                        trace_v2.finished_at = utc_now()
                        result_payload["generation_v2"] = {
                            "contract_version": CONTRACT_V2_VERSION,
                            "schema_version": "runtime-foundation.generation-result.v2",
                            "execution_binding": trace_v2.execution_binding.to_dict(),
                            "execution_binding_fingerprint": trace_v2.execution_binding_fingerprint,
                            "trace": trace_v2.to_dict(),
                        }
                        trace.status = "completed"
                        trace.metrics = metrics_payload
                        trace.finish_reason = result_payload.get("finish_reason")
                        trace.finished_at = utc_now()
                        result_payload["trace"] = trace.to_dict()
                        event.result = result_payload
                        completed = True
                    elif event.type == "error":
                        terminal_error = True
                        error = dict(event.error or {})
                        if timeout_event.is_set() and error.get("code") != "runtime_timeout":
                            error = error_payload(self._timeout_error(request, trace))
                        error.setdefault("details", {})["execution_id"] = trace.execution_id
                        event.error = error
                        trace.status = "cancelled" if error.get("code") == "cancelled" else "error"
                        trace.error = error
                        trace.finished_at = utc_now()
                        trace_v2_error = trace.trace_v2
                        if trace_v2_error is not None:
                            trace_v2_error.status = trace.status
                            trace_v2_error.finished_at = trace.finished_at
                            trace_v2_error.raw_execution_error = error
                        self._last_error = error
                    event.sequence = sequence
                    yield event
                if not terminal_error:
                    self._raise_if_timed_out(request, timeout_event, trace)
            except RuntimeFoundationError as exc:
                if timeout_event.is_set() and exc.code != "runtime_timeout":
                    exc = self._timeout_error(request, trace)
                trace.status = "cancelled" if exc.code == "cancelled" else "error"
                trace.error = error_payload(exc, execution_id=trace.execution_id)
                trace.finished_at = utc_now()
                trace_v2_error = trace.trace_v2
                if trace_v2_error is not None:
                    trace_v2_error.status = trace.status
                    trace_v2_error.finished_at = trace.finished_at
                    trace_v2_error.raw_execution_error = trace.error
                self._last_error = trace.error
                sequence += 1
                yield StreamEvent(
                    type="error",
                    request_id=request.request_id,
                    sequence=sequence,
                    done=True,
                    error=trace.error,
                )
            except Exception:  # noqa: BLE001 - convert arbitrary adapter failures to stable runtime errors
                wrapped = EngineRuntimeError("engine streaming failed", details={"engine": selected.name})
                trace.status = "error"
                trace.error = error_payload(wrapped, execution_id=trace.execution_id)
                trace.finished_at = utc_now()
                trace_v2_error = trace.trace_v2
                if trace_v2_error is not None:
                    trace_v2_error.status = trace.status
                    trace_v2_error.finished_at = trace.finished_at
                    trace_v2_error.raw_execution_error = trace.error
                self._last_error = trace.error
                sequence += 1
                yield StreamEvent(
                    type="error",
                    request_id=request.request_id,
                    sequence=sequence,
                    done=True,
                    error=trace.error,
                )
            finally:
                if not completed and trace.status == "running":
                    trace.status = (
                        "error" if timeout_event.is_set() else "cancelled" if cancel_event.is_set() else "abandoned"
                    )
                    trace.finished_at = utc_now()
                    trace_v2_error = trace.trace_v2
                    if trace_v2_error is not None:
                        trace_v2_error.status = trace.status
                        trace_v2_error.finished_at = trace.finished_at
                        trace_v2_error.raw_execution_error = trace.error
                self._finish_trace(trace)
                self._end(request.request_id)

        return iterator()

    def cancel(self, request_id: str) -> dict[str, Any]:
        with self._lock:
            active = self._active.get(request_id)
        if active is None:
            return {
                "contract_version": CONTRACT_VERSION,
                "request_id": request_id,
                "cancelled": False,
                "state": "not_found",
            }
        active.cancel_event.set()
        adapter_cancelled = active.adapter.cancel(request_id)
        return {
            "contract_version": CONTRACT_VERSION,
            "request_id": request_id,
            "cancelled": True,
            "adapter_acknowledged": adapter_cancelled,
            "state": "cancellation_requested",
        }

    def health(self) -> dict[str, Any]:
        with self._lock:
            loaded = self._loaded_artifact
            selected = self._loaded_adapter
            state = self._state
            active = list(self._active)
            last_error = self._last_error
        if state == LifecycleState.ERROR:
            status = "error"
        elif state == LifecycleState.GENERATING:
            status = "busy"
        elif loaded is not None:
            status = "loaded"
        else:
            status = "ready"
        return HealthResult(
            status=status,
            lifecycle_state=state,
            foundation_version=self.foundation_version,
            contract_version=CONTRACT_VERSION,
            loaded_artifact_id=loaded.artifact_id if loaded else None,
            loaded_engine=selected.identity() if selected else None,
            active_request_ids=active,
            host=self.host_profile.to_dict(),
            last_error=last_error,
            supported_contract_versions=list(SUPPORTED_CONTRACT_VERSIONS),
        ).to_dict()

    def runtime_metrics(self) -> dict[str, Any]:
        with self._lock:
            loaded = self._loaded_artifact
            selected = self._loaded_adapter
            active = list(self._active)
            state = self._state
        return {
            "contract_version": CONTRACT_VERSION,
            "foundation_version": self.foundation_version,
            "supported_contract_versions": list(SUPPORTED_CONTRACT_VERSIONS),
            "captured_at": utc_now(),
            "lifecycle_state": state.value,
            "loaded_artifact_id": loaded.artifact_id if loaded else None,
            "loaded_engine": selected.identity().to_dict() if selected else None,
            "active_request_ids": active,
            "load_duration_ms": self._last_load_duration_ms,
            "unload_duration_ms": self._last_unload_duration_ms,
            "host_observation": self.host_profile.to_dict(),
            "adapters": [self.adapters[name].runtime_metrics() for name in sorted(self.adapters)],
        }

    def get_execution(self, execution_id: str) -> dict[str, Any] | None:
        with self._lock:
            trace = self._traces.get(execution_id)
        return trace.to_dict() if trace else None


# Migration-friendly spelling for consumers that used the PR #10 name.
InferenceManager = RuntimeCore
