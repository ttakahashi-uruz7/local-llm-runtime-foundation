"""Versioned, engine-neutral contracts for the Local LLM Runtime Foundation.

The contracts intentionally describe observations and execution requests.  They
do not contain benchmark scores, deployment decisions, model registry status,
or Studio-specific policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from .errors import RuntimeOptionsAliasConflictError, RuntimeOptionsSchemaMismatchError


CONTRACT_VERSION = "runtime-foundation.contract.v1"
RUNTIME_OPTIONS_VERSION = "runtime-foundation.runtime-options.v1"
GENERATION_REQUEST_VERSION = "runtime-foundation.generation-request.v1"
GENERATION_RESULT_VERSION = "runtime-foundation.generation-result.v1"
STREAM_EVENT_VERSION = "runtime-foundation.stream-event.v1"

RUNTIME_OPTION_PATHS = (
    "context.context_length",
    "context.sliding_window",
    "kv_cache.mode",
    "kv_cache.precision",
    "kv_cache.bits",
    "kv_cache.group_size",
    "kv_cache.quantization_start",
    "kv_cache.max_size_tokens",
    "kv_cache.cache_limit_bytes",
    "prefill.chunk_size",
    "prefill.batch_size",
    "prompt_cache.enabled",
    "prompt_cache.max_entries",
    "prompt_cache.max_size_tokens",
    "acceleration.backend",
    "acceleration.device",
    "acceleration.threads",
    "engine_options",
)


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp suitable for trace payloads."""

    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return dict(value)


def _keys(value: dict[str, Any], allowed: set[str], field_name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unsupported fields in {field_name}: {', '.join(unknown)}")


def _positive_int(value: Any, field_name: str, *, maximum: int | None = None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}")
    return value


def _non_negative_int(value: Any, field_name: str, *, maximum: int | None = None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}")
    return value


def _optional_number(value: Any, field_name: str, *, minimum: float | None = None, maximum: float | None = None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}")
    return number


class LifecycleState(str, Enum):
    UNLOADED = "UNLOADED"
    LOADING = "LOADING"
    LOADED = "LOADED"
    GENERATING = "GENERATING"
    UNLOADING = "UNLOADING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class ContextSettings:
    context_length: int | None = None
    sliding_window: int | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> "ContextSettings":
        value = _mapping(payload, "context")
        _keys(value, {"context_length", "sliding_window"}, "context")
        return cls(
            context_length=_positive_int(value.get("context_length"), "context.context_length", maximum=1_048_576),
            sliding_window=_positive_int(value.get("sliding_window"), "context.sliding_window", maximum=1_048_576),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"context_length": self.context_length, "sliding_window": self.sliding_window}


@dataclass(frozen=True)
class KVCacheSettings:
    mode: str = "auto"
    precision: str | None = None
    bits: int | None = None
    group_size: int | None = None
    quantization_start: int | None = None
    max_size_tokens: int | None = None
    cache_limit_bytes: int | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> "KVCacheSettings":
        value = _mapping(payload, "kv_cache")
        _keys(
            value,
            {
                "mode",
                "precision",
                "bits",
                "group_size",
                "quantization_start",
                "max_size_tokens",
                "cache_limit_bytes",
            },
            "kv_cache",
        )
        mode = value.get("mode", "auto")
        if mode not in {"auto", "full_precision", "quantized"}:
            raise ValueError("kv_cache.mode must be auto, full_precision, or quantized")
        precision = value.get("precision")
        if precision is not None and (not isinstance(precision, str) or not precision.strip()):
            raise ValueError("kv_cache.precision must be a non-empty string")
        return cls(
            mode=mode,
            precision=precision.strip() if isinstance(precision, str) else None,
            bits=_positive_int(value.get("bits"), "kv_cache.bits", maximum=64),
            group_size=_positive_int(value.get("group_size"), "kv_cache.group_size", maximum=65_536),
            quantization_start=_non_negative_int(
                value.get("quantization_start"), "kv_cache.quantization_start", maximum=1_048_576
            ),
            max_size_tokens=_positive_int(value.get("max_size_tokens"), "kv_cache.max_size_tokens", maximum=1_048_576),
            cache_limit_bytes=_positive_int(value.get("cache_limit_bytes"), "kv_cache.cache_limit_bytes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "precision": self.precision,
            "bits": self.bits,
            "group_size": self.group_size,
            "quantization_start": self.quantization_start,
            "max_size_tokens": self.max_size_tokens,
            "cache_limit_bytes": self.cache_limit_bytes,
        }


@dataclass(frozen=True)
class PrefillSettings:
    chunk_size: int | None = None
    batch_size: int | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> "PrefillSettings":
        value = _mapping(payload, "prefill")
        _keys(value, {"chunk_size", "batch_size"}, "prefill")
        return cls(
            chunk_size=_positive_int(value.get("chunk_size"), "prefill.chunk_size", maximum=1_048_576),
            batch_size=_positive_int(value.get("batch_size"), "prefill.batch_size", maximum=4096),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_size": self.chunk_size, "batch_size": self.batch_size}


@dataclass(frozen=True)
class PromptCacheSettings:
    enabled: bool = False
    max_entries: int | None = None
    max_size_tokens: int | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> "PromptCacheSettings":
        value = _mapping(payload, "prompt_cache")
        _keys(value, {"enabled", "max_entries", "max_size_tokens"}, "prompt_cache")
        enabled = value.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("prompt_cache.enabled must be a boolean")
        return cls(
            enabled=enabled,
            max_entries=_positive_int(value.get("max_entries"), "prompt_cache.max_entries", maximum=1_000_000),
            max_size_tokens=_positive_int(
                value.get("max_size_tokens"), "prompt_cache.max_size_tokens", maximum=1_048_576
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "max_entries": self.max_entries,
            "max_size_tokens": self.max_size_tokens,
        }


@dataclass(frozen=True)
class AccelerationSettings:
    backend: str = "auto"
    device: str | None = None
    threads: int | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> "AccelerationSettings":
        value = _mapping(payload, "acceleration")
        _keys(value, {"backend", "device", "threads"}, "acceleration")
        backend = value.get("backend", "auto")
        if not isinstance(backend, str) or not backend.strip():
            raise ValueError("acceleration.backend must be a non-empty string")
        device = value.get("device")
        if device is not None and (not isinstance(device, str) or not device.strip()):
            raise ValueError("acceleration.device must be a non-empty string when provided")
        return cls(
            backend=backend.strip().lower(),
            device=device.strip() if isinstance(device, str) else None,
            threads=_positive_int(value.get("threads"), "acceleration.threads", maximum=4096),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"backend": self.backend, "device": self.device, "threads": self.threads}


@dataclass(frozen=True)
class RuntimeOptions:
    """Requested or effective engine-neutral runtime settings."""

    schema_version: str = RUNTIME_OPTIONS_VERSION
    context: ContextSettings = field(default_factory=ContextSettings)
    kv_cache: KVCacheSettings = field(default_factory=KVCacheSettings)
    prefill: PrefillSettings = field(default_factory=PrefillSettings)
    prompt_cache: PromptCacheSettings = field(default_factory=PromptCacheSettings)
    acceleration: AccelerationSettings = field(default_factory=AccelerationSettings)
    engine_options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> "RuntimeOptions":
        value = _mapping(payload, "runtime_options")
        _keys(
            value,
            {
                "contract_version",
                "schema_version",
                "context",
                "kv_cache",
                "kv",
                "prefill",
                "prefill_chunk",
                "prompt_cache",
                "acceleration",
                "engine_options",
            },
            "runtime_options",
        )
        contract_version = value.get("contract_version")
        if contract_version is not None and contract_version != CONTRACT_VERSION:
            raise RuntimeOptionsSchemaMismatchError(
                "runtime_options contract_version is not supported",
                details={"requested": contract_version, "supported": CONTRACT_VERSION},
            )
        schema_version = value.get("schema_version")
        if schema_version is not None and schema_version != RUNTIME_OPTIONS_VERSION:
            raise RuntimeOptionsSchemaMismatchError(
                "runtime_options schema_version is not supported",
                details={"requested": schema_version, "supported": RUNTIME_OPTIONS_VERSION},
            )
        if "kv" in value and "kv_cache" in value:
            raise RuntimeOptionsAliasConflictError(
                "kv and kv_cache cannot both be specified", details={"fields": ["kv", "kv_cache"]}
            )
        if "prefill" in value and "prefill_chunk" in value:
            raise RuntimeOptionsAliasConflictError(
                "prefill and prefill_chunk cannot both be specified", details={"fields": ["prefill", "prefill_chunk"]}
            )
        engine_options = value.get("engine_options", {})
        if not isinstance(engine_options, dict):
            raise ValueError("runtime_options.engine_options must be an object")
        return cls(
            context=ContextSettings.from_payload(value.get("context")),
            kv_cache=KVCacheSettings.from_payload(value.get("kv_cache", value.get("kv"))),
            prefill=PrefillSettings.from_payload(value.get("prefill", value.get("prefill_chunk"))),
            prompt_cache=PromptCacheSettings.from_payload(value.get("prompt_cache")),
            acceleration=AccelerationSettings.from_payload(value.get("acceleration")),
            engine_options=dict(engine_options),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "schema_version": self.schema_version,
            "context": self.context.to_dict(),
            "kv_cache": self.kv_cache.to_dict(),
            "prefill": self.prefill.to_dict(),
            "prompt_cache": self.prompt_cache.to_dict(),
            "acceleration": self.acceleration.to_dict(),
            "engine_options": dict(self.engine_options),
        }


@dataclass(frozen=True)
class RuntimeSettingsResolution:
    requested: RuntimeOptions
    effective: RuntimeOptions
    option_status: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "requested_runtime_settings": self.requested.to_dict(),
            "effective_runtime_settings": self.effective.to_dict(),
            "option_status": dict(self.option_status),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class ModelArtifactBinding:
    """A consumer-supplied artifact binding, not a model registry record."""

    artifact_id: str
    local_path: str
    format: str
    quantization: str | None = None
    artifact_hash: str | None = None
    revision: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> "ModelArtifactBinding":
        value = _mapping(payload, "artifact")
        _keys(
            value,
            {"contract_version", "artifact_id", "local_path", "format", "quantization", "artifact_hash", "revision", "metadata"},
            "artifact",
        )
        if value.get("contract_version") not in {None, CONTRACT_VERSION}:
            raise ValueError("artifact contract_version is not supported")
        for name in ("artifact_id", "local_path", "format"):
            if not isinstance(value.get(name), str) or not value[name].strip():
                raise ValueError(f"artifact.{name} is required")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("artifact.metadata must be an object")
        return cls(
            artifact_id=value["artifact_id"].strip(),
            local_path=value["local_path"].strip(),
            format=value["format"].strip().lower(),
            quantization=value.get("quantization"),
            artifact_hash=value.get("artifact_hash"),
            revision=value.get("revision"),
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "artifact_id": self.artifact_id,
            "local_path": self.local_path,
            "format": self.format,
            "quantization": self.quantization,
            "artifact_hash": self.artifact_hash,
            "revision": self.revision,
            "metadata": dict(self.metadata),
        }

    def execution_identity(self) -> dict[str, Any]:
        """Return the fields that identify the artifact actually loaded for execution.

        Metadata is descriptive consumer data and is intentionally not part of
        the execution identity. The path, format, quantization, hash, and
        revision are the lineage-bearing fields used for load reuse decisions.
        """

        return {
            "artifact_id": self.artifact_id,
            "local_path": self.local_path,
            "format": self.format,
            "quantization": self.quantization,
            "artifact_hash": self.artifact_hash,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class EngineIdentity:
    engine: str
    version: str | None = None
    build: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "engine": self.engine,
            "version": self.version,
            "build": self.build,
        }


@dataclass(frozen=True)
class EngineCapability:
    identity: EngineIdentity
    available: bool
    streaming: bool
    cancellation: bool
    load_unload: bool
    chat_template: str = "unknown"
    thinking_flag: str = "unknown"
    artifact_formats: list[str] = field(default_factory=list)
    runtime_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    generation_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "engine": self.identity.engine,
            "identity": self.identity.to_dict(),
            "available": self.available,
            "streaming": self.streaming,
            "cancellation": self.cancellation,
            "load_unload": self.load_unload,
            "chat_template": self.chat_template,
            "thinking_flag": self.thinking_flag,
            "artifact_formats": list(self.artifact_formats),
            "runtime_options": {key: dict(value) for key, value in self.runtime_options.items()},
            "generation_options": {key: dict(value) for key, value in self.generation_options.items()},
            "reason": self.reason,
        }


@dataclass(frozen=True)
class GenerationRequest:
    model_artifact_id: str
    messages: list[dict[str, str]]
    request_id: str = field(default_factory=lambda: new_id("req"))
    consumer_id: str | None = None
    lease_id: str | None = None
    max_tokens: int = 64
    temperature: float = 0.0
    top_p: float | None = None
    thinking_enabled: bool | None = None
    timeout_ms: int | None = None
    runtime_options: RuntimeOptions = field(default_factory=RuntimeOptions)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Any) -> "GenerationRequest":
        value = _mapping(payload, "generation_request")
        _keys(
            value,
            {
                "contract_version",
                "schema_version",
                "model_artifact_id",
                "artifact_id",
                "messages",
                "request_id",
                "consumer_id",
                "lease_id",
                "max_tokens",
                "temperature",
                "top_p",
                "thinking_enabled",
                "timeout_ms",
                "runtime_options",
                "runtime_settings",
                "metadata",
            },
            "generation_request",
        )
        version = value.get("contract_version")
        if version not in {None, CONTRACT_VERSION}:
            raise ValueError("generation_request contract_version is not supported")
        schema_version = value.get("schema_version")
        if schema_version not in {None, GENERATION_REQUEST_VERSION}:
            raise ValueError("generation_request schema_version is not supported")
        artifact_id = value.get("model_artifact_id") or value.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ValueError("model_artifact_id is required")
        messages = value.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        normalized: list[dict[str, str]] = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("each message must be an object")
            role, content = message.get("role"), message.get("content")
            if not isinstance(role, str) or not isinstance(content, str):
                raise ValueError("each message requires string role and content")
            normalized.append({"role": role, "content": content})
        max_tokens = value.get("max_tokens", 64)
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= 131_072:
            raise ValueError("max_tokens must be an integer between 1 and 131072")
        temperature = _optional_number(value.get("temperature", 0.0), "temperature", minimum=0, maximum=2)
        top_p = _optional_number(value.get("top_p"), "top_p", minimum=0, maximum=1)
        thinking_enabled = value.get("thinking_enabled")
        if thinking_enabled is not None and not isinstance(thinking_enabled, bool):
            raise ValueError("thinking_enabled must be a boolean when provided")
        timeout_ms = _positive_int(value.get("timeout_ms"), "timeout_ms", maximum=86_400_000)
        consumer_id = value.get("consumer_id")
        lease_id = value.get("lease_id")
        for name, item in (("consumer_id", consumer_id), ("lease_id", lease_id), ("request_id", value.get("request_id"))):
            if item is not None and (not isinstance(item, str) or not item.strip()):
                raise ValueError(f"{name} must be a non-empty string when provided")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        if "runtime_options" in value and "runtime_settings" in value:
            raise RuntimeOptionsAliasConflictError(
                "runtime_options and runtime_settings cannot both be specified",
                details={"fields": ["runtime_options", "runtime_settings"]},
            )
        options_payload = value.get("runtime_options", value.get("runtime_settings"))
        return cls(
            model_artifact_id=artifact_id.strip(),
            messages=normalized,
            request_id=value.get("request_id") or new_id("req"),
            consumer_id=consumer_id.strip() if isinstance(consumer_id, str) else None,
            lease_id=lease_id.strip() if isinstance(lease_id, str) else None,
            max_tokens=max_tokens,
            temperature=temperature if temperature is not None else 0.0,
            top_p=top_p,
            thinking_enabled=thinking_enabled,
            timeout_ms=timeout_ms,
            runtime_options=RuntimeOptions.from_payload(options_payload),
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "schema_version": GENERATION_REQUEST_VERSION,
            "model_artifact_id": self.model_artifact_id,
            "messages": [dict(message) for message in self.messages],
            "request_id": self.request_id,
            "consumer_id": self.consumer_id,
            "lease_id": self.lease_id,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "thinking_enabled": self.thinking_enabled,
            "timeout_ms": self.timeout_ms,
            "runtime_options": self.runtime_options.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass
class TokenUsage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class RuntimeMetrics:
    """Raw runtime observations; no pass/fail or eligibility fields."""

    started_at: str
    finished_at: str | None = None
    measurement_kind: str = "unknown"
    measurement_provenance: str = "unavailable"
    load_duration_ms: float | None = None
    unload_duration_ms: float | None = None
    cold_ttft_ms: float | None = None
    warm_ttft_ms: float | None = None
    prefill_tokens: int | None = None
    prefill_duration_ms: float | None = None
    prefill_tokens_per_second: float | None = None
    completion_tokens: int | None = None
    generation_duration_ms: float | None = None
    generation_tokens_per_second: float | None = None
    process_memory_bytes: int | None = None
    peak_memory_bytes: int | None = None
    memory_pressure: str | None = None
    swap_before_bytes: int | None = None
    swap_after_bytes: int | None = None
    swap_delta_bytes: int | None = None
    context_length: int | None = None
    engine_failure: str | None = None
    metal_allocation_failure: bool = False
    context_failure: bool = False
    timeout: bool = False
    cancellation: bool = False
    cleanup_status: str | None = None
    finish_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def mark_first_output(self, timestamp: str | None = None) -> None:
        """Record TTFT using a monotonic timer supplied by the adapter."""

        del timestamp  # The adapter records the exact duration in cold_ttft_ms.

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "measurement_kind": self.measurement_kind,
            "measurement_provenance": self.measurement_provenance,
            "load_duration_ms": self.load_duration_ms,
            "unload_duration_ms": self.unload_duration_ms,
            "cold_ttft_ms": self.cold_ttft_ms,
            "warm_ttft_ms": self.warm_ttft_ms,
            "prefill_tokens": self.prefill_tokens,
            "prefill_duration_ms": self.prefill_duration_ms,
            "prefill_tokens_per_second": self.prefill_tokens_per_second,
            "completion_tokens": self.completion_tokens,
            "generation_duration_ms": self.generation_duration_ms,
            "generation_tokens_per_second": self.generation_tokens_per_second,
            "process_memory_bytes": self.process_memory_bytes,
            "peak_memory_bytes": self.peak_memory_bytes,
            "memory_pressure": self.memory_pressure,
            "swap_before_bytes": self.swap_before_bytes,
            "swap_after_bytes": self.swap_after_bytes,
            "swap_delta_bytes": self.swap_delta_bytes,
            "context_length": self.context_length,
            "engine_failure": self.engine_failure,
            "metal_allocation_failure": self.metal_allocation_failure,
            "context_failure": self.context_failure,
            "timeout": self.timeout,
            "cancellation": self.cancellation,
            "cleanup_status": self.cleanup_status,
            "finish_reason": self.finish_reason,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class RuntimeErrorRecord:
    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = field(default_factory=dict)
    execution_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": dict(self.details),
            "execution_id": self.execution_id,
        }


@dataclass
class ExecutionTrace:
    execution_id: str
    request_id: str
    engine: EngineIdentity
    artifact: ModelArtifactBinding
    requested_runtime_settings: dict[str, Any]
    effective_runtime_settings: dict[str, Any] | None
    host_observation: dict[str, Any]
    started_at: str
    runtime_settings_resolution: RuntimeSettingsResolution | None = None
    finished_at: str | None = None
    status: str = "running"
    metrics: dict[str, Any] | None = None
    finish_reason: str | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "execution_id": self.execution_id,
            "request_id": self.request_id,
            "engine": self.engine.to_dict(),
            "artifact": self.artifact.to_dict(),
            "requested_runtime_settings": dict(self.requested_runtime_settings),
            "effective_runtime_settings": self.effective_runtime_settings,
            "runtime_settings_resolution": (
                self.runtime_settings_resolution.to_dict() if self.runtime_settings_resolution else None
            ),
            "host_observation": dict(self.host_observation),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "metrics": self.metrics,
            "finish_reason": self.finish_reason,
            "error": self.error,
        }


@dataclass
class GenerationResult:
    request_id: str
    model_artifact_id: str
    engine: str
    text: str
    finish_reason: str
    usage: TokenUsage
    metrics: RuntimeMetrics
    execution_id: str | None = None
    trace: ExecutionTrace | None = None
    requested_runtime_settings: dict[str, Any] | None = None
    effective_runtime_settings: dict[str, Any] | None = None
    runtime_settings_resolution: RuntimeSettingsResolution | None = None

    @property
    def adapter(self) -> str:
        """Compatibility spelling for consumers migrating from PR #10."""

        return self.engine

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "schema_version": GENERATION_RESULT_VERSION,
            "request_id": self.request_id,
            "model_artifact_id": self.model_artifact_id,
            "engine": self.engine,
            "adapter": self.engine,
            "text": self.text,
            "finish_reason": self.finish_reason,
            "usage": self.usage.to_dict(),
            "metrics": self.metrics.to_dict(),
            "execution_id": self.execution_id,
            "trace": self.trace.to_dict() if self.trace else None,
            "requested_runtime_settings": self.requested_runtime_settings,
            "effective_runtime_settings": self.effective_runtime_settings,
            "runtime_settings_resolution": (
                self.runtime_settings_resolution.to_dict() if self.runtime_settings_resolution else None
            ),
        }


@dataclass
class StreamEvent:
    type: str
    request_id: str
    sequence: int
    delta: str = ""
    done: bool = False
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    timestamp: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "schema_version": STREAM_EVENT_VERSION,
            "type": self.type,
            "request_id": self.request_id,
            "sequence": self.sequence,
            "delta": self.delta,
            "done": self.done,
            "result": self.result,
            "error": self.error,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class LoadResult:
    artifact: ModelArtifactBinding
    engine: EngineIdentity
    lifecycle_state: LifecycleState
    lease_id: str
    consumer_id: str
    reused: bool
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "artifact": self.artifact.to_dict(),
            "engine": self.engine.to_dict(),
            "lifecycle_state": self.lifecycle_state.value,
            "lease_id": self.lease_id,
            "consumer_id": self.consumer_id,
            "reused": self.reused,
            "raw": dict(self.raw),
        }


@dataclass(frozen=True)
class UnloadResult:
    artifact_id: str | None
    lifecycle_state: LifecycleState
    released: bool
    unloaded: bool
    noop: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "artifact_id": self.artifact_id,
            "lifecycle_state": self.lifecycle_state.value,
            "released": self.released,
            "unloaded": self.unloaded,
            "noop": self.noop,
            "raw": dict(self.raw),
        }


@dataclass(frozen=True)
class HealthResult:
    status: str
    lifecycle_state: LifecycleState
    foundation_version: str
    contract_version: str
    loaded_artifact_id: str | None
    loaded_engine: EngineIdentity | None
    active_request_ids: list[str]
    host: dict[str, Any]
    last_error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "foundation_version": self.foundation_version,
            "status": self.status,
            "lifecycle_state": self.lifecycle_state.value,
            "loaded_artifact_id": self.loaded_artifact_id,
            "loaded_engine": self.loaded_engine.to_dict() if self.loaded_engine else None,
            "active_request_ids": list(self.active_request_ids),
            "host": dict(self.host),
            "last_error": self.last_error,
        }
