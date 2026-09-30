"""Additive engine-neutral v3 request and separated runtime options.

The v1/v2 payloads remain authoritative for their own contract versions. v3
uses distinct load, generation, and Foundation execution-budget objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contracts import (
    CONTRACT_VERSION,
    RuntimeOptions,
    _mapping,
    _non_negative_int,
    _optional_number,
    _positive_int,
)
from .contracts_v2 import (
    CONTRACT_V2_VERSION,
    ExecutionInputV1,
    ArtifactBindingV2,
    EngineBindingV2,
    FoundationBindingV2,
    GenerationRequestV2,
    ThinkingIntent,
    ThinkingMode,
    _canonical_copy,
    canonical_fingerprint,
    canonical_json,
)
from .errors import UnsupportedRuntimeOptionError

CONTRACT_V3_VERSION = "runtime-foundation.contract.v3"
SUPPORTED_CONTRACT_VERSIONS_V3 = (CONTRACT_VERSION, CONTRACT_V2_VERSION, CONTRACT_V3_VERSION)
LOAD_OPTIONS_VERSION = "runtime-foundation.load-options.v1"
GENERATION_OPTIONS_VERSION = "runtime-foundation.generation-options.v1"
EXECUTION_CONSTRAINTS_VERSION = "runtime-foundation.execution-constraints.v1"
GENERATION_REQUEST_V3_VERSION = "runtime-foundation.generation-request.v3"
EXECUTION_BINDING_V3_VERSION = "runtime-foundation.execution-binding.v3"
EXECUTION_EVIDENCE_V3_VERSION = "runtime-foundation.execution-evidence.v3"
EXECUTION_TRACE_V3_VERSION = "runtime-foundation.execution-trace.v3"
GENERATION_RESULT_V3_VERSION = "runtime-foundation.generation-result.v3"
STREAM_EVENT_V3_VERSION = "runtime-foundation.stream-event.v3"
LOAD_IDENTITY_V1_VERSION = "runtime-foundation.load-identity.v1"


def _keys(value: dict[str, Any], allowed: set[str], field_name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unsupported fields in {field_name}: {', '.join(unknown)}")


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string when provided")
    return value.strip()


@dataclass(frozen=True)
class KVCacheConfiguration:
    """Requested KV key/value storage types; null leaves engine defaults intact."""

    key_type: str | None = None
    value_type: str | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> KVCacheConfiguration:
        value = _mapping(payload, "load_options.kv_cache")
        _keys(value, {"key_type", "value_type"}, "load_options.kv_cache")
        return cls(
            key_type=_optional_text(value.get("key_type"), "load_options.kv_cache.key_type"),
            value_type=_optional_text(value.get("value_type"), "load_options.kv_cache.value_type"),
        )

    def to_dict(self) -> dict[str, str | None]:
        return {"key_type": self.key_type, "value_type": self.value_type}


@dataclass(frozen=True)
class AccelerationConfiguration:
    """Load-time accelerator selection and explicit offload request."""

    backend: str = "auto"
    device: str | None = None
    gpu_offload_layers: int | str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or self.backend.strip().lower() not in {"auto", "cpu", "metal"}:
            raise ValueError("load_options.acceleration.backend must be auto, cpu, or metal")
        object.__setattr__(self, "backend", self.backend.strip().lower())
        object.__setattr__(self, "device", _optional_text(self.device, "load_options.acceleration.device"))
        if self.gpu_offload_layers == "all":
            return
        object.__setattr__(
            self,
            "gpu_offload_layers",
            _non_negative_int(self.gpu_offload_layers, "load_options.acceleration.gpu_offload_layers", maximum=4096),
        )

    @classmethod
    def from_payload(cls, payload: Any) -> AccelerationConfiguration:
        value = _mapping(payload, "load_options.acceleration")
        _keys(value, {"backend", "device", "gpu_offload_layers"}, "load_options.acceleration")
        return cls(
            backend=value.get("backend", "auto"),
            device=value.get("device"),
            gpu_offload_layers=value.get("gpu_offload_layers"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "device": self.device,
            "gpu_offload_layers": self.gpu_offload_layers,
        }


@dataclass(frozen=True)
class LoadOptions:
    """Settings that affect model/context load state and require a reload."""

    model_context_size: int | None = None
    batch: int | None = None
    ubatch: int | None = None
    threads: int | None = None
    kv_cache: KVCacheConfiguration = field(default_factory=KVCacheConfiguration)
    acceleration: AccelerationConfiguration = field(default_factory=AccelerationConfiguration)
    schema_version: str = LOAD_OPTIONS_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LOAD_OPTIONS_VERSION:
            raise ValueError(f"load_options.schema_version must be {LOAD_OPTIONS_VERSION}")
        for name, maximum in (("model_context_size", 1_048_576), ("batch", 1_048_576), ("ubatch", 1_048_576), ("threads", 4096)):
            object.__setattr__(self, name, _positive_int(getattr(self, name), f"load_options.{name}", maximum=maximum))
        if self.batch is not None and self.ubatch is not None and self.ubatch > self.batch:
            raise ValueError("load_options.ubatch must be <= load_options.batch")
        if not isinstance(self.kv_cache, KVCacheConfiguration):
            raise ValueError("load_options.kv_cache must be a validated KVCacheConfiguration")
        if not isinstance(self.acceleration, AccelerationConfiguration):
            raise ValueError("load_options.acceleration must be a validated AccelerationConfiguration")

    @classmethod
    def from_payload(cls, payload: Any) -> LoadOptions:
        value = _mapping(payload, "load_options")
        _keys(
            value,
            {
                "contract_version",
                "schema_version",
                "model_context_size",
                "batch",
                "ubatch",
                "threads",
                "kv_cache",
                "acceleration",
            },
            "load_options",
        )
        if value.get("contract_version", CONTRACT_V3_VERSION) != CONTRACT_V3_VERSION:
            raise ValueError("load_options contract_version must be runtime-foundation.contract.v3")
        return cls(
            model_context_size=value.get("model_context_size"),
            batch=value.get("batch"),
            ubatch=value.get("ubatch"),
            threads=value.get("threads"),
            kv_cache=KVCacheConfiguration.from_payload(value.get("kv_cache")),
            acceleration=AccelerationConfiguration.from_payload(value.get("acceleration")),
            schema_version=value.get("schema_version", LOAD_OPTIONS_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_V3_VERSION,
            "schema_version": self.schema_version,
            "model_context_size": self.model_context_size,
            "batch": self.batch,
            "ubatch": self.ubatch,
            "threads": self.threads,
            "kv_cache": self.kv_cache.to_dict(),
            "acceleration": self.acceleration.to_dict(),
        }


@dataclass(frozen=True)
class GenerationOptions:
    """Per-request sampler, stopping, and explicit thinking controls."""

    max_tokens: int = 64
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    repetition_window: int | None = None
    stop: tuple[str, ...] = ()
    seed: int | None = None
    thinking_intent: ThinkingIntent = field(default_factory=lambda: ThinkingIntent(mode=ThinkingMode.OFF))
    schema_version: str = GENERATION_OPTIONS_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GENERATION_OPTIONS_VERSION:
            raise ValueError(f"generation_options.schema_version must be {GENERATION_OPTIONS_VERSION}")
        maximum = _positive_int(self.max_tokens, "generation_options.max_tokens", maximum=131_072)
        assert maximum is not None
        object.__setattr__(self, "max_tokens", maximum)
        temperature = _optional_number(self.temperature, "generation_options.temperature", minimum=0, maximum=2)
        object.__setattr__(self, "temperature", 0.0 if temperature is None else temperature)
        object.__setattr__(self, "top_p", _optional_number(self.top_p, "generation_options.top_p", minimum=0, maximum=1))
        object.__setattr__(self, "top_k", _non_negative_int(self.top_k, "generation_options.top_k", maximum=1_000_000))
        repetition_penalty = _optional_number(
            self.repetition_penalty, "generation_options.repetition_penalty", minimum=0
        )
        if repetition_penalty == 0:
            raise ValueError("generation_options.repetition_penalty must be greater than zero")
        object.__setattr__(self, "repetition_penalty", repetition_penalty)
        object.__setattr__(
            self,
            "repetition_window",
            _non_negative_int(self.repetition_window, "generation_options.repetition_window", maximum=1_048_576),
        )
        object.__setattr__(self, "seed", _non_negative_int(self.seed, "generation_options.seed", maximum=2**63 - 1))
        if not isinstance(self.stop, (tuple, list)) or any(not isinstance(item, str) for item in self.stop):
            raise ValueError("generation_options.stop must be an array of strings")
        stops = tuple(self.stop)
        if any(not item for item in stops) or len(set(stops)) != len(stops):
            raise ValueError("generation_options.stop values must be non-empty and unique")
        object.__setattr__(self, "stop", stops)
        if not isinstance(self.thinking_intent, ThinkingIntent):
            raise ValueError("generation_options.thinking_intent must be a validated ThinkingIntent")

    @classmethod
    def from_payload(cls, payload: Any) -> GenerationOptions:
        value = _mapping(payload, "generation_options")
        _keys(
            value,
            {
                "contract_version",
                "schema_version",
                "max_tokens",
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
                "repetition_window",
                "stop",
                "seed",
                "thinking_intent",
            },
            "generation_options",
        )
        if value.get("contract_version", CONTRACT_V3_VERSION) != CONTRACT_V3_VERSION:
            raise ValueError("generation_options contract_version must be runtime-foundation.contract.v3")
        stop = value.get("stop", ())
        if not isinstance(stop, (tuple, list)):
            raise ValueError("generation_options.stop must be an array")
        return cls(
            max_tokens=value.get("max_tokens", 64),
            temperature=value.get("temperature", 0.0),
            top_p=value.get("top_p"),
            top_k=value.get("top_k"),
            repetition_penalty=value.get("repetition_penalty"),
            repetition_window=value.get("repetition_window"),
            stop=tuple(stop),
            seed=value.get("seed"),
            thinking_intent=ThinkingIntent.from_payload(value.get("thinking_intent", {"mode": "OFF"})),
            schema_version=value.get("schema_version", GENERATION_OPTIONS_VERSION),
        )

    @classmethod
    def from_legacy_v1(cls, request: Any) -> GenerationOptions:
        """Explicitly map only v1 request-owned generation fields."""

        from .contracts import GenerationRequest

        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be a GenerationRequest v1 value")
        return cls(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            thinking_intent=ThinkingIntent.from_legacy(request.thinking_enabled),
        )

    @classmethod
    def from_legacy_v2(cls, request: GenerationRequestV2) -> GenerationOptions:
        """Explicitly map v2 generation controls and reject duplicate conflicts."""

        if not isinstance(request, GenerationRequestV2):
            raise TypeError("request must be a GenerationRequestV2 value")
        legacy_thinking = request.thinking_enabled
        intent_thinking = request.thinking_intent.to_legacy_enabled()
        if legacy_thinking is not None and legacy_thinking != intent_thinking:
            raise ValueError("v2 thinking_enabled conflicts with thinking_intent")
        return cls(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            thinking_intent=request.thinking_intent,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_V3_VERSION,
            "schema_version": self.schema_version,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "repetition_window": self.repetition_window,
            "stop": list(self.stop),
            "seed": self.seed,
            "thinking_intent": self.thinking_intent.to_dict(),
        }


@dataclass(frozen=True)
class ExecutionConstraints:
    """Foundation-enforced execution budgets, distinct from model load size."""

    max_context_tokens: int | None = None
    timeout_ms: int | None = None
    schema_version: str = EXECUTION_CONSTRAINTS_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTION_CONSTRAINTS_VERSION:
            raise ValueError(f"execution_constraints.schema_version must be {EXECUTION_CONSTRAINTS_VERSION}")
        object.__setattr__(
            self,
            "max_context_tokens",
            _positive_int(self.max_context_tokens, "execution_constraints.max_context_tokens", maximum=1_048_576),
        )
        object.__setattr__(
            self,
            "timeout_ms",
            _positive_int(self.timeout_ms, "execution_constraints.timeout_ms", maximum=86_400_000),
        )

    @classmethod
    def from_payload(cls, payload: Any) -> ExecutionConstraints:
        value = _mapping(payload, "execution_constraints")
        _keys(value, {"contract_version", "schema_version", "max_context_tokens", "timeout_ms"}, "execution_constraints")
        if value.get("contract_version", CONTRACT_V3_VERSION) != CONTRACT_V3_VERSION:
            raise ValueError("execution_constraints contract_version must be runtime-foundation.contract.v3")
        return cls(
            max_context_tokens=value.get("max_context_tokens"),
            timeout_ms=value.get("timeout_ms"),
            schema_version=value.get("schema_version", EXECUTION_CONSTRAINTS_VERSION),
        )

    @classmethod
    def from_legacy_v1(cls, options: RuntimeOptions, *, timeout_ms: int | None = None) -> ExecutionConstraints:
        """Map v1's documented context execution budget without changing its meaning."""

        if not isinstance(options, RuntimeOptions):
            raise TypeError("options must be RuntimeOptions v1")
        defaults = RuntimeOptions().to_dict()
        values = options.to_dict()
        unsupported_legacy_fields = [
            name for name in ("kv_cache", "prefill", "prompt_cache", "acceleration", "engine_options")
            if values[name] != defaults[name]
        ]
        if options.context.sliding_window is not None:
            unsupported_legacy_fields.append("context.sliding_window")
        if unsupported_legacy_fields:
            raise UnsupportedRuntimeOptionError(
                "the explicit v1-to-v3 conversion cannot map these legacy runtime settings",
                details={"fields": sorted(unsupported_legacy_fields), "conversion": "runtime-options-v1-to-constraints-v1"},
            )
        return cls(max_context_tokens=options.context.context_length, timeout_ms=timeout_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_V3_VERSION,
            "schema_version": self.schema_version,
            "max_context_tokens": self.max_context_tokens,
            "timeout_ms": self.timeout_ms,
        }


@dataclass(frozen=True)
class GenerationRequestV3:
    """v3 generation payload: one authoritative options object per concern."""

    model_artifact_id: str
    messages: tuple[dict[str, str], ...]
    generation_options: GenerationOptions = field(default_factory=GenerationOptions)
    execution_constraints: ExecutionConstraints = field(default_factory=ExecutionConstraints)
    request_id: str | None = None
    consumer_id: str | None = None
    lease_id: str | None = None
    studio_resolved_thinking: ThinkingIntent | None = None
    execution_input: ExecutionInputV1 | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = GENERATION_REQUEST_V3_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GENERATION_REQUEST_V3_VERSION:
            raise ValueError(f"generation_request.schema_version must be {GENERATION_REQUEST_V3_VERSION}")
        if not isinstance(self.model_artifact_id, str) or not self.model_artifact_id.strip():
            raise ValueError("model_artifact_id is required")
        object.__setattr__(self, "model_artifact_id", self.model_artifact_id.strip())
        if not isinstance(self.messages, (tuple, list)) or not self.messages:
            raise ValueError("messages must be a non-empty array")
        messages: list[dict[str, str]] = []
        for index, message in enumerate(self.messages):
            if not isinstance(message, dict) or not isinstance(message.get("role"), str) or not isinstance(
                message.get("content"), str
            ):
                raise ValueError(f"messages[{index}] requires string role and content")
            messages.append({"role": message["role"], "content": message["content"]})
        object.__setattr__(self, "messages", tuple(messages))
        for name in ("request_id", "consumer_id", "lease_id"):
            object.__setattr__(self, name, _optional_text(getattr(self, name), name))
        if not isinstance(self.generation_options, GenerationOptions):
            raise ValueError("generation_options must be a validated GenerationOptions")
        if not isinstance(self.execution_constraints, ExecutionConstraints):
            raise ValueError("execution_constraints must be a validated ExecutionConstraints")
        if self.studio_resolved_thinking is not None and not isinstance(self.studio_resolved_thinking, ThinkingIntent):
            raise ValueError("studio_resolved_thinking must be a validated ThinkingIntent")
        if self.execution_input is not None and not isinstance(self.execution_input, ExecutionInputV1):
            raise ValueError("execution_input must be a validated ExecutionInputV1")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be an object")

    @classmethod
    def from_payload(cls, payload: Any) -> GenerationRequestV3:
        value = _mapping(payload, "generation_request")
        _keys(
            value,
            {
                "contract_version",
                "schema_version",
                "model_artifact_id",
                "messages",
                "generation_options",
                "execution_constraints",
                "request_id",
                "consumer_id",
                "lease_id",
                "studio_resolved_thinking",
                "execution_input",
                "metadata",
            },
            "generation_request v3",
        )
        if value.get("contract_version", CONTRACT_V3_VERSION) != CONTRACT_V3_VERSION:
            raise ValueError("generation_request contract_version must be runtime-foundation.contract.v3")
        messages = value.get("messages")
        if not isinstance(messages, list):
            raise ValueError("messages must be an array")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        execution_input = value.get("execution_input")
        return cls(
            model_artifact_id=value.get("model_artifact_id"),
            messages=tuple(messages),
            generation_options=GenerationOptions.from_payload(value.get("generation_options")),
            execution_constraints=ExecutionConstraints.from_payload(value.get("execution_constraints")),
            request_id=value.get("request_id"),
            consumer_id=value.get("consumer_id"),
            lease_id=value.get("lease_id"),
            studio_resolved_thinking=ThinkingIntent.from_payload(value["studio_resolved_thinking"])
            if value.get("studio_resolved_thinking") is not None
            else None,
            execution_input=ExecutionInputV1.from_payload(execution_input) if execution_input is not None else None,
            metadata=dict(metadata),
            schema_version=value.get("schema_version", GENERATION_REQUEST_V3_VERSION),
        )

    @classmethod
    def from_legacy_v1(cls, request: Any) -> GenerationRequestV3:
        """Convert a v1 request explicitly; old context length remains a budget."""

        from .contracts import GenerationRequest

        if not isinstance(request, GenerationRequest):
            raise TypeError("request must be a GenerationRequest v1 value")
        constraints = ExecutionConstraints.from_legacy_v1(
            request.runtime_options,
            timeout_ms=request.timeout_ms,
        )
        return cls(
            model_artifact_id=request.model_artifact_id,
            messages=tuple(request.messages),
            generation_options=GenerationOptions.from_legacy_v1(request),
            execution_constraints=constraints,
            request_id=request.request_id,
            consumer_id=request.consumer_id,
            lease_id=request.lease_id,
            metadata=dict(request.metadata),
        )

    @classmethod
    def from_legacy_v2(cls, request: GenerationRequestV2) -> GenerationRequestV3:
        """Convert v2 only after validating its legacy and Thinking fields agree."""

        if not isinstance(request, GenerationRequestV2):
            raise TypeError("request must be a GenerationRequestV2 value")
        return cls(
            model_artifact_id=request.model_artifact_id,
            messages=tuple(request.messages),
            generation_options=GenerationOptions.from_legacy_v2(request),
            execution_constraints=ExecutionConstraints.from_legacy_v1(
                request.runtime_options,
                timeout_ms=request.timeout_ms,
            ),
            request_id=request.request_id,
            consumer_id=request.consumer_id,
            lease_id=request.lease_id,
            studio_resolved_thinking=request.studio_resolved_thinking,
            execution_input=request.execution_input,
            metadata=dict(request.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_V3_VERSION,
            "schema_version": self.schema_version,
            "model_artifact_id": self.model_artifact_id,
            "messages": [dict(item) for item in self.messages],
            "generation_options": self.generation_options.to_dict(),
            "execution_constraints": self.execution_constraints.to_dict(),
            "request_id": self.request_id,
            "consumer_id": self.consumer_id,
            "lease_id": self.lease_id,
            "studio_resolved_thinking": self.studio_resolved_thinking.to_dict()
            if self.studio_resolved_thinking
            else None,
            "execution_input": self.execution_input.to_dict() if self.execution_input else None,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LoadIdentityV1:
    """Stable identity for one loaded artifact/engine/effective-load tuple."""

    fingerprint: str
    artifact_identity: dict[str, Any]
    engine_binding: EngineBindingV2
    effective_load_options: LoadOptions
    schema_version: str = LOAD_IDENTITY_V1_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LOAD_IDENTITY_V1_VERSION:
            raise ValueError(f"load_identity.schema_version must be {LOAD_IDENTITY_V1_VERSION}")
        if not isinstance(self.artifact_identity, dict):
            raise ValueError("load_identity.artifact_identity must be an object")
        object.__setattr__(self, "artifact_identity", _canonical_copy(self.artifact_identity))
        if not isinstance(self.engine_binding, EngineBindingV2):
            raise ValueError("load_identity.engine_binding must be a validated EngineBindingV2")
        if not isinstance(self.effective_load_options, LoadOptions):
            raise ValueError("load_identity.effective_load_options must be validated LoadOptions")
        expected = canonical_fingerprint(self.canonical_payload())
        if self.fingerprint != expected:
            raise ValueError("load_identity fingerprint does not match canonical components")

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_V3_VERSION,
            "schema_version": self.schema_version,
            "artifact_identity": _canonical_copy(self.artifact_identity),
            "engine_binding": self.engine_binding.to_dict(),
            "effective_load_options": self.effective_load_options.to_dict(),
        }

    @classmethod
    def create(
        cls,
        *,
        artifact: ArtifactBindingV2,
        engine: EngineBindingV2,
        effective_load_options: LoadOptions,
    ) -> LoadIdentityV1:
        provisional = object.__new__(cls)
        object.__setattr__(provisional, "fingerprint", "")
        object.__setattr__(provisional, "artifact_identity", artifact.canonical_payload())
        object.__setattr__(provisional, "engine_binding", engine)
        object.__setattr__(provisional, "effective_load_options", effective_load_options)
        object.__setattr__(provisional, "schema_version", LOAD_IDENTITY_V1_VERSION)
        fingerprint = canonical_fingerprint(provisional.canonical_payload())
        return cls(
            fingerprint=fingerprint,
            artifact_identity=artifact.canonical_payload(),
            engine_binding=engine,
            effective_load_options=effective_load_options,
        )

    @classmethod
    def from_payload(cls, payload: Any) -> LoadIdentityV1:
        value = _mapping(payload, "load_identity")
        _keys(value, {"contract_version", "schema_version", "fingerprint", "artifact_identity", "engine_binding", "effective_load_options"}, "load_identity")
        if value.get("contract_version") != CONTRACT_V3_VERSION:
            raise ValueError("load_identity contract_version must be runtime-foundation.contract.v3")
        return cls(
            fingerprint=value.get("fingerprint"),
            artifact_identity=value.get("artifact_identity"),
            engine_binding=EngineBindingV2.from_payload(value.get("engine_binding")),
            effective_load_options=LoadOptions.from_payload(value.get("effective_load_options")),
            schema_version=value.get("schema_version", LOAD_IDENTITY_V1_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.canonical_payload(), "fingerprint": self.fingerprint}


@dataclass(frozen=True)
class OptionResolutionV3:
    """Reasoned record for a setting whose resolved/effective value changed."""

    path: str
    requested: Any
    resolved: Any
    effective: Any
    status: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("option_resolution.path is required")
        if self.status not in {"resolved", "observed", "unavailable"}:
            raise ValueError("option_resolution.status is invalid")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("option_resolution.reason is required")
        for name in ("requested", "resolved", "effective"):
            value = getattr(self, name)
            canonical_json(value)
            object.__setattr__(self, name, _canonical_copy(value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "requested": self.requested,
            "resolved": self.resolved,
            "effective": self.effective,
            "status": self.status,
            "reason": self.reason,
        }


def _leaf_values(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, child in value.items():
            if key in {"contract_version", "schema_version"}:
                continue
            path = f"{prefix}.{key}" if prefix else key
            result.update(_leaf_values(child, path))
        return result
    if isinstance(value, list):
        return {prefix: value}
    return {prefix: value}


@dataclass(frozen=True)
class SettingsEvidenceV3:
    """Requested, resolved, and effective settings for one execution scope."""

    scope: str
    requested: dict[str, Any]
    resolved: dict[str, Any]
    effective: dict[str, Any]
    resolutions: tuple[OptionResolutionV3, ...] = ()

    def __post_init__(self) -> None:
        if self.scope not in {"LOAD", "GENERATION", "CONSTRAINT"}:
            raise ValueError("settings_evidence.scope must be LOAD, GENERATION, or CONSTRAINT")
        for name in ("requested", "resolved", "effective"):
            value = getattr(self, name)
            if not isinstance(value, dict):
                raise ValueError(f"settings_evidence.{name} must be an object")
            canonical_json(value)
            object.__setattr__(self, name, _canonical_copy(value))
        if not isinstance(self.resolutions, (tuple, list)) or any(
            not isinstance(item, OptionResolutionV3) for item in self.resolutions
        ):
            raise ValueError("settings_evidence.resolutions must contain OptionResolutionV3 values")
        records = {item.path: item for item in self.resolutions}
        requested_values = _leaf_values(self.requested)
        resolved_values = _leaf_values(self.resolved)
        effective_values = _leaf_values(self.effective)
        changed_paths = {
            path
            for path in set(requested_values) | set(resolved_values) | set(effective_values)
            if requested_values.get(path) != resolved_values.get(path)
            or resolved_values.get(path) != effective_values.get(path)
        }
        if not changed_paths.issubset(records):
            raise ValueError(
                "settings evidence changes require explicit resolution records: "
                + ", ".join(sorted(changed_paths - set(records)))
            )
        for path in changed_paths:
            record = records[path]
            if (
                record.requested != requested_values.get(path)
                or record.resolved != resolved_values.get(path)
                or record.effective != effective_values.get(path)
            ):
                raise ValueError(f"option resolution values do not match settings evidence at {path}")

    @classmethod
    def from_options(
        cls,
        *,
        scope: str,
        requested: LoadOptions | GenerationOptions | ExecutionConstraints,
        resolved: LoadOptions | GenerationOptions | ExecutionConstraints,
        effective: LoadOptions | GenerationOptions | ExecutionConstraints,
        resolutions: tuple[OptionResolutionV3, ...] = (),
    ) -> SettingsEvidenceV3:
        values = (requested.to_dict(), resolved.to_dict(), effective.to_dict())
        return cls(scope=scope, requested=values[0], resolved=values[1], effective=values[2], resolutions=resolutions)

    @classmethod
    def from_payload(cls, payload: Any) -> SettingsEvidenceV3:
        value = _mapping(payload, "settings_evidence")
        _keys(value, {"scope", "requested", "resolved", "effective", "resolutions"}, "settings_evidence")
        resolutions = value.get("resolutions", [])
        if not isinstance(resolutions, list):
            raise ValueError("settings_evidence.resolutions must be an array")
        return cls(
            scope=value.get("scope"),
            requested=value.get("requested"),
            resolved=value.get("resolved"),
            effective=value.get("effective"),
            resolutions=tuple(OptionResolutionV3(**item) for item in resolutions),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "requested": _canonical_copy(self.requested),
            "resolved": _canonical_copy(self.resolved),
            "effective": _canonical_copy(self.effective),
            "resolutions": [item.to_dict() for item in self.resolutions],
        }


@dataclass(frozen=True)
class LoadOptionsResolutionV1:
    requested: LoadOptions
    resolved: LoadOptions
    effective: LoadOptions
    resolutions: tuple[OptionResolutionV3, ...] = ()
    schema_version: str = "runtime-foundation.load-options-resolution.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "runtime-foundation.load-options-resolution.v1":
            raise ValueError("load_options_resolution.schema_version is unsupported")
        if not all(isinstance(value, LoadOptions) for value in (self.requested, self.resolved, self.effective)):
            raise ValueError("load_options_resolution values must be validated LoadOptions")
        SettingsEvidenceV3.from_options(
            scope="LOAD",
            requested=self.requested,
            resolved=self.resolved,
            effective=self.effective,
            resolutions=self.resolutions,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_V3_VERSION,
            "schema_version": self.schema_version,
            "requested": self.requested.to_dict(),
            "resolved": self.resolved.to_dict(),
            "effective": self.effective.to_dict(),
            "resolutions": [item.to_dict() for item in self.resolutions],
        }


@dataclass(frozen=True)
class ExecutionOptionsEvidenceV3:
    load: SettingsEvidenceV3
    generation: SettingsEvidenceV3
    constraints: SettingsEvidenceV3

    def __post_init__(self) -> None:
        if self.load.scope != "LOAD" or self.generation.scope != "GENERATION" or self.constraints.scope != "CONSTRAINT":
            raise ValueError("execution options evidence scopes do not match their fields")

    @classmethod
    def from_payload(cls, payload: Any) -> ExecutionOptionsEvidenceV3:
        value = _mapping(payload, "execution_options_evidence")
        _keys(value, {"load", "generation", "constraints"}, "execution_options_evidence")
        return cls(
            load=SettingsEvidenceV3.from_payload(value.get("load")),
            generation=SettingsEvidenceV3.from_payload(value.get("generation")),
            constraints=SettingsEvidenceV3.from_payload(value.get("constraints")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"load": self.load.to_dict(), "generation": self.generation.to_dict(), "constraints": self.constraints.to_dict()}


@dataclass(frozen=True)
class ExecutionBindingV3:
    """Fingerprint for one v3 execution using observed effective settings."""

    artifact_binding: ArtifactBindingV2
    engine_binding: EngineBindingV2
    foundation_binding: FoundationBindingV2
    load_identity: LoadIdentityV1
    effective_load_options: LoadOptions
    effective_generation_options: GenerationOptions
    effective_constraints: ExecutionConstraints
    execution_input: ExecutionInputV1 | None = None

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, expected)
            for value, expected in (
                (self.artifact_binding, ArtifactBindingV2),
                (self.engine_binding, EngineBindingV2),
                (self.foundation_binding, FoundationBindingV2),
                (self.load_identity, LoadIdentityV1),
                (self.effective_load_options, LoadOptions),
                (self.effective_generation_options, GenerationOptions),
                (self.effective_constraints, ExecutionConstraints),
            )
        ):
            raise ValueError("execution_binding v3 fields must use validated contract values")
        if self.load_identity.artifact_identity != self.artifact_binding.canonical_payload():
            raise ValueError("load_identity artifact does not match execution binding artifact")
        if self.load_identity.engine_binding != self.engine_binding:
            raise ValueError("load_identity engine does not match execution binding engine")
        if self.load_identity.effective_load_options != self.effective_load_options:
            raise ValueError("load_identity options do not match execution binding effective load options")
        if self.execution_input is not None:
            if not isinstance(self.execution_input, ExecutionInputV1):
                raise ValueError("execution_input must be a validated ExecutionInputV1")
            if self.execution_input.base.canonical_payload() != self.artifact_binding.canonical_payload():
                raise ValueError("execution_input base does not match execution binding artifact")

    def canonical_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "contract_version": CONTRACT_V3_VERSION,
            "schema_version": EXECUTION_BINDING_V3_VERSION,
            "artifact_identity": self.artifact_binding.canonical_payload(),
            "engine_binding": self.engine_binding.to_dict(),
            "foundation_binding": self.foundation_binding.to_dict(),
            "load_identity": self.load_identity.to_dict(),
            "effective_load_options": self.effective_load_options.to_dict(),
            "effective_generation_options": self.effective_generation_options.to_dict(),
            "effective_constraints": self.effective_constraints.to_dict(),
        }
        if self.execution_input is not None:
            payload["execution_input"] = self.execution_input.canonical_payload()
        return payload

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.canonical_payload())

    @classmethod
    def create(
        cls,
        *,
        artifact: ArtifactBindingV2,
        engine: EngineBindingV2,
        foundation: FoundationBindingV2,
        effective_load_options: LoadOptions,
        effective_generation_options: GenerationOptions,
        effective_constraints: ExecutionConstraints,
        execution_input: ExecutionInputV1 | None = None,
    ) -> ExecutionBindingV3:
        return cls(
            artifact_binding=artifact,
            engine_binding=engine,
            foundation_binding=foundation,
            load_identity=LoadIdentityV1.create(
                artifact=artifact,
                engine=engine,
                effective_load_options=effective_load_options,
            ),
            effective_load_options=effective_load_options,
            effective_generation_options=effective_generation_options,
            effective_constraints=effective_constraints,
            execution_input=execution_input,
        )

    @classmethod
    def from_payload(cls, payload: Any) -> ExecutionBindingV3:
        value = _mapping(payload, "execution_binding")
        _keys(
            value,
            {
                "contract_version",
                "schema_version",
                "artifact_binding",
                "engine_binding",
                "foundation_binding",
                "load_identity",
                "effective_load_options",
                "effective_generation_options",
                "effective_constraints",
                "execution_input",
                "execution_binding_fingerprint",
            },
            "execution_binding v3",
        )
        if value.get("contract_version") != CONTRACT_V3_VERSION:
            raise ValueError("execution_binding contract_version must be runtime-foundation.contract.v3")
        if value.get("schema_version") != EXECUTION_BINDING_V3_VERSION:
            raise ValueError(f"execution_binding.schema_version must be {EXECUTION_BINDING_V3_VERSION}")
        binding = cls(
            artifact_binding=ArtifactBindingV2.from_payload(value.get("artifact_binding")),
            engine_binding=EngineBindingV2.from_payload(value.get("engine_binding")),
            foundation_binding=FoundationBindingV2(**_mapping(value.get("foundation_binding"), "foundation_binding")),
            load_identity=LoadIdentityV1.from_payload(value.get("load_identity")),
            effective_load_options=LoadOptions.from_payload(value.get("effective_load_options")),
            effective_generation_options=GenerationOptions.from_payload(value.get("effective_generation_options")),
            effective_constraints=ExecutionConstraints.from_payload(value.get("effective_constraints")),
            execution_input=ExecutionInputV1.from_payload(value["execution_input"])
            if value.get("execution_input") is not None
            else None,
        )
        expected = value.get("execution_binding_fingerprint")
        if expected is not None and expected != binding.fingerprint:
            raise ValueError("execution_binding_fingerprint does not match execution binding v3")
        if binding.load_identity.artifact_identity != binding.artifact_binding.canonical_payload():
            raise ValueError("load_identity artifact does not match execution binding artifact")
        if binding.load_identity.engine_binding != binding.engine_binding:
            raise ValueError("load_identity engine does not match execution binding engine")
        if binding.load_identity.effective_load_options != binding.effective_load_options:
            raise ValueError("load_identity options do not match execution binding effective load options")
        return binding

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "contract_version": CONTRACT_V3_VERSION,
            "schema_version": EXECUTION_BINDING_V3_VERSION,
            "artifact_binding": self.artifact_binding.to_dict(),
            "engine_binding": self.engine_binding.to_dict(),
            "foundation_binding": self.foundation_binding.to_dict(),
            "load_identity": self.load_identity.to_dict(),
            "effective_load_options": self.effective_load_options.to_dict(),
            "effective_generation_options": self.effective_generation_options.to_dict(),
            "effective_constraints": self.effective_constraints.to_dict(),
        }
        if self.execution_input is not None:
            payload["execution_input"] = self.execution_input.to_dict()
        return {**payload, "execution_binding_fingerprint": self.fingerprint}


@dataclass(frozen=True)
class ExecutionEvidenceV3:
    execution_binding: ExecutionBindingV3
    options: ExecutionOptionsEvidenceV3
    schema_version: str = EXECUTION_EVIDENCE_V3_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTION_EVIDENCE_V3_VERSION:
            raise ValueError(f"execution_evidence.schema_version must be {EXECUTION_EVIDENCE_V3_VERSION}")
        if not isinstance(self.execution_binding, ExecutionBindingV3) or not isinstance(
            self.options, ExecutionOptionsEvidenceV3
        ):
            raise ValueError("execution_evidence requires validated binding and options evidence")
        if self.options.load.effective != self.execution_binding.effective_load_options.to_dict():
            raise ValueError("effective LoadOptions evidence does not match execution binding")
        if self.options.generation.effective != self.execution_binding.effective_generation_options.to_dict():
            raise ValueError("effective GenerationOptions evidence does not match execution binding")
        if self.options.constraints.effective != self.execution_binding.effective_constraints.to_dict():
            raise ValueError("effective ExecutionConstraints evidence does not match execution binding")

    @classmethod
    def from_payload(cls, payload: Any) -> ExecutionEvidenceV3:
        value = _mapping(payload, "execution_evidence")
        _keys(value, {"contract_version", "schema_version", "execution_binding", "options"}, "execution_evidence")
        if value.get("contract_version") != CONTRACT_V3_VERSION:
            raise ValueError("execution_evidence contract_version must be runtime-foundation.contract.v3")
        return cls(
            execution_binding=ExecutionBindingV3.from_payload(value.get("execution_binding")),
            options=ExecutionOptionsEvidenceV3.from_payload(value.get("options")),
            schema_version=value.get("schema_version", EXECUTION_EVIDENCE_V3_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_V3_VERSION,
            "schema_version": self.schema_version,
            "execution_binding": self.execution_binding.to_dict(),
            "execution_binding_fingerprint": self.execution_binding.fingerprint,
            "options": self.options.to_dict(),
        }


@dataclass
class ExecutionTraceV3:
    execution_id: str
    request_id: str
    engine: dict[str, Any]
    artifact_identity: dict[str, Any]
    status: str
    started_at: str
    evidence: ExecutionEvidenceV3 | None = None
    finished_at: str | None = None
    metrics: dict[str, Any] | None = None
    finish_reason: str | None = None
    error: dict[str, Any] | None = None
    schema_version: str = EXECUTION_TRACE_V3_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_V3_VERSION,
            "schema_version": self.schema_version,
            "execution_id": self.execution_id,
            "request_id": self.request_id,
            "engine": dict(self.engine),
            "artifact_identity": dict(self.artifact_identity),
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metrics": self.metrics,
            "finish_reason": self.finish_reason,
            "error": self.error,
            "execution_evidence": self.evidence.to_dict() if self.evidence else None,
        }


@dataclass
class GenerationResultV3:
    request_id: str
    model_artifact_id: str
    engine: str
    text: str
    finish_reason: str
    usage: dict[str, Any]
    metrics: dict[str, Any]
    execution_id: str
    evidence: ExecutionEvidenceV3
    schema_version: str = GENERATION_RESULT_V3_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_V3_VERSION,
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "model_artifact_id": self.model_artifact_id,
            "engine": self.engine,
            "text": self.text,
            "finish_reason": self.finish_reason,
            "usage": dict(self.usage),
            "metrics": dict(self.metrics),
            "execution_id": self.execution_id,
            "execution_evidence": self.evidence.to_dict(),
        }


@dataclass
class StreamEventV3:
    type: str
    request_id: str
    sequence: int
    execution_id: str
    evidence: ExecutionEvidenceV3 | None = None
    delta: str = ""
    done: bool = False
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    timestamp: str | None = None
    schema_version: str = STREAM_EVENT_V3_VERSION

    def to_dict(self) -> dict[str, Any]:
        from .contracts import utc_now

        return {
            "contract_version": CONTRACT_V3_VERSION,
            "schema_version": self.schema_version,
            "type": self.type,
            "request_id": self.request_id,
            "sequence": self.sequence,
            "execution_id": self.execution_id,
            "delta": self.delta,
            "done": self.done,
            "result": self.result,
            "error": self.error,
            "execution_evidence": self.evidence.to_dict() if self.evidence else None,
            "timestamp": self.timestamp or utc_now(),
        }


__all__ = [
    "CONTRACT_V3_VERSION",
    "SUPPORTED_CONTRACT_VERSIONS_V3",
    "EXECUTION_CONSTRAINTS_VERSION",
    "EXECUTION_BINDING_V3_VERSION",
    "EXECUTION_EVIDENCE_V3_VERSION",
    "EXECUTION_TRACE_V3_VERSION",
    "GENERATION_RESULT_V3_VERSION",
    "STREAM_EVENT_V3_VERSION",
    "GENERATION_OPTIONS_VERSION",
    "GENERATION_REQUEST_V3_VERSION",
    "LOAD_OPTIONS_VERSION",
    "LOAD_IDENTITY_V1_VERSION",
    "AccelerationConfiguration",
    "ExecutionConstraints",
    "ExecutionBindingV3",
    "ExecutionEvidenceV3",
    "ExecutionTraceV3",
    "ExecutionOptionsEvidenceV3",
    "GenerationOptions",
    "GenerationRequestV3",
    "GenerationResultV3",
    "StreamEventV3",
    "KVCacheConfiguration",
    "LoadOptions",
    "LoadOptionsResolutionV1",
    "LoadIdentityV1",
    "OptionResolutionV3",
    "SettingsEvidenceV3",
]
