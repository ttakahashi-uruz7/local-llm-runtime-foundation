"""Foundation Contract v2 primitives.

This module is deliberately additive.  The v1 contracts remain in
``runtime_foundation.contracts`` and are still the wire contract used by the
existing service paths.  v2 adds explicit execution provenance and stable
canonical fingerprints without importing Studio policy into Foundation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .contracts import (
    CONTRACT_VERSION,
    GENERATION_REQUEST_VERSION,
    GenerationRequest,
    GenerationResult,
    ModelArtifactBinding,
    RuntimeOptions,
    _mapping,
    _non_negative_int,
)

CONTRACT_V2_VERSION = "runtime-foundation.contract.v2"
GENERATION_REQUEST_V2_VERSION = "runtime-foundation.generation-request.v2"
GENERATION_RESULT_V2_VERSION = "runtime-foundation.generation-result.v2"
EXECUTION_TRACE_V2_VERSION = "runtime-foundation.execution-trace.v2"
EXECUTION_GUARD_VERSION = "runtime-foundation.execution-guard.v1"
EXECUTION_BINDING_VERSION = "runtime-foundation.execution-binding.v1"

SUPPORTED_CONTRACT_VERSIONS = (CONTRACT_VERSION, CONTRACT_V2_VERSION)


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data deterministically.

    Object key order is normalized and insignificant whitespace is removed.
    ``allow_nan=False`` prevents non-JSON numeric values from acquiring a
    platform-specific representation.
    """

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_fingerprint(value: Any) -> str:
    """Return a self-describing SHA-256 fingerprint for canonical JSON data."""

    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")
    return value.strip()


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_dir():
        for child in sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
            relative = child.relative_to(path).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            with child.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    else:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _fast_file_digest(path: Path) -> str:
    """Create a development/change-detection digest without claiming full content."""

    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("ascii"))
    digest.update(str(stat.st_mtime_ns).encode("ascii"))
    if path.is_file():
        with path.open("rb") as handle:
            first = handle.read(64 * 1024)
            digest.update(first)
            if stat.st_size > len(first):
                handle.seek(max(0, stat.st_size - 64 * 1024))
                digest.update(handle.read(64 * 1024))
    return digest.hexdigest()


class ContentIdentityScheme(str, Enum):
    COMPLETE = "complete"
    FAST = "fast"
    LEGACY = "legacy"


@dataclass(frozen=True)
class ContentIdentity:
    """Content identity independent of any filesystem locator."""

    algorithm: str
    digest: str
    scope: str = "artifact"
    scheme: str = ContentIdentityScheme.COMPLETE.value

    def __post_init__(self) -> None:
        algorithm = _required_text(self.algorithm, "content_identity.algorithm").lower()
        digest = _required_text(self.digest, "content_identity.digest")
        scope = _required_text(self.scope, "content_identity.scope")
        scheme = _required_text(self.scheme, "content_identity.scheme").lower()
        if ":" in digest and digest.lower().startswith(f"{algorithm}:"):
            digest = digest.split(":", 1)[1]
        if algorithm == "sha256" and scheme == ContentIdentityScheme.COMPLETE.value:
            if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
                raise ValueError("complete sha256 content identity requires a 64-character hexadecimal digest")
            digest = digest.lower()
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "scheme", scheme)

    @classmethod
    def from_payload(cls, payload: Any) -> ContentIdentity:
        value = _mapping(payload, "content_identity")
        allowed = {"algorithm", "digest", "scope", "scheme"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unsupported fields in content_identity: {', '.join(unknown)}")
        return cls(
            algorithm=value.get("algorithm", "sha256"),
            digest=_required_text(value.get("digest"), "content_identity.digest"),
            scope=_required_text(value.get("scope", "artifact"), "content_identity.scope"),
            scheme=_required_text(value.get("scheme", ContentIdentityScheme.COMPLETE.value), "content_identity.scheme"),
        )

    @classmethod
    def from_legacy_hash(cls, value: str) -> ContentIdentity:
        raw = _required_text(value, "artifact_hash")
        algorithm, separator, digest = raw.partition(":")
        if not separator:
            algorithm, digest = "sha256", raw
        scheme = (
            ContentIdentityScheme.COMPLETE.value
            if re.fullmatch(r"[0-9a-fA-F]{64}", digest)
            else ContentIdentityScheme.LEGACY.value
        )
        return cls(algorithm=algorithm, digest=digest, scope="artifact", scheme=scheme)

    @classmethod
    def from_file(cls, path: str | Path, *, scheme: str = "complete", scope: str = "artifact") -> ContentIdentity:
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(target)
        if scheme == ContentIdentityScheme.COMPLETE.value:
            digest = _sha256_file(target)
        elif scheme == ContentIdentityScheme.FAST.value:
            digest = _fast_file_digest(target)
        else:
            raise ValueError("content identity scheme must be complete or fast")
        return cls(algorithm="sha256", digest=digest, scope=scope, scheme=scheme)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "digest": self.digest,
            "scope": self.scope,
            "scheme": self.scheme,
        }


@dataclass(frozen=True)
class ArtifactLocator:
    """A place where an artifact can be found; never a content identity."""

    type: str
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", _required_text(self.type, "locator.type").lower())
        object.__setattr__(self, "value", _required_text(self.value, "locator.value"))

    @classmethod
    def from_payload(cls, payload: Any) -> ArtifactLocator:
        value = _mapping(payload, "locator")
        unknown = sorted(set(value) - {"type", "value"})
        if unknown:
            raise ValueError(f"unsupported fields in locator: {', '.join(unknown)}")
        return cls(
            type=_required_text(value.get("type"), "locator.type"),
            value=_required_text(value.get("value"), "locator.value"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"type": self.type, "value": self.value}


@dataclass(frozen=True)
class ArtifactBindingV2:
    """Registry identity, content identity, and locator with separate semantics."""

    artifact_id: str
    locator: ArtifactLocator
    content_identity: ContentIdentity | None = None
    format: str | None = None
    quantization: str | None = None
    revision: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _required_text(self.artifact_id, "artifact_id"))
        if self.format is not None:
            object.__setattr__(self, "format", _required_text(self.format, "format").lower())

    @property
    def local_path(self) -> str | None:
        return self.locator.value if self.locator.type == "filesystem" else None

    @classmethod
    def from_payload(cls, payload: Any) -> ArtifactBindingV2:
        value = _mapping(payload, "artifact")
        allowed = {
            "contract_version",
            "registry_identity",
            "artifact_id",
            "content_identity",
            "locator",
            "local_path",
            "format",
            "quantization",
            "revision",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unsupported fields in artifact v2: {', '.join(unknown)}")
        version = value.get("contract_version")
        if version not in {None, CONTRACT_V2_VERSION}:
            raise ValueError("artifact contract_version is not runtime-foundation.contract.v2")
        registry = (
            _mapping(value.get("registry_identity"), "registry_identity")
            if value.get("registry_identity") is not None
            else {}
        )
        artifact_id = value.get("artifact_id", registry.get("artifact_id"))
        locator_payload = value.get("locator")
        if locator_payload is None and value.get("local_path") is not None:
            locator_payload = {"type": "filesystem", "value": value.get("local_path")}
        return cls(
            artifact_id=_required_text(artifact_id, "artifact_id"),
            locator=ArtifactLocator.from_payload(locator_payload),
            content_identity=(
                ContentIdentity.from_payload(value["content_identity"])
                if value.get("content_identity") is not None
                else None
            ),
            format=value.get("format"),
            quantization=value.get("quantization"),
            revision=value.get("revision"),
        )

    @classmethod
    def from_legacy(cls, artifact: ModelArtifactBinding) -> ArtifactBindingV2:
        content = ContentIdentity.from_legacy_hash(artifact.artifact_hash) if artifact.artifact_hash else None
        return cls(
            artifact_id=artifact.artifact_id,
            locator=ArtifactLocator(type="filesystem", value=artifact.local_path),
            content_identity=content,
            format=artifact.format,
            quantization=artifact.quantization,
            revision=artifact.revision,
        )

    def to_legacy(self) -> ModelArtifactBinding:
        artifact_hash = None
        if self.content_identity is not None:
            artifact_hash = f"{self.content_identity.algorithm}:{self.content_identity.digest}"
        return ModelArtifactBinding(
            artifact_id=self.artifact_id,
            local_path=self.local_path or self.locator.value,
            format=self.format or "unknown",
            quantization=self.quantization,
            artifact_hash=artifact_hash,
            revision=self.revision,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_V2_VERSION,
            "registry_identity": {"artifact_id": self.artifact_id},
            "artifact_id": self.artifact_id,
            "content_identity": self.content_identity.to_dict() if self.content_identity else None,
            "locator": self.locator.to_dict(),
            "format": self.format,
            "quantization": self.quantization,
            "revision": self.revision,
        }


@dataclass(frozen=True)
class EngineImplementationBinding:
    id: str
    version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_text(self.id, "implementation.id"))
        object.__setattr__(self, "version", _optional_text(self.version, "implementation.version"))

    def to_dict(self) -> dict[str, str | None]:
        return {"id": self.id, "version": self.version}


@dataclass(frozen=True)
class EngineBindingV2:
    family: str
    implementation: EngineImplementationBinding
    build_identity: str | None
    adapter_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _required_text(self.family, "engine.family").lower())
        object.__setattr__(self, "build_identity", _optional_text(self.build_identity, "engine.build_identity"))
        object.__setattr__(self, "adapter_id", _required_text(self.adapter_id, "engine.adapter_id"))

    @classmethod
    def from_legacy(cls, identity: Any, *, adapter_id: str) -> EngineBindingV2:
        engine = _required_text(getattr(identity, "engine", None), "engine")
        if engine == "mlx":
            family, implementation_id = "mlx", "mlx-lm"
            candidate = getattr(identity, "build", None)
            # The current MLX v1 identity historically exposed the mlx
            # package version in ``build``.  That is not an implementation
            # build identity.  Preserve an explicitly build-shaped value,
            # while leaving package-version-shaped/unknown values null.
            build_identity = (
                candidate
                if isinstance(candidate, str)
                and candidate.strip()
                and re.fullmatch(r"v?\d+(?:\.\d+){1,3}(?:[-+].*)?", candidate.strip()) is None
                else None
            )
        elif engine == "mock":
            family, implementation_id, build_identity = "mock", "mock-runtime", getattr(identity, "build", None)
        elif engine in {"llama.cpp", "llama_cpp"}:
            family, implementation_id, build_identity = "llama.cpp", "llama.cpp", None
        else:
            family, implementation_id, build_identity = engine, engine, getattr(identity, "build", None)
        return cls(
            family=family,
            implementation=EngineImplementationBinding(
                id=implementation_id,
                version=getattr(identity, "version", None),
            ),
            build_identity=build_identity,
            adapter_id=adapter_id,
        )

    @classmethod
    def from_payload(cls, payload: Any) -> EngineBindingV2:
        value = _mapping(payload, "engine_binding")
        implementation = EngineImplementationBinding(**_mapping(value.get("implementation"), "implementation"))
        return cls(
            family=_required_text(value.get("family"), "engine.family"),
            implementation=implementation,
            build_identity=value.get("build_identity"),
            adapter_id=_required_text(value.get("adapter_id"), "engine.adapter_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "implementation": self.implementation.to_dict(),
            "build_identity": self.build_identity,
            "adapter_id": self.adapter_id,
        }

    @property
    def implementation_id(self) -> str:
        return self.implementation.id

    @property
    def implementation_version(self) -> str | None:
        return self.implementation.version


@dataclass(frozen=True)
class FoundationBindingV2:
    contract_version: str
    foundation_version: str
    build_identity: str | None
    adapter_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "contract_version", _required_text(self.contract_version, "foundation.contract_version")
        )
        object.__setattr__(
            self, "foundation_version", _required_text(self.foundation_version, "foundation.foundation_version")
        )
        object.__setattr__(self, "build_identity", _optional_text(self.build_identity, "foundation.build_identity"))
        object.__setattr__(self, "adapter_id", _required_text(self.adapter_id, "foundation.adapter_id"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "foundation_version": self.foundation_version,
            "build_identity": self.build_identity,
            "adapter_id": self.adapter_id,
        }


@dataclass(frozen=True)
class RuntimeSettingsBindingV2:
    runtime_options_schema_version: str
    exact_settings: dict[str, Any] | None
    exact_settings_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "runtime_options_schema_version",
            _required_text(self.runtime_options_schema_version, "runtime_options_schema_version"),
        )
        if self.exact_settings is not None:
            if not isinstance(self.exact_settings, dict):
                raise ValueError("exact_settings must be an object")
            exact = dict(self.exact_settings)
            object.__setattr__(self, "exact_settings", exact)
            expected = canonical_fingerprint(exact)
            if self.exact_settings_fingerprint is not None and self.exact_settings_fingerprint != expected:
                raise ValueError("exact_settings_fingerprint does not match exact_settings")
            object.__setattr__(self, "exact_settings_fingerprint", expected)

    @classmethod
    def from_effective(cls, options: RuntimeOptions | None) -> RuntimeSettingsBindingV2 | None:
        if options is None:
            return None
        return cls(
            runtime_options_schema_version=options.schema_version,
            exact_settings=options.to_dict(),
        )

    @classmethod
    def from_payload(cls, payload: Any) -> RuntimeSettingsBindingV2:
        value = _mapping(payload, "runtime_settings_binding")
        return cls(
            runtime_options_schema_version=_required_text(
                value.get("runtime_options_schema_version"), "runtime_options_schema_version"
            ),
            exact_settings=value.get("exact_settings"),
            exact_settings_fingerprint=value.get("exact_settings_fingerprint"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_options_schema_version": self.runtime_options_schema_version,
            "exact_settings": dict(self.exact_settings) if self.exact_settings is not None else None,
            "exact_settings_fingerprint": self.exact_settings_fingerprint,
        }


@dataclass(frozen=True)
class ExecutionBindingV2:
    artifact_binding: ArtifactBindingV2
    engine_binding: EngineBindingV2
    foundation_binding: FoundationBindingV2
    runtime_settings_binding: RuntimeSettingsBindingV2 | None

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_V2_VERSION,
            "schema_version": EXECUTION_BINDING_VERSION,
            "artifact_binding": self.artifact_binding.to_dict(),
            "engine_binding": self.engine_binding.to_dict(),
            "foundation_binding": self.foundation_binding.to_dict(),
            "runtime_settings_binding": self.runtime_settings_binding.to_dict()
            if self.runtime_settings_binding
            else None,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.canonical_payload())

    @classmethod
    def from_payload(cls, payload: Any) -> ExecutionBindingV2:
        value = _mapping(payload, "execution_binding")
        expected = value.get("execution_binding_fingerprint")
        binding = cls(
            artifact_binding=ArtifactBindingV2.from_payload(value.get("artifact_binding")),
            engine_binding=EngineBindingV2.from_payload(value.get("engine_binding")),
            foundation_binding=FoundationBindingV2(**_mapping(value.get("foundation_binding"), "foundation_binding")),
            runtime_settings_binding=(
                RuntimeSettingsBindingV2.from_payload(value["runtime_settings_binding"])
                if value.get("runtime_settings_binding") is not None
                else None
            ),
        )
        if expected is not None and expected != binding.fingerprint:
            raise ValueError("execution_binding_fingerprint does not match execution binding")
        return binding

    def to_dict(self) -> dict[str, Any]:
        return {**self.canonical_payload(), "execution_binding_fingerprint": self.fingerprint}


class ThinkingMode(str, Enum):
    OFF = "OFF"
    AUTO = "AUTO"
    ON = "ON"


class ThinkingEffort(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True)
class ThinkingIntent:
    mode: ThinkingMode | str = ThinkingMode.AUTO
    effort: ThinkingEffort | str | None = None
    budget_tokens: int | None = None

    def __post_init__(self) -> None:
        try:
            mode = self.mode if isinstance(self.mode, ThinkingMode) else ThinkingMode(str(self.mode).upper())
        except ValueError as exc:
            raise ValueError("thinking_intent.mode must be OFF, AUTO, or ON") from exc
        effort: ThinkingEffort | None
        if self.effort is None:
            effort = None
        else:
            try:
                effort = (
                    self.effort if isinstance(self.effort, ThinkingEffort) else ThinkingEffort(str(self.effort).upper())
                )
            except ValueError as exc:
                raise ValueError("thinking_intent.effort must be LOW, MEDIUM, or HIGH") from exc
        budget = _non_negative_int(self.budget_tokens, "thinking_intent.budget_tokens", maximum=10_000_000)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "effort", effort)
        object.__setattr__(self, "budget_tokens", budget)

    @classmethod
    def from_payload(cls, payload: Any) -> ThinkingIntent:
        if payload is None:
            return cls()
        value = _mapping(payload, "thinking_intent")
        unknown = sorted(set(value) - {"mode", "effort", "budget_tokens"})
        if unknown:
            raise ValueError(f"unsupported fields in thinking_intent: {', '.join(unknown)}")
        return cls(mode=value.get("mode", "AUTO"), effort=value.get("effort"), budget_tokens=value.get("budget_tokens"))

    @classmethod
    def from_legacy(cls, thinking_enabled: bool | None) -> ThinkingIntent:
        if thinking_enabled is False:
            return cls(mode=ThinkingMode.OFF)
        if thinking_enabled is True:
            return cls(mode=ThinkingMode.ON)
        return cls(mode=ThinkingMode.AUTO)

    def to_legacy_enabled(self) -> bool | None:
        if self.mode == ThinkingMode.OFF:
            return False
        if self.mode == ThinkingMode.ON:
            return True
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value if isinstance(self.mode, ThinkingMode) else str(self.mode),
            "effort": (
                self.effort.value
                if isinstance(self.effort, ThinkingEffort)
                else str(self.effort)
                if self.effort is not None
                else None
            ),
            "budget_tokens": self.budget_tokens,
        }


@dataclass(frozen=True)
class ThinkingResolution:
    requested: ThinkingIntent
    studio_resolved: ThinkingIntent | None
    foundation_effective: ThinkingIntent | None
    status: str = "unresolved"
    reason: str | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested.to_dict(),
            "studio_resolved": self.studio_resolved.to_dict() if self.studio_resolved else None,
            "foundation_effective": self.foundation_effective.to_dict() if self.foundation_effective else None,
            "status": self.status,
            "reason": self.reason,
            "error": dict(self.error) if self.error else None,
        }


@dataclass(frozen=True)
class ExecutionGuardV1:
    expected_execution_binding_fingerprint: str | None = None
    expected_runtime_settings_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_V2_VERSION,
            "schema_version": EXECUTION_GUARD_VERSION,
            "expected_execution_binding_fingerprint": self.expected_execution_binding_fingerprint,
            "expected_runtime_settings_fingerprint": self.expected_runtime_settings_fingerprint,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> ExecutionGuardV1 | None:
        if payload is None:
            return None
        value = _mapping(payload, "execution_guard")
        allowed = {
            "contract_version",
            "schema_version",
            "expected_execution_binding_fingerprint",
            "expected_runtime_settings_fingerprint",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unsupported fields in execution_guard: {', '.join(unknown)}")
        return cls(
            expected_execution_binding_fingerprint=_optional_text(
                value.get("expected_execution_binding_fingerprint"), "expected_execution_binding_fingerprint"
            ),
            expected_runtime_settings_fingerprint=_optional_text(
                value.get("expected_runtime_settings_fingerprint"), "expected_runtime_settings_fingerprint"
            ),
        )


# Public short spelling for callers that use the contract concept rather than
# its versioned wire name.
ExecutionGuard = ExecutionGuardV1


@dataclass(frozen=True)
class GenerationRequestV2(GenerationRequest):
    """Generation request v2 with explicit thinking intent and guard fields."""

    thinking_intent: ThinkingIntent = field(default_factory=ThinkingIntent)
    studio_resolved_thinking: ThinkingIntent | None = None
    execution_guard: ExecutionGuardV1 | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> GenerationRequestV2:
        value = _mapping(payload, "generation_request")
        version = value.get("contract_version")
        schema = value.get("schema_version")
        if version not in {None, CONTRACT_VERSION, CONTRACT_V2_VERSION}:
            raise ValueError("generation_request contract_version is not supported")
        if schema not in {None, GENERATION_REQUEST_VERSION, GENERATION_REQUEST_V2_VERSION}:
            raise ValueError("generation_request schema_version is not supported")
        legacy_payload = dict(value)
        legacy_payload["contract_version"] = CONTRACT_VERSION
        legacy_payload["schema_version"] = GENERATION_REQUEST_VERSION
        thinking_intent_payload = legacy_payload.pop("thinking_intent", None)
        studio_payload = legacy_payload.pop("studio_resolved_thinking", None)
        guard_payload = legacy_payload.pop("execution_guard", None)
        expected_execution = legacy_payload.pop("expected_execution_binding_fingerprint", None)
        expected_settings = legacy_payload.pop("expected_runtime_settings_fingerprint", None)
        base = GenerationRequest.from_payload(legacy_payload)
        thinking_intent = (
            ThinkingIntent.from_payload(thinking_intent_payload)
            if thinking_intent_payload is not None
            else ThinkingIntent.from_legacy(base.thinking_enabled)
        )
        guard = ExecutionGuardV1.from_payload(guard_payload)
        if expected_execution is not None or expected_settings is not None:
            top_level_guard = ExecutionGuardV1(
                expected_execution_binding_fingerprint=expected_execution,
                expected_runtime_settings_fingerprint=expected_settings,
            )
            if guard is not None and guard != top_level_guard:
                raise ValueError("execution_guard and top-level expected fingerprints disagree")
            guard = top_level_guard
        return cls(
            model_artifact_id=base.model_artifact_id,
            messages=base.messages,
            request_id=base.request_id,
            consumer_id=base.consumer_id,
            lease_id=base.lease_id,
            max_tokens=base.max_tokens,
            temperature=base.temperature,
            top_p=base.top_p,
            thinking_enabled=base.thinking_enabled,
            timeout_ms=base.timeout_ms,
            runtime_options=base.runtime_options,
            metadata=base.metadata,
            thinking_intent=thinking_intent,
            studio_resolved_thinking=ThinkingIntent.from_payload(studio_payload)
            if studio_payload is not None
            else None,
            execution_guard=guard,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "contract_version": CONTRACT_V2_VERSION,
                "schema_version": GENERATION_REQUEST_V2_VERSION,
                "thinking_intent": self.thinking_intent.to_dict(),
                "studio_resolved_thinking": self.studio_resolved_thinking.to_dict()
                if self.studio_resolved_thinking
                else None,
                "execution_guard": self.execution_guard.to_dict() if self.execution_guard else None,
            }
        )
        if self.execution_guard:
            payload["expected_execution_binding_fingerprint"] = (
                self.execution_guard.expected_execution_binding_fingerprint
            )
            payload["expected_runtime_settings_fingerprint"] = (
                self.execution_guard.expected_runtime_settings_fingerprint
            )
        return payload


@dataclass
class ExecutionTraceV2:
    execution_id: str
    request_id: str
    request_contract_version: str
    execution_binding: ExecutionBindingV2
    thinking_resolution: ThinkingResolution
    started_at: str
    guard: ExecutionGuardV1 | None = None
    finished_at: str | None = None
    status: str = "running"
    metrics: dict[str, Any] | None = None
    finish_reason: str | None = None
    raw_execution_error: dict[str, Any] | None = None

    @property
    def execution_binding_fingerprint(self) -> str | None:
        return (
            self.execution_binding.fingerprint if self.execution_binding.runtime_settings_binding is not None else None
        )

    @property
    def artifact_binding(self) -> ArtifactBindingV2:
        return self.execution_binding.artifact_binding

    @property
    def engine_binding(self) -> EngineBindingV2:
        return self.execution_binding.engine_binding

    @property
    def foundation_binding(self) -> FoundationBindingV2:
        return self.execution_binding.foundation_binding

    @property
    def runtime_settings_binding(self) -> RuntimeSettingsBindingV2 | None:
        return self.execution_binding.runtime_settings_binding

    def with_runtime_settings(self, options: RuntimeOptions) -> None:
        self.execution_binding = ExecutionBindingV2(
            artifact_binding=self.execution_binding.artifact_binding,
            engine_binding=self.execution_binding.engine_binding,
            foundation_binding=self.execution_binding.foundation_binding,
            runtime_settings_binding=RuntimeSettingsBindingV2.from_effective(options),
        )

    def to_dict(self) -> dict[str, Any]:
        binding = self.execution_binding.to_dict()
        return {
            "contract_version": CONTRACT_V2_VERSION,
            "schema_version": EXECUTION_TRACE_V2_VERSION,
            "execution_id": self.execution_id,
            "request_id": self.request_id,
            "request_contract_version": self.request_contract_version,
            "execution_binding": binding,
            "execution_binding_fingerprint": self.execution_binding_fingerprint,
            "artifact_binding": self.artifact_binding.to_dict(),
            "engine_binding": self.engine_binding.to_dict(),
            "foundation_binding": self.foundation_binding.to_dict(),
            "effective_runtime_settings_identity": self.runtime_settings_binding.to_dict()
            if self.runtime_settings_binding
            else None,
            "thinking_requested": self.thinking_resolution.requested.to_dict(),
            "thinking_studio_resolved": self.thinking_resolution.studio_resolved.to_dict()
            if self.thinking_resolution.studio_resolved
            else None,
            "thinking_foundation_effective": self.thinking_resolution.foundation_effective.to_dict()
            if self.thinking_resolution.foundation_effective
            else None,
            "thinking_resolution": self.thinking_resolution.to_dict(),
            "execution_guard": self.guard.to_dict() if self.guard else None,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "metrics": self.metrics,
            "finish_reason": self.finish_reason,
            "raw_execution_error": self.raw_execution_error,
        }


@dataclass
class GenerationResultV2:
    """v2 result envelope derived from a v1 result without rewriting it."""

    result: GenerationResult
    trace: ExecutionTraceV2

    @classmethod
    def from_result(cls, result: GenerationResult, trace: ExecutionTraceV2) -> GenerationResultV2:
        return cls(result=result, trace=trace)

    def to_dict(self) -> dict[str, Any]:
        payload = self.result.to_dict(include_v2=False)
        payload.update(
            {
                "contract_version": CONTRACT_V2_VERSION,
                "schema_version": GENERATION_RESULT_V2_VERSION,
                "execution_binding": self.trace.execution_binding.to_dict(),
                "execution_binding_fingerprint": self.trace.execution_binding_fingerprint,
                "trace": self.trace.to_dict(),
                "trace_v1": self.result.trace.to_dict() if self.result.trace else None,
            }
        )
        return payload


def build_execution_binding(
    *,
    artifact: ModelArtifactBinding,
    engine: Any,
    adapter_id: str,
    foundation_version: str,
    foundation_build_identity: str | None,
    effective_runtime_options: RuntimeOptions | None,
) -> ExecutionBindingV2:
    return ExecutionBindingV2(
        artifact_binding=ArtifactBindingV2.from_legacy(artifact),
        engine_binding=EngineBindingV2.from_legacy(engine, adapter_id=adapter_id),
        foundation_binding=FoundationBindingV2(
            contract_version=CONTRACT_V2_VERSION,
            foundation_version=foundation_version,
            build_identity=foundation_build_identity,
            adapter_id=adapter_id,
        ),
        runtime_settings_binding=RuntimeSettingsBindingV2.from_effective(effective_runtime_options),
    )


__all__ = [
    "CONTRACT_V2_VERSION",
    "EXECUTION_BINDING_VERSION",
    "EXECUTION_GUARD_VERSION",
    "EXECUTION_TRACE_V2_VERSION",
    "GENERATION_REQUEST_V2_VERSION",
    "GENERATION_RESULT_V2_VERSION",
    "SUPPORTED_CONTRACT_VERSIONS",
    "ArtifactBindingV2",
    "ArtifactLocator",
    "ContentIdentity",
    "ContentIdentityScheme",
    "EngineBindingV2",
    "EngineImplementationBinding",
    "ExecutionBindingV2",
    "ExecutionGuard",
    "ExecutionGuardV1",
    "ExecutionTraceV2",
    "FoundationBindingV2",
    "GenerationRequestV2",
    "GenerationResultV2",
    "RuntimeSettingsBindingV2",
    "ThinkingEffort",
    "ThinkingIntent",
    "ThinkingMode",
    "ThinkingResolution",
    "build_execution_binding",
    "canonical_fingerprint",
    "canonical_json",
]
