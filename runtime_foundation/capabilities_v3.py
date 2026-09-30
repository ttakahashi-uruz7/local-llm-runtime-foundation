"""Versioned, provenance-aware engine option capability contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .contracts import EngineCapability
from .contracts_v2 import ArtifactBindingV2, EngineBindingV2, canonical_fingerprint
from .contracts_v3 import CONTRACT_V3_VERSION

CAPABILITY_V3_VERSION = "runtime-foundation.engine-capability.v3"


class CapabilityStatus(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


class CapabilityScope(str, Enum):
    LOAD = "LOAD"
    GENERATION = "GENERATION"
    CONSTRAINT = "CONSTRAINT"


class CapabilityDependency(str, Enum):
    ARTIFACT = "artifact"
    ENGINE = "engine"
    ENGINE_BUILD = "engine_build"
    HOST = "host"
    LOAD = "load"


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


@dataclass(frozen=True)
class CapabilityApplicabilityV3:
    depends_on: tuple[CapabilityDependency, ...] = (
        CapabilityDependency.ENGINE,
        CapabilityDependency.ENGINE_BUILD,
        CapabilityDependency.HOST,
    )
    artifact_identity: dict[str, Any] | None = None
    engine_identity: dict[str, Any] | None = None
    engine_build_identity: dict[str, Any] | None = None
    host_fingerprint: str | None = None
    load_identity_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.depends_on, (tuple, list)) or any(
            not isinstance(item, CapabilityDependency) for item in self.depends_on
        ):
            raise ValueError("capability applicability depends_on contains an invalid dependency")
        normalized = tuple(dict.fromkeys(self.depends_on))
        object.__setattr__(self, "depends_on", normalized)
        for name in ("artifact_identity", "engine_identity", "engine_build_identity"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, dict):
                raise ValueError(f"capability applicability {name} must be an object")

    def to_dict(self) -> dict[str, Any]:
        return {
            "depends_on": [item.value for item in self.depends_on],
            "artifact_identity": self.artifact_identity,
            "engine_identity": self.engine_identity,
            "engine_build_identity": self.engine_build_identity,
            "host_fingerprint": self.host_fingerprint,
            "load_identity_fingerprint": self.load_identity_fingerprint,
        }


@dataclass(frozen=True)
class OptionCapabilityV3:
    status: CapabilityStatus
    scope: CapabilityScope
    requires_reload: bool = False
    allowed_values: tuple[Any, ...] = ()
    minimum: float | int | None = None
    maximum: float | int | None = None
    unit: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    applicability: CapabilityApplicabilityV3 = field(default_factory=CapabilityApplicabilityV3)

    def __post_init__(self) -> None:
        if not isinstance(self.status, CapabilityStatus):
            raise ValueError("option capability status is invalid")
        if not isinstance(self.scope, CapabilityScope):
            raise ValueError("option capability scope is invalid")
        if not isinstance(self.requires_reload, bool):
            raise ValueError("option capability requires_reload must be boolean")
        if self.scope == CapabilityScope.LOAD and not self.requires_reload:
            object.__setattr__(self, "requires_reload", True)
        if self.scope != CapabilityScope.LOAD and self.requires_reload:
            raise ValueError("only LOAD-scoped capabilities can require reload")
        if self.minimum is not None and (isinstance(self.minimum, bool) or not isinstance(self.minimum, (int, float))):
            raise ValueError("option capability minimum must be numeric")
        if self.maximum is not None and (isinstance(self.maximum, bool) or not isinstance(self.maximum, (int, float))):
            raise ValueError("option capability maximum must be numeric")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("option capability minimum must be <= maximum")
        if self.unit is not None and (not isinstance(self.unit, str) or not self.unit.strip()):
            raise ValueError("option capability unit must be a non-empty string")
        if not isinstance(self.evidence, dict):
            raise ValueError("option capability evidence must be an object")
        if self.reason is not None and (not isinstance(self.reason, str) or not self.reason.strip()):
            raise ValueError("option capability reason must be a non-empty string")
        if not isinstance(self.applicability, CapabilityApplicabilityV3):
            raise ValueError("option capability applicability must be validated")

    @classmethod
    def from_legacy(
        cls,
        *,
        value: dict[str, Any] | None,
        scope: CapabilityScope,
        engine_available: bool,
        reason: str | None,
        applicability: CapabilityApplicabilityV3,
    ) -> OptionCapabilityV3:
        item = value or {}
        reported = item.get("status")
        status = {
            "supported": CapabilityStatus.SUPPORTED if engine_available else CapabilityStatus.UNAVAILABLE,
            "unsupported": CapabilityStatus.UNSUPPORTED,
            "unavailable": CapabilityStatus.UNAVAILABLE,
            "unknown": CapabilityStatus.UNKNOWN,
        }.get(reported, CapabilityStatus.UNAVAILABLE if not engine_available else CapabilityStatus.UNKNOWN)
        allowed = item.get("supported_values", item.get("allowed_values", ()))
        if not isinstance(allowed, (tuple, list)):
            allowed = ()
        evidence = {key: _plain(val) for key, val in item.items() if key not in {
            "status", "reason", "minimum", "maximum", "unit", "supported_values", "allowed_values", "requires_reload"
        }}
        return cls(
            status=status,
            scope=scope,
            requires_reload=bool(item.get("requires_reload", scope == CapabilityScope.LOAD)),
            allowed_values=tuple(allowed),
            minimum=item.get("minimum", item.get("min")),
            maximum=item.get("maximum", item.get("max")),
            unit=item.get("unit"),
            evidence=evidence,
            reason=item.get("reason", reason if status != CapabilityStatus.UNKNOWN else "legacy adapter did not report this option"),
            applicability=applicability,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "scope": self.scope.value,
            "requires_reload": self.requires_reload,
            "allowed_values": [_plain(value) for value in self.allowed_values],
            "range": {"minimum": self.minimum, "maximum": self.maximum, "unit": self.unit},
            "evidence": dict(self.evidence),
            "reason": self.reason,
            "applicability": self.applicability.to_dict(),
        }


@dataclass(frozen=True)
class EngineCapabilityV3:
    engine: EngineBindingV2
    status: CapabilityStatus
    options: dict[str, OptionCapabilityV3]
    streaming: CapabilityStatus
    cancellation: CapabilityStatus
    load_unload: CapabilityStatus
    chat_template: OptionCapabilityV3
    thinking_supported: OptionCapabilityV3
    thinking_effort_supported: OptionCapabilityV3
    thinking_budget_supported: OptionCapabilityV3
    reason: str | None = None
    schema_version: str = CAPABILITY_V3_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CAPABILITY_V3_VERSION:
            raise ValueError(f"engine capability schema_version must be {CAPABILITY_V3_VERSION}")
        if not isinstance(self.engine, EngineBindingV2):
            raise ValueError("engine capability requires a validated EngineBindingV2")
        for name in ("status", "streaming", "cancellation", "load_unload"):
            if not isinstance(getattr(self, name), CapabilityStatus):
                raise ValueError(f"engine capability {name} is invalid")
        if not isinstance(self.options, dict) or any(
            not isinstance(value, OptionCapabilityV3) for value in self.options.values()
        ):
            raise ValueError("engine capability options must map names to OptionCapabilityV3")
        for value in (
            self.chat_template,
            self.thinking_supported,
            self.thinking_effort_supported,
            self.thinking_budget_supported,
        ):
            if not isinstance(value, OptionCapabilityV3):
                raise ValueError("engine capability feature fields must be OptionCapabilityV3")

    @classmethod
    def from_legacy(
        cls,
        legacy: EngineCapability,
        *,
        engine_binding: EngineBindingV2,
        host_observation: dict[str, Any] | None = None,
        artifact: ArtifactBindingV2 | None = None,
        load_identity_fingerprint: str | None = None,
    ) -> EngineCapabilityV3:
        host_fingerprint = canonical_fingerprint(host_observation) if host_observation is not None else None
        applicability = CapabilityApplicabilityV3(
            depends_on=(
                CapabilityDependency.ARTIFACT,
                CapabilityDependency.ENGINE,
                CapabilityDependency.ENGINE_BUILD,
                CapabilityDependency.HOST,
                CapabilityDependency.LOAD,
            ),
            artifact_identity=artifact.canonical_payload() if artifact else None,
            engine_identity={
                "family": engine_binding.family,
                "implementation": engine_binding.implementation.to_dict(),
                "adapter_id": engine_binding.adapter_id,
            },
            engine_build_identity=engine_binding.build_identity.to_dict()
            if engine_binding.build_identity
            else None,
            host_fingerprint=host_fingerprint,
            load_identity_fingerprint=load_identity_fingerprint,
        )
        available = legacy.available
        option_map: dict[str, OptionCapabilityV3] = {}
        for name, value in legacy.runtime_options.items():
            option_map[name] = OptionCapabilityV3.from_legacy(
                value=value,
                scope=CapabilityScope.LOAD,
                engine_available=available,
                reason=legacy.reason,
                applicability=applicability,
            )
        for name, value in legacy.generation_options.items():
            option_map[name] = OptionCapabilityV3.from_legacy(
                value=value,
                scope=CapabilityScope.GENERATION,
                engine_available=available,
                reason=legacy.reason,
                applicability=applicability,
            )

        def feature(value: str, scope: CapabilityScope) -> OptionCapabilityV3:
            legacy_status = value.lower()
            if legacy_status in {"unavailable", "unknown", "unsupported"}:
                item = {"status": legacy_status}
            elif not available:
                item = {"status": "unavailable"}
            elif legacy_status in {"none", "false"}:
                item = {"status": "unsupported"}
            else:
                item = {"status": "supported", "evidence": {"legacy_declaration": value}}
            return OptionCapabilityV3.from_legacy(
                value=item,
                scope=scope,
                engine_available=available,
                reason=legacy.reason,
                applicability=applicability,
            )

        status = CapabilityStatus.SUPPORTED if available else CapabilityStatus.UNAVAILABLE
        return cls(
            engine=engine_binding,
            status=status,
            options=option_map,
            streaming=CapabilityStatus.SUPPORTED if available and legacy.streaming else (
                CapabilityStatus.UNAVAILABLE if not available else CapabilityStatus.UNSUPPORTED
            ),
            cancellation=CapabilityStatus.SUPPORTED if available and legacy.cancellation else (
                CapabilityStatus.UNAVAILABLE if not available else CapabilityStatus.UNSUPPORTED
            ),
            load_unload=CapabilityStatus.SUPPORTED if available and legacy.load_unload else (
                CapabilityStatus.UNAVAILABLE if not available else CapabilityStatus.UNSUPPORTED
            ),
            chat_template=feature(legacy.chat_template, CapabilityScope.GENERATION),
            thinking_supported=feature(legacy.thinking_flag, CapabilityScope.GENERATION),
            thinking_effort_supported=feature(
                "supported" if legacy.generation_options.get("thinking_effort", {}).get("status") == "supported" else "unknown",
                CapabilityScope.GENERATION,
            ),
            thinking_budget_supported=feature(
                "supported" if legacy.generation_options.get("thinking_budget_tokens", {}).get("status") == "supported" else "unknown",
                CapabilityScope.GENERATION,
            ),
            reason=legacy.reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_V3_VERSION,
            "schema_version": self.schema_version,
            "engine": self.engine.to_dict(),
            "status": self.status.value,
            "streaming": self.streaming.value,
            "cancellation": self.cancellation.value,
            "load_unload": self.load_unload.value,
            "options": {key: self.options[key].to_dict() for key in sorted(self.options)},
            "features": {
                "chat_template": self.chat_template.to_dict(),
                "thinking_supported": self.thinking_supported.to_dict(),
                "thinking_effort_supported": self.thinking_effort_supported.to_dict(),
                "thinking_budget_supported": self.thinking_budget_supported.to_dict(),
            },
            "reason": self.reason,
        }


@dataclass(frozen=True)
class HostExecutionCapabilitiesV3:
    """Separate host, MLX, llama.cpp-build, and selected-device observations."""

    host_apple_silicon: CapabilityStatus
    host_metal: CapabilityStatus
    mlx_availability: CapabilityStatus
    mlx_metal: CapabilityStatus
    llama_cpp_availability: CapabilityStatus
    llama_cpp_build_metal: CapabilityStatus
    selected_execution_acceleration: dict[str, Any] | None = None
    evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    schema_version: str = "runtime-foundation.host-execution-capabilities.v3"

    @classmethod
    def observe(
        cls,
        *,
        platform: str,
        architecture: str,
        host_metal: bool | None,
        host_metal_reason: str,
        mlx_available: bool,
        mlx_default_device_metal: bool,
        llama_cpp_capability: EngineCapabilityV3,
        selected_execution_acceleration: dict[str, Any] | None = None,
    ) -> HostExecutionCapabilitiesV3:
        apple_silicon = platform == "darwin" and architecture == "arm64"
        llama_status = llama_cpp_capability.status
        metal_option = llama_cpp_capability.options.get("acceleration.backend")
        metal_capability_status = (
            metal_option.evidence.get("metal_capability_status") if metal_option is not None else None
        )
        if llama_status == CapabilityStatus.UNAVAILABLE:
            llama_metal = CapabilityStatus.UNAVAILABLE
            llama_metal_reason = "llama.cpp engine is unavailable"
        elif metal_capability_status == CapabilityStatus.UNKNOWN.value:
            llama_metal = CapabilityStatus.UNKNOWN
            llama_metal_reason = metal_option.reason if metal_option is not None else "Metal capability is unknown"
        elif metal_capability_status == CapabilityStatus.UNAVAILABLE.value:
            llama_metal = CapabilityStatus.UNAVAILABLE
            llama_metal_reason = metal_option.reason if metal_option is not None else "Metal capability is unavailable"
        elif metal_capability_status == CapabilityStatus.UNSUPPORTED.value:
            llama_metal = CapabilityStatus.UNSUPPORTED
            llama_metal_reason = metal_option.reason if metal_option is not None else "Metal capability is unsupported"
        elif metal_option is None or metal_option.status == CapabilityStatus.UNKNOWN:
            llama_metal = CapabilityStatus.UNKNOWN
            llama_metal_reason = "the selected llama.cpp build did not report Metal capability"
        elif metal_option.status == CapabilityStatus.UNAVAILABLE:
            llama_metal = CapabilityStatus.UNAVAILABLE
            llama_metal_reason = metal_option.reason or "llama.cpp Metal capability is unavailable"
        elif metal_option.status == CapabilityStatus.UNSUPPORTED or "metal" not in {
            str(value).lower() for value in metal_option.allowed_values
        }:
            llama_metal = CapabilityStatus.UNSUPPORTED
            llama_metal_reason = metal_option.reason or "the selected llama.cpp build does not declare Metal support"
        else:
            llama_metal = CapabilityStatus.SUPPORTED
            llama_metal_reason = metal_option.reason or "the selected llama.cpp build declares Metal support"
        host_metal_status = (
            CapabilityStatus.UNKNOWN
            if host_metal is None
            else CapabilityStatus.SUPPORTED if host_metal else CapabilityStatus.UNSUPPORTED
        )
        if not mlx_available:
            mlx_status = CapabilityStatus.UNAVAILABLE
            mlx_metal_status = CapabilityStatus.UNAVAILABLE
            mlx_reason = "mlx and mlx-lm are not both available"
        else:
            mlx_status = CapabilityStatus.SUPPORTED
            mlx_metal_status = CapabilityStatus.SUPPORTED if mlx_default_device_metal else CapabilityStatus.UNSUPPORTED
            mlx_reason = "derived from the existing MLX default-device observation"
        return cls(
            host_apple_silicon=CapabilityStatus.SUPPORTED if apple_silicon else CapabilityStatus.UNSUPPORTED,
            host_metal=host_metal_status,
            mlx_availability=mlx_status,
            mlx_metal=mlx_metal_status,
            llama_cpp_availability=llama_status,
            llama_cpp_build_metal=llama_metal,
            selected_execution_acceleration=selected_execution_acceleration,
            evidence={
                "host_apple_silicon": {"platform": platform, "architecture": architecture},
                "host_metal": {"reason": host_metal_reason},
                "mlx": {"reason": mlx_reason, "legacy_default_device_metal": mlx_default_device_metal},
                "llama_cpp": {"reason": llama_metal_reason},
                "selected_execution_acceleration": selected_execution_acceleration or {"status": "no_active_load"},
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_V3_VERSION,
            "schema_version": self.schema_version,
            "host_apple_silicon": self.host_apple_silicon.value,
            "host_metal": self.host_metal.value,
            "mlx_availability": self.mlx_availability.value,
            "mlx_metal": self.mlx_metal.value,
            "llama_cpp_availability": self.llama_cpp_availability.value,
            "llama_cpp_build_metal": self.llama_cpp_build_metal.value,
            "selected_execution_acceleration": self.selected_execution_acceleration,
            "evidence": {key: dict(value) for key, value in self.evidence.items()},
        }
