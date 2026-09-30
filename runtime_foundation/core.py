"""Policy-free runtime lifecycle and execution core."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from .adapters import EngineAdapter, LlamaCppAdapter, MLXAdapter, MockAdapter
from .adapters.base import PreparedGenerationV3
from .contracts import (
    CONTRACT_VERSION,
    EngineCapability,
    EngineIdentity,
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
    BuildIdentityV1,
    CONTRACT_V2_VERSION,
    SUPPORTED_CONTRACT_VERSIONS,
    ArtifactBindingV2,
    ContentIdentity,
    ExecutionInputV1,
    ExecutionGuardVerification,
    ExecutionTraceV2,
    EngineBindingV2,
    FoundationBindingV2,
    GenerationRequestV2,
    GenerationResultV2,
    ThinkingIntent,
    ThinkingMode,
    ThinkingResolution,
    build_execution_binding,
    is_valid_fingerprint,
)
from .contracts_v3 import (
    CONTRACT_V3_VERSION,
    SUPPORTED_CONTRACT_VERSIONS_V3,
    ExecutionBindingV3,
    ExecutionEvidenceV3,
    ExecutionOptionsEvidenceV3,
    ExecutionTraceV3,
    GenerationOptions,
    GenerationRequestV3,
    GenerationResultV3,
    LoadIdentityV1,
    LoadOptions,
    LoadOptionsResolutionV1,
    OptionResolutionV3,
    SettingsEvidenceV3,
    StreamEventV3,
)
from .build_identity import observe_foundation_build_identity
from .capabilities_v3 import EngineCapabilityV3, HostExecutionCapabilitiesV3
from .errors import (
    ArtifactCompatibilityError,
    ArtifactNotFoundError,
    ContextLengthExceededError,
    EngineNotFoundError,
    EngineRuntimeError,
    EngineUnavailableError,
    ExecutionBindingUnresolvableError,
    ExecutionGuardMismatchError,
    ExecutionInputMismatchError,
    InvalidRequestError,
    LoadConflictError,
    ModelNotLoadedError,
    RuntimeBusyError,
    RuntimeFoundationError,
    RuntimeTimeoutError,
    ThinkingResolutionError,
    UnloadConflictError,
    UnsupportedArtifactLocatorError,
    UnsupportedGenerationSettingError,
    error_payload,
)
from .engines import default_engine_for_artifact_format, engine_wire_identifier, normalize_engine_identifier
from .host import HostProfile, host_metal_capability
from .gguf import VerifiedGGUFArtifact
from .version import FOUNDATION_VERSION

DEFAULT_CONSUMER_ID = "anonymous"


def _build_identity_payload(identity: Any | None) -> dict[str, Any] | None:
    if identity is None:
        return None
    if hasattr(identity, "to_dict"):
        payload = identity.to_dict()
        return payload if isinstance(payload, dict) else None
    return dict(identity) if isinstance(identity, dict) else None


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

    def __init__(
        self,
        *,
        host_profile: HostProfile | None = None,
        adapters: dict[str, EngineAdapter] | None = None,
        foundation_version: str = FOUNDATION_VERSION,
        foundation_build_identity: BuildIdentityV1 | dict[str, Any] | None = None,
    ) -> None:
        self.foundation_version = foundation_version
        if foundation_build_identity is None:
            self.foundation_build_identity = observe_foundation_build_identity()
        elif isinstance(foundation_build_identity, BuildIdentityV1):
            self.foundation_build_identity = foundation_build_identity
        elif isinstance(foundation_build_identity, dict):
            self.foundation_build_identity = BuildIdentityV1.from_payload(foundation_build_identity)
        else:
            raise ValueError("foundation_build_identity must be a validated BuildIdentityV1 or payload")
        self.host_profile = host_profile or HostProfile.detect()
        adapter_source: dict[str, EngineAdapter] = adapters or {
            "mock": MockAdapter(),
            "mlx": MLXAdapter(),
            "llama.cpp": LlamaCppAdapter(host_profile=self.host_profile),
        }
        self.adapters = {}
        for identifier, adapter in adapter_source.items():
            canonical = normalize_engine_identifier(identifier)
            if canonical in self.adapters:
                raise ValueError(f"multiple adapters resolve to the same engine identifier: {canonical}")
            self.adapters[canonical] = adapter
        self._lock = threading.RLock()
        self._state = LifecycleState.UNLOADED
        self._loaded_artifact: ModelArtifactBinding | None = None
        self._loaded_artifact_v2: ArtifactBindingV2 | None = None
        self._loaded_execution_input: ExecutionInputV1 | None = None
        self._loaded_load_options: LoadOptions | None = None
        self._loaded_load_options_resolution: LoadOptionsResolutionV1 | None = None
        self._loaded_load_identity: LoadIdentityV1 | None = None
        self._loaded_adapter: EngineAdapter | None = None
        self._leases: dict[str, str] = {}
        self._active: dict[str, _ActiveRequest] = {}
        self._traces: dict[str, ExecutionTrace] = {}
        self._v3_traces: dict[str, ExecutionTraceV3] = {}
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
    def _artifact(
        value: ModelArtifactBinding | ArtifactBindingV2 | dict[str, Any],
    ) -> tuple[ArtifactBindingV2, ModelArtifactBinding]:
        if isinstance(value, ModelArtifactBinding):
            v2 = ArtifactBindingV2.from_legacy(value)
        elif isinstance(value, ArtifactBindingV2):
            v2 = value
        elif isinstance(value, dict) and (
            value.get("contract_version") == CONTRACT_V2_VERSION
            or "registry_identity" in value
            or "content_identity" in value
            or "locator" in value
        ):
            v2 = ArtifactBindingV2.from_payload(value)
        else:
            v2 = ArtifactBindingV2.from_legacy(ModelArtifactBinding.from_payload(value))
        if v2.locator.type != "filesystem":
            raise UnsupportedArtifactLocatorError(
                "the current Foundation execution boundary supports filesystem locators only",
                details={"locator_type": v2.locator.type, "artifact_id": v2.artifact_id},
            )
        return v2, v2.to_legacy()

    def _select_adapter(self, artifact: ModelArtifactBinding, requested: str | None) -> EngineAdapter:
        engine_name = normalize_engine_identifier(requested) if isinstance(requested, str) and requested.strip() else None
        if engine_name:
            try:
                selected = self.adapters[engine_name]
            except KeyError as exc:
                raise EngineNotFoundError(
                    "requested engine is not registered", details={"engine": engine_name}
                ) from exc
        else:
            family = default_engine_for_artifact_format(artifact.format)
            if family is None:
                raise EngineNotFoundError(
                    "no engine is registered for the artifact format; specify a supported engine explicitly",
                    details={"format": artifact.format, "artifact_id": artifact.artifact_id},
                )
            engine_name = engine_wire_identifier(family)
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
        supported_formats = {
            item.strip().lower()
            for item in capability.artifact_formats
            if isinstance(item, str) and item.strip()
        }
        if artifact.format.strip().lower() not in supported_formats:
            raise ArtifactCompatibilityError(
                "the selected engine does not declare compatibility with the artifact format",
                details={
                    "engine": selected.name,
                    "format": artifact.format,
                    "supported_formats": sorted(supported_formats),
                    "compatibility_status": "unsupported",
                },
            )
        return selected

    def capabilities(self) -> list[dict[str, Any]]:
        return [self.adapters[name].discover_capability().to_dict() for name in sorted(self.adapters)]

    def capabilities_v3(self) -> list[dict[str, Any]]:
        host_observation = self.host_profile.to_dict()
        return [
            self.adapters[name].discover_capability_v3(host_observation=host_observation).to_dict()
            for name in sorted(self.adapters)
        ]

    def capability(self, engine: str) -> dict[str, Any]:
        try:
            adapter = self.adapters[engine]
        except KeyError as exc:
            raise EngineNotFoundError("requested engine is not registered", details={"engine": engine}) from exc
        return adapter.discover_capability().to_dict()

    def capability_v3(self, engine: str) -> dict[str, Any]:
        normalized = normalize_engine_identifier(engine)
        try:
            adapter = self.adapters[normalized]
        except KeyError as exc:
            raise EngineNotFoundError("requested engine is not registered", details={"engine": normalized}) from exc
        return adapter.discover_capability_v3(host_observation=self.host_profile.to_dict()).to_dict()

    def host(self) -> dict[str, Any]:
        return self.host_profile.to_dict()

    def host_v3(self) -> dict[str, Any]:
        host_metal, host_metal_reason = host_metal_capability(self.host_profile.platform)
        llama_adapter = self.adapters.get("llama.cpp")
        if llama_adapter is None:
            llama_capability = EngineCapabilityV3.from_legacy(
                EngineCapability(
                    identity=EngineIdentity("llama.cpp"),
                    available=False,
                    streaming=False,
                    cancellation=False,
                    load_unload=False,
                    reason="llama.cpp adapter is not registered",
                ),
                engine_binding=EngineBindingV2.from_legacy(EngineIdentity("llama.cpp"), adapter_id="llama.cpp"),
            )
        else:
            llama_capability = llama_adapter.discover_capability_v3(
                host_observation={
                    **self.host_profile.to_dict(),
                    "host_metal_capability": host_metal,
                    "host_metal_reason": host_metal_reason,
                }
            )
        selected_acceleration: dict[str, Any] | None = None
        with self._lock:
            loaded_adapter = self._loaded_adapter
        if loaded_adapter is not None:
            try:
                observed = loaded_adapter.health().get("selected_execution_acceleration")
            except Exception:  # noqa: BLE001 - host diagnostics must not mask runtime health
                observed = None
            selected_acceleration = observed or {"engine": loaded_adapter.name, "status": "unknown"}
        execution_capabilities = HostExecutionCapabilitiesV3.observe(
            platform=self.host_profile.platform,
            architecture=self.host_profile.architecture,
            host_metal=host_metal,
            host_metal_reason=host_metal_reason,
            mlx_available=self.host_profile.mlx_available,
            mlx_default_device_metal=self.host_profile.metal_available,
            llama_cpp_capability=llama_capability,
            selected_execution_acceleration=selected_acceleration,
        )
        return {
            "contract_version": CONTRACT_V3_VERSION,
            "supported_contract_versions": list(SUPPORTED_CONTRACT_VERSIONS_V3),
            "host": self.host_profile.to_dict(),
            "execution_capabilities": execution_capabilities.to_dict(),
        }

    def load(
        self,
        artifact: ModelArtifactBinding | ArtifactBindingV2 | ExecutionInputV1 | dict[str, Any] | None = None,
        *,
        execution_input: ExecutionInputV1 | dict[str, Any] | None = None,
        adapter: str | None = None,
        consumer_id: str | None = None,
        load_options: LoadOptions | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if artifact is not None and execution_input is not None:
            raise InvalidRequestError("provide either artifact or execution_input, not both")
        candidate = execution_input if execution_input is not None else artifact
        if candidate is None:
            raise InvalidRequestError("artifact or execution_input is required")
        options_were_explicit = load_options is not None
        if isinstance(load_options, LoadOptions):
            requested_load_options = LoadOptions.from_payload(load_options.to_dict())
        elif isinstance(load_options, dict):
            try:
                requested_load_options = LoadOptions.from_payload(load_options)
            except ValueError as exc:
                raise InvalidRequestError(str(exc)) from exc
        elif load_options is None:
            requested_load_options = LoadOptions()
        else:
            raise InvalidRequestError("load_options must be a validated LoadOptions or object payload")
        direct_input: ExecutionInputV1 | None
        if isinstance(candidate, ExecutionInputV1):
            direct_input = ExecutionInputV1.from_payload(candidate.to_dict())
        elif isinstance(candidate, dict) and candidate.get("schema_version") == "runtime-foundation.execution-input.v1":
            try:
                direct_input = ExecutionInputV1.from_payload(candidate)
            except ValueError as exc:
                raise InvalidRequestError(str(exc)) from exc
        else:
            direct_input = None
        if direct_input is not None:
            artifact_v2, binding = direct_input.base, direct_input.base.to_legacy()
        else:
            artifact_v2, binding = self._artifact(cast(Any, candidate))
        if not Path(binding.local_path).exists():
            raise ArtifactNotFoundError(
                "model artifact path does not exist",
                details={"role": "base", "artifact_id": binding.artifact_id, "local_path": binding.local_path},
            )
        if direct_input is not None:
            identities = [("base", direct_input.base, None)] + [
                ("adapter", item, index) for index, item in enumerate(direct_input.adapters)
            ]
            for role, identity_binding, index in identities:
                path = identity_binding.local_path
                if path is None or not Path(path).exists():
                    raise ArtifactNotFoundError(
                        f"{role} artifact path does not exist",
                        details={"role": role, "index": index, "artifact_id": identity_binding.artifact_id, "local_path": path},
                    )
                expected_identity = identity_binding.content_identity
                assert expected_identity is not None  # Enforced by ExecutionInputV1.
                actual_identity = ContentIdentity.from_file(
                    path,
                    scheme="complete",
                    scope=expected_identity.scope,
                )
                if actual_identity != expected_identity:
                    raise ArtifactCompatibilityError(
                        f"{role.capitalize()} content does not match the supplied complete content identity",
                        details={
                            "role": role,
                            "artifact_id": identity_binding.artifact_id,
                            "expected_content_identity": expected_identity.to_dict(),
                            "actual_content_identity": actual_identity.to_dict(),
                        },
                    )
        verified_gguf: VerifiedGGUFArtifact | None = None
        gguf_observation: dict[str, Any] | None = None
        if (binding.format or "").lower() == "gguf":
            try:
                verified_gguf = VerifiedGGUFArtifact.open(artifact_v2)
                gguf_observation = verified_gguf.observed.to_dict()
            except RuntimeFoundationError:
                raise
            except (OSError, ValueError) as exc:
                raise ArtifactCompatibilityError(
                    "GGUF artifact could not be validated",
                    details={"artifact_id": binding.artifact_id, "reason": str(exc)},
                ) from exc
            if direct_input is None:
                # The existing v2 ArtifactBinding remains the one identity
                # system. For GGUF, Core enriches it with observed content and
                # quantization before selection, load reuse, and evidence.
                artifact_v2 = replace(
                    artifact_v2,
                    content_identity=verified_gguf.observed.content_identity,
                    format="gguf",
                    quantization=verified_gguf.observed.quantization,
                )
                binding = artifact_v2.to_legacy()
        try:
            return self._load_validated(
                artifact_v2=artifact_v2,
                binding=binding,
                direct_input=direct_input,
                options_were_explicit=options_were_explicit,
                requested_load_options=requested_load_options,
                verified_gguf=verified_gguf,
                consumer_id=consumer_id,
                adapter=adapter,
            )
        finally:
            if verified_gguf is not None:
                verified_gguf.close()

    def _load_validated(
        self,
        *,
        artifact_v2: ArtifactBindingV2,
        binding: ModelArtifactBinding,
        direct_input: ExecutionInputV1 | None,
        options_were_explicit: bool,
        requested_load_options: LoadOptions,
        verified_gguf: VerifiedGGUFArtifact | None,
        consumer_id: str | None,
        adapter: str | None,
    ) -> dict[str, Any]:
        gguf_observation = verified_gguf.observed.to_dict() if verified_gguf is not None else None
        owner = self._consumer_id(consumer_id)
        selected: EngineAdapter
        try:
            selected = self._select_adapter(binding, adapter)
        except RuntimeFoundationError as exc:
            with self._lock:
                self._state = LifecycleState.ERROR
                self._last_error = exc.as_dict()
            raise
        assert selected is not None
        if verified_gguf is not None:
            load_options_resolution = selected.resolve_load_options_with_observation(
                binding,
                requested_load_options,
                verified_gguf.observed,
            )
        else:
            load_options_resolution = selected.resolve_load_options(binding, requested_load_options)
        if not isinstance(load_options_resolution, LoadOptionsResolutionV1):
            raise EngineRuntimeError(
                "engine adapter returned an invalid LoadOptions resolution",
                details={"engine": selected.name},
            )
        engine_binding = EngineBindingV2.from_legacy(
            selected.identity(),
            adapter_id=selected.name,
            build_identity=selected.build_identity(),
        )
        requested_load_identity = LoadIdentityV1.create(
            artifact=artifact_v2,
            engine=engine_binding,
            effective_load_options=load_options_resolution.effective,
        )
        with self._lock:
            loaded_adapter = self._loaded_adapter
            if self._loaded_artifact is not None and loaded_adapter is not None:
                same_direct_mode = (self._loaded_execution_input is None) == (direct_input is None)
                same_binding = (
                    self._loaded_execution_input.fingerprint == direct_input.fingerprint
                    if same_direct_mode and direct_input is not None and self._loaded_execution_input is not None
                    else same_direct_mode and self._loaded_artifact.execution_identity() == binding.execution_identity()
                )
                if (
                    same_binding
                    and loaded_adapter is selected
                    and self._loaded_load_identity is not None
                    and self._loaded_load_identity.fingerprint == requested_load_identity.fingerprint
                ):
                    lease_id = self._leases.setdefault(owner, new_id("lease"))
                    raw = {
                        "loaded_once": True,
                        "loaded_artifact_identity": self._loaded_artifact.execution_identity(),
                    }
                    if gguf_observation is not None:
                        raw["gguf_validation"] = gguf_observation
                    if options_were_explicit:
                        raw.update(
                            {
                                "load_identity": self._loaded_load_identity.to_dict(),
                                "load_options_resolution": load_options_resolution.to_dict(),
                            }
                        )
                    if self._loaded_execution_input is not None:
                        raw.update({
                            "execution_input": self._loaded_execution_input.to_dict(),
                            "execution_input_fingerprint": self._loaded_execution_input.fingerprint,
                        })
                    result = LoadResult(
                        artifact=self._loaded_artifact,
                        engine=loaded_adapter.identity(),
                        lifecycle_state=self._state,
                        lease_id=lease_id,
                        consumer_id=owner,
                        reused=True,
                        raw=raw,
                    )
                    return result.to_dict()
                raise LoadConflictError(
                    "the loaded execution identity differs; explicitly unload before changing artifact, engine, build, or LoadOptions",
                    details={
                        "loaded_artifact": self._loaded_artifact.to_dict(),
                        "requested_artifact": binding.to_dict(),
                        "loaded_execution_input_fingerprint": self._loaded_execution_input.fingerprint
                        if self._loaded_execution_input else None,
                        "requested_execution_input_fingerprint": direct_input.fingerprint if direct_input else None,
                        "loaded_load_identity_fingerprint": self._loaded_load_identity.fingerprint
                        if self._loaded_load_identity else None,
                        "requested_load_identity_fingerprint": requested_load_identity.fingerprint,
                        "requested_load_options": load_options_resolution.requested.to_dict(),
                        "resolved_load_options": load_options_resolution.resolved.to_dict(),
                        "effective_loaded_load_options": self._loaded_load_options.to_dict()
                        if self._loaded_load_options else None,
                        "active_request_ids": list(self._active),
                        "lease_owners": sorted(self._leases),
                    },
                )
            if self._state in {LifecycleState.LOADING, LifecycleState.UNLOADING, LifecycleState.GENERATING}:
                raise RuntimeBusyError("runtime lifecycle is busy", details={"lifecycle_state": self._state.value})
            self._state = LifecycleState.LOADING
            started = time.perf_counter()
            effective_load_options = load_options_resolution.effective
            try:
                if isinstance(selected, LlamaCppAdapter) and verified_gguf is not None:
                    raw = selected.load_verified_gguf(binding, load_options_resolution, verified_gguf)
                    if direct_input is not None:
                        raw.update(
                            {
                                "execution_input_kind": direct_input.kind,
                                "execution_input_fingerprint": direct_input.fingerprint,
                                "adapter_count": 0,
                            }
                        )
                elif options_were_explicit:
                    raw = (
                        selected.load_execution_input_with_options(direct_input, load_options_resolution)
                        if direct_input is not None
                        else selected.load_with_options(binding, load_options_resolution)
                    )
                else:
                    raw = selected.load_execution_input(direct_input) if direct_input is not None else selected.load(binding)
                if options_were_explicit and raw.get("effective_load_options") is not None:
                    reported_effective = LoadOptions.from_payload(raw["effective_load_options"])
                    raw_resolutions = raw.get("load_option_resolutions", [])
                    if not isinstance(raw_resolutions, (tuple, list)):
                        raise ValueError("load_option_resolutions must be an array")
                    resolution_records = tuple(
                        item if isinstance(item, OptionResolutionV3) else OptionResolutionV3(**item)
                        for item in raw_resolutions
                    )
                    load_options_resolution = LoadOptionsResolutionV1(
                        requested=load_options_resolution.requested,
                        resolved=load_options_resolution.resolved,
                        effective=reported_effective,
                        resolutions=load_options_resolution.resolutions + resolution_records,
                    )
                    effective_load_options = reported_effective
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
            self._loaded_artifact_v2 = artifact_v2
            self._loaded_execution_input = direct_input
            self._loaded_adapter = selected
            self._loaded_load_options = effective_load_options
            self._loaded_load_options_resolution = load_options_resolution
            self._loaded_load_identity = LoadIdentityV1.create(
                artifact=artifact_v2,
                engine=engine_binding,
                effective_load_options=effective_load_options,
            )
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
                raw={
                    **raw,
                    "load_duration_ms": self._last_load_duration_ms,
                    **({"gguf_validation": gguf_observation} if gguf_observation is not None else {}),
                    **({
                        "execution_input": direct_input.to_dict(),
                        "execution_input_fingerprint": direct_input.fingerprint,
                    } if direct_input is not None else {}),
                },
            )
            if options_were_explicit:
                result.raw.update(
                    {
                        "load_identity": self._loaded_load_identity.to_dict(),
                        "load_options_resolution": load_options_resolution.to_dict(),
                    }
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
            self._loaded_artifact_v2 = None
            self._loaded_execution_input = None
            self._loaded_adapter = None
            self._loaded_load_options = None
            self._loaded_load_options_resolution = None
            self._loaded_load_identity = None
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
        request_input = request.execution_input if isinstance(request, GenerationRequestV2) else None
        if request_input is not None:
            request_input = ExecutionInputV1.from_payload(request_input.to_dict())
        if self._loaded_execution_input is not None:
            if request_input is None or request_input.fingerprint != self._loaded_execution_input.fingerprint:
                raise ExecutionInputMismatchError(
                    "generation execution input does not match the loaded Base/Adapter composition",
                    details={
                        "loaded_execution_input_fingerprint": self._loaded_execution_input.fingerprint,
                        "requested_execution_input_fingerprint": request_input.fingerprint if request_input else None,
                        "generation_started": False,
                    },
                )
        elif request_input is not None:
            raise ExecutionInputMismatchError(
                "a direct Base/Adapter execution input was supplied for a single-artifact load",
                details={
                    "loaded_artifact_id": loaded.artifact_id,
                    "requested_execution_input_fingerprint": request_input.fingerprint,
                    "generation_started": False,
                },
            )
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
            loaded_v2 = self._loaded_artifact_v2 or ArtifactBindingV2.from_legacy(loaded)
            trace_v2 = self._make_v2_trace(request, loaded_v2, selected, execution_id, trace.started_at)
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
        artifact: ArtifactBindingV2,
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
            engine_build_identity=selected.build_identity(),
            foundation_version=self.foundation_version,
            foundation_build_identity=self.foundation_build_identity,
            effective_runtime_options=None,
            execution_input=self._loaded_execution_input,
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

        if request.thinking_intent.mode == ThinkingMode.AUTO:
            studio_resolved = request.studio_resolved_thinking
            if studio_resolved is None or studio_resolved.mode not in {ThinkingMode.OFF, ThinkingMode.ON}:
                resolution_error = ThinkingResolutionError(
                    "AUTO thinking requests require an explicit Studio ON/OFF resolution",
                    details={
                        "field": "studio_resolved_thinking",
                        "requested": request.thinking_intent.to_dict(),
                        "studio_resolved": studio_resolved.to_dict() if studio_resolved else None,
                        "resolution": "studio_authority_required",
                    },
                )
                trace_v2.thinking_resolution = ThinkingResolution(
                    requested=request.thinking_intent,
                    studio_resolved=studio_resolved,
                    foundation_effective=None,
                    status="unresolved",
                    reason="Foundation does not resolve AUTO; Studio must provide ON or OFF",
                    error=resolution_error.as_dict(),
                )
                raise resolution_error
            resolved = studio_resolved
        else:
            # ON/OFF are already resolved by the caller; Studio resolution is
            # not required and Foundation does not reinterpret the request.
            resolved = request.thinking_intent
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
            unsupported_error = UnsupportedGenerationSettingError(
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
                error=unsupported_error.as_dict(),
            )
            raise unsupported_error
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

    @staticmethod
    def _guard_details(
        request: GenerationRequestV2,
        trace_v2: ExecutionTraceV2,
        *,
        mismatch_category: str,
    ) -> dict[str, Any]:
        guard = trace_v2.guard
        assert guard is not None
        return {
            "expected_execution_binding_fingerprint": guard.expected_execution_binding_fingerprint,
            "actual_execution_binding_fingerprint": trace_v2.execution_binding_fingerprint,
            "expected_runtime_settings_fingerprint": guard.expected_runtime_settings_fingerprint,
            "actual_runtime_settings_fingerprint": (
                trace_v2.runtime_settings_binding.exact_settings_fingerprint
                if trace_v2.runtime_settings_binding
                else None
            ),
            "mismatch_category": mismatch_category,
            "request_id": request.request_id,
            "execution_id": trace_v2.execution_id,
            "generation_started": False,
        }

    def _verify_execution_guard(self, request: GenerationRequest, trace_v2: ExecutionTraceV2) -> None:
        """Verify strict v2 expectations after effective settings are bound."""

        if not isinstance(request, GenerationRequestV2) or request.execution_guard is None:
            return

        guard = request.execution_guard
        trace_v2.guard = guard
        expected_execution = guard.expected_execution_binding_fingerprint
        expected_settings = guard.expected_runtime_settings_fingerprint
        actual_execution = trace_v2.execution_binding_fingerprint
        actual_settings = (
            trace_v2.runtime_settings_binding.exact_settings_fingerprint if trace_v2.runtime_settings_binding else None
        )
        error: RuntimeFoundationError

        if expected_execution is None and expected_settings is None:
            category = "invalid_expectation"
            details = self._guard_details(request, trace_v2, mismatch_category=category)
            error = InvalidRequestError(
                "execution_guard must specify at least one expected fingerprint",
                details=details,
            )
            trace_v2.guard_verification = ExecutionGuardVerification(
                status="INVALID_EXPECTATION",
                mismatch_category=category,
                expected_execution_binding_fingerprint=expected_execution,
                actual_execution_binding_fingerprint=actual_execution,
                expected_runtime_settings_fingerprint=expected_settings,
                actual_runtime_settings_fingerprint=actual_settings,
                error=error.as_dict(),
            )
            raise error

        for field_name, value in (
            ("expected_execution_binding_fingerprint", expected_execution),
            ("expected_runtime_settings_fingerprint", expected_settings),
        ):
            if value is not None and not is_valid_fingerprint(value):
                category = "invalid_expectation"
                details = self._guard_details(request, trace_v2, mismatch_category=category)
                details["field"] = field_name
                error = InvalidRequestError(
                    f"{field_name} must use sha256:<64 lowercase hex characters>",
                    details=details,
                )
                trace_v2.guard_verification = ExecutionGuardVerification(
                    status="INVALID_EXPECTATION",
                    mismatch_category=category,
                    expected_execution_binding_fingerprint=expected_execution,
                    actual_execution_binding_fingerprint=actual_execution,
                    expected_runtime_settings_fingerprint=expected_settings,
                    actual_runtime_settings_fingerprint=actual_settings,
                    error=error.as_dict(),
                )
                raise error

        if expected_execution is not None:
            binding = trace_v2.execution_binding
            missing: list[str] = []
            content = binding.artifact_binding.content_identity
            if content is None or content.scheme != "complete":
                missing.append("artifact_binding.content_identity.complete")
            engine = binding.engine_binding
            if not engine.family:
                missing.append("engine_binding.family")
            if not engine.implementation.id:
                missing.append("engine_binding.implementation.id")
            if not engine.implementation.version:
                missing.append("engine_binding.implementation.version")
            if not engine.build_identity:
                missing.append("engine_binding.build_identity")
            if not engine.adapter_id:
                missing.append("engine_binding.adapter_id")
            foundation = binding.foundation_binding
            if not foundation.contract_version:
                missing.append("foundation_binding.contract_version")
            if not foundation.foundation_version:
                missing.append("foundation_binding.foundation_version")
            if not foundation.build_identity:
                missing.append("foundation_binding.build_identity")
            if not foundation.adapter_id:
                missing.append("foundation_binding.adapter_id")
            if actual_execution is None:
                missing.append("execution_binding_fingerprint")
            if missing:
                category = "execution_binding_unresolvable"
                details = self._guard_details(request, trace_v2, mismatch_category=category)
                details["unresolvable_fields"] = missing
                error = ExecutionBindingUnresolvableError(
                    "actual execution binding cannot be safely certified",
                    details=details,
                )
                trace_v2.guard_verification = ExecutionGuardVerification(
                    status="UNRESOLVABLE",
                    mismatch_category=category,
                    expected_execution_binding_fingerprint=expected_execution,
                    actual_execution_binding_fingerprint=actual_execution,
                    expected_runtime_settings_fingerprint=expected_settings,
                    actual_runtime_settings_fingerprint=actual_settings,
                    error=error.as_dict(),
                )
                raise error

        mismatch: str | None = None
        if expected_execution is not None and expected_execution != actual_execution:
            mismatch = "execution_binding_mismatch"
        if expected_settings is not None and expected_settings != actual_settings:
            mismatch = "runtime_settings_mismatch" if mismatch is None else "execution_guard_mismatch"
        if mismatch is not None:
            details = self._guard_details(request, trace_v2, mismatch_category=mismatch)
            error = ExecutionGuardMismatchError(
                "execution guard expectations do not match actual state", details=details
            )
            trace_v2.guard_verification = ExecutionGuardVerification(
                status="MISMATCH",
                mismatch_category=mismatch,
                expected_execution_binding_fingerprint=expected_execution,
                actual_execution_binding_fingerprint=actual_execution,
                expected_runtime_settings_fingerprint=expected_settings,
                actual_runtime_settings_fingerprint=actual_settings,
                error=error.as_dict(),
            )
            raise error

        trace_v2.guard_verification = ExecutionGuardVerification(
            status="MATCH",
            expected_execution_binding_fingerprint=expected_execution,
            actual_execution_binding_fingerprint=actual_execution,
            expected_runtime_settings_fingerprint=expected_settings,
            actual_runtime_settings_fingerprint=actual_settings,
            generation_started=False,
        )

    def _resolve_runtime_and_guard(
        self,
        request: GenerationRequest,
        selected: EngineAdapter,
        timeout_event: threading.Event,
        trace: ExecutionTrace,
    ) -> RuntimeSettingsResolution:
        """Shared settings-resolution and pre-generation safety path."""

        resolution = selected.resolve_runtime_options(request.runtime_options)
        self._raise_if_timed_out(request, timeout_event, trace)
        trace.runtime_settings_resolution = resolution
        trace.effective_runtime_settings = resolution.effective.to_dict()
        trace_v2: ExecutionTraceV2 = cast(ExecutionTraceV2, trace.trace_v2)
        trace_v2.with_runtime_settings(resolution.effective)
        self._verify_execution_guard(request, trace_v2)
        return resolution

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
            resolution = self._resolve_runtime_and_guard(request, selected, timeout_event, trace)
            trace_v2: ExecutionTraceV2 = cast(ExecutionTraceV2, trace.trace_v2)
            effective_request = self._prepare_effective_request(request, selected, resolution, trace_v2)
            trace_v2.generation_started = True
            if trace_v2.guard_verification is not None:
                trace_v2.guard_verification = replace(trace_v2.guard_verification, generation_started=True)
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
            adapter_events: Iterator[StreamEvent] | None = None
            adapter_events_closed = False
            finalized = False

            def close_adapter_events() -> None:
                nonlocal adapter_events_closed
                if adapter_events_closed:
                    return
                adapter_events_closed = True
                close = getattr(adapter_events, "close", None)
                if callable(close):
                    close()

            def finalize() -> None:
                nonlocal finalized
                if finalized:
                    return
                finalized = True
                try:
                    self._finish_trace(trace)
                finally:
                    self._end(request.request_id)

            try:
                resolution = self._resolve_runtime_and_guard(request, selected, timeout_event, trace)
                trace_v2: ExecutionTraceV2 = cast(ExecutionTraceV2, trace.trace_v2)
                effective_request = self._prepare_effective_request(request, selected, resolution, trace_v2)
                yield StreamEvent(type="started", request_id=request.request_id, sequence=sequence)
                trace_v2.generation_started = True
                if trace_v2.guard_verification is not None:
                    trace_v2.guard_verification = replace(trace_v2.guard_verification, generation_started=True)
                adapter_events = iter(selected.stream(effective_request, cancel_event))
                for event in adapter_events:
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
                    if event.type in {"completed", "error"}:
                        # Finalize both the engine generator and Core lifecycle before
                        # exposing a terminal event. HTTP clients commonly stop reading
                        # as soon as they receive done=true, so cleanup cannot depend on
                        # the consumer requesting another item from this generator.
                        close_adapter_events()
                        finalize()
                        yield event
                        return
                    yield event
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
                terminal_event = StreamEvent(
                    type="error",
                    request_id=request.request_id,
                    sequence=sequence,
                    done=True,
                    error=trace.error,
                )
                close_adapter_events()
                finalize()
                yield terminal_event
                return
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
                terminal_event = StreamEvent(
                    type="error",
                    request_id=request.request_id,
                    sequence=sequence,
                    done=True,
                    error=trace.error,
                )
                close_adapter_events()
                finalize()
                yield terminal_event
                return
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
                try:
                    close_adapter_events()
                finally:
                    finalize()

        return iterator()

    @staticmethod
    def _bridge_v3_request(request: GenerationRequestV3) -> tuple[GenerationRequestV3, GenerationRequestV2]:
        request_id = request.request_id or new_id("req")
        normalized = request if request.request_id == request_id else replace(request, request_id=request_id)
        options = normalized.generation_options
        bridge = GenerationRequestV2(
            model_artifact_id=normalized.model_artifact_id,
            messages=[dict(item) for item in normalized.messages],
            request_id=request_id,
            consumer_id=normalized.consumer_id,
            lease_id=normalized.lease_id,
            max_tokens=options.max_tokens,
            temperature=options.temperature,
            top_p=options.top_p,
            thinking_enabled=options.thinking_intent.to_legacy_enabled(),
            timeout_ms=normalized.execution_constraints.timeout_ms,
            metadata=dict(normalized.metadata),
            thinking_intent=options.thinking_intent,
            studio_resolved_thinking=normalized.studio_resolved_thinking,
            execution_input=normalized.execution_input,
        )
        return normalized, bridge

    def _execution_evidence_v3(
        self,
        request: GenerationRequestV3,
        selected: EngineAdapter,
    ) -> tuple[ExecutionEvidenceV3, GenerationOptions, PreparedGenerationV3]:
        with self._lock:
            artifact = self._loaded_artifact_v2
            load_resolution = self._loaded_load_options_resolution
            effective_load_options = self._loaded_load_options
            load_identity = self._loaded_load_identity
            execution_input = self._loaded_execution_input
        if artifact is None or effective_load_options is None or load_identity is None:
            raise ModelNotLoadedError("no v3 load identity is available")
        if load_resolution is None:
            load_resolution = LoadOptionsResolutionV1(
                requested=effective_load_options,
                resolved=effective_load_options,
                effective=effective_load_options,
            )
        load_evidence = SettingsEvidenceV3.from_options(
            scope="LOAD",
            requested=load_resolution.requested,
            resolved=load_resolution.resolved,
            effective=load_resolution.effective,
            resolutions=load_resolution.resolutions,
        )
        generation_evidence = selected.resolve_generation_options_v3(request)
        effective_generation = GenerationOptions.from_payload(generation_evidence.effective)
        prepared = selected.prepare_generation_v3(request, effective_generation)
        if not isinstance(prepared, PreparedGenerationV3):
            raise EngineRuntimeError(
                "engine adapter returned an invalid v3 prepared generation",
                details={"engine": selected.name},
            )

        requested_constraints = request.execution_constraints
        resolved_constraints = requested_constraints
        effective_constraints = requested_constraints
        constraint_resolutions: tuple[OptionResolutionV3, ...] = ()
        loaded_context = effective_load_options.model_context_size
        requested_budget = requested_constraints.max_context_tokens
        if requested_budget is not None and loaded_context is not None and requested_budget > loaded_context:
            capped_budget = loaded_context
            resolved_constraints = replace(requested_constraints, max_context_tokens=capped_budget)
            effective_constraints = resolved_constraints
            constraint_resolutions = (
                OptionResolutionV3(
                    path="max_context_tokens",
                    requested=requested_budget,
                    resolved=capped_budget,
                    effective=capped_budget,
                    status="resolved",
                    reason="execution budget was capped at the effective loaded model context size",
                ),
            )
        constraints_evidence = SettingsEvidenceV3.from_options(
            scope="CONSTRAINT",
            requested=requested_constraints,
            resolved=resolved_constraints,
            effective=effective_constraints,
            resolutions=constraint_resolutions,
        )
        if effective_constraints.max_context_tokens is not None:
            prompt_tokens = prepared.prompt_tokens
            if prompt_tokens is None:
                raise UnsupportedRuntimeOptionError(
                    "the selected engine cannot provide a verified prompt-token count for max_context_tokens",
                    details={"engine": selected.name, "constraint": "max_context_tokens", "generation_started": False},
                )
            required = prompt_tokens + effective_generation.max_tokens
            if required > effective_constraints.max_context_tokens:
                raise ContextLengthExceededError(
                    "prompt plus requested generation exceeds max_context_tokens",
                    details={
                        "prompt_tokens": prompt_tokens,
                        "max_tokens": effective_generation.max_tokens,
                        "required_tokens": required,
                        "max_context_tokens": effective_constraints.max_context_tokens,
                        "generation_started": False,
                    },
                )

        engine_binding = EngineBindingV2.from_legacy(
            selected.identity(),
            adapter_id=selected.name,
            build_identity=selected.build_identity(),
        )
        foundation_binding = FoundationBindingV2(
            contract_version=CONTRACT_V3_VERSION,
            foundation_version=self.foundation_version,
            build_identity=self.foundation_build_identity,
            adapter_id=selected.name,
        )
        binding = ExecutionBindingV3.create(
            artifact=artifact,
            engine=engine_binding,
            foundation=foundation_binding,
            effective_load_options=effective_load_options,
            effective_generation_options=effective_generation,
            effective_constraints=effective_constraints,
            execution_input=execution_input,
        )
        return (
            ExecutionEvidenceV3(
                execution_binding=binding,
                options=ExecutionOptionsEvidenceV3(
                    load=load_evidence,
                    generation=generation_evidence,
                    constraints=constraints_evidence,
                ),
            ),
            effective_generation,
            prepared,
        )

    def _save_v3_trace(self, trace: ExecutionTraceV3) -> None:
        with self._lock:
            self._v3_traces[trace.execution_id] = trace
            if len(self._v3_traces) > 256:
                oldest = next(iter(self._v3_traces))
                self._v3_traces.pop(oldest, None)

    def get_execution_v3(self, execution_id: str) -> dict[str, Any] | None:
        with self._lock:
            trace = self._v3_traces.get(execution_id)
        return trace.to_dict() if trace else None

    def generate_v3(self, request: GenerationRequestV3) -> GenerationResultV3:
        normalized_request, bridge = self._bridge_v3_request(request)
        _, selected, cancel_event, timeout_event, legacy_trace = self._begin(bridge)
        trace = ExecutionTraceV3(
            execution_id=legacy_trace.execution_id,
            request_id=bridge.request_id,
            engine=EngineBindingV2.from_legacy(
                selected.identity(), adapter_id=selected.name, build_identity=selected.build_identity()
            ).to_dict(),
            artifact_identity=(self._loaded_artifact_v2 or ArtifactBindingV2.from_legacy(legacy_trace.artifact)).canonical_payload(),
            status="running",
            started_at=legacy_trace.started_at,
        )
        try:
            evidence, effective_options, prepared = self._execution_evidence_v3(normalized_request, selected)
            trace.evidence = evidence
            self._raise_if_timed_out(bridge, timeout_event, legacy_trace)
            result = selected.generate_v3(normalized_request, effective_options, cancel_event, prepared=prepared)
            self._raise_if_timed_out(bridge, timeout_event, legacy_trace)
            if result.metrics.load_duration_ms is None:
                result.metrics.load_duration_ms = self._last_load_duration_ms
            if result.metrics.unload_duration_ms is None:
                result.metrics.unload_duration_ms = self._last_unload_duration_ms
            trace.status = "completed"
            trace.metrics = {key: value for key, value in result.metrics.to_dict().items() if key != "contract_version"}
            trace.finish_reason = result.finish_reason
            trace.finished_at = utc_now()
            usage = {key: value for key, value in result.usage.to_dict().items() if key != "contract_version"}
            self._last_error = None
            return GenerationResultV3(
                request_id=bridge.request_id,
                model_artifact_id=normalized_request.model_artifact_id,
                engine=selected.name,
                text=result.text,
                finish_reason=result.finish_reason,
                usage=usage,
                metrics=trace.metrics,
                execution_id=trace.execution_id,
                evidence=evidence,
            )
        except RuntimeFoundationError as exc:
            if timeout_event.is_set() and exc.code != "runtime_timeout":
                exc = self._timeout_error(bridge, legacy_trace)
            exc.details.setdefault("execution_id", legacy_trace.execution_id)
            trace.status = "cancelled" if exc.code == "cancelled" else "error"
            trace.error = error_payload(exc, execution_id=legacy_trace.execution_id)
            trace.finished_at = utc_now()
            self._last_error = trace.error
            raise exc
        except Exception as exc:
            wrapped = EngineRuntimeError(
                "engine generation failed", details={"engine": selected.name, "execution_id": legacy_trace.execution_id}
            )
            trace.status = "error"
            trace.error = error_payload(wrapped, execution_id=legacy_trace.execution_id)
            trace.finished_at = utc_now()
            self._last_error = trace.error
            raise wrapped from exc
        finally:
            if trace.status == "running":
                trace.status = "error" if timeout_event.is_set() else "cancelled" if cancel_event.is_set() else "abandoned"
                trace.finished_at = utc_now()
            self._save_v3_trace(trace)
            self._end(bridge.request_id)

    def stream_v3(self, request: GenerationRequestV3) -> Iterator[StreamEventV3]:
        normalized_request, bridge = self._bridge_v3_request(request)
        _, selected, cancel_event, timeout_event, legacy_trace = self._begin(bridge)
        trace = ExecutionTraceV3(
            execution_id=legacy_trace.execution_id,
            request_id=bridge.request_id,
            engine=EngineBindingV2.from_legacy(
                selected.identity(), adapter_id=selected.name, build_identity=selected.build_identity()
            ).to_dict(),
            artifact_identity=(self._loaded_artifact_v2 or ArtifactBindingV2.from_legacy(legacy_trace.artifact)).canonical_payload(),
            status="running",
            started_at=legacy_trace.started_at,
        )

        def iterator() -> Iterator[StreamEventV3]:
            sequence = 0
            adapter_events: Iterator[StreamEvent] | None = None
            terminal = False

            def close_adapter_events() -> None:
                close = getattr(adapter_events, "close", None)
                if callable(close):
                    close()

            def error_event(exc: RuntimeFoundationError) -> StreamEventV3:
                exc.details.setdefault("execution_id", trace.execution_id)
                trace.status = "cancelled" if exc.code == "cancelled" else "error"
                trace.error = error_payload(exc, execution_id=trace.execution_id)
                trace.finished_at = utc_now()
                self._last_error = trace.error
                return StreamEventV3(
                    type="error",
                    request_id=bridge.request_id,
                    sequence=sequence + 1,
                    execution_id=trace.execution_id,
                    evidence=trace.evidence,
                    done=True,
                    error=trace.error,
                )

            try:
                evidence, effective_options, prepared = self._execution_evidence_v3(normalized_request, selected)
                trace.evidence = evidence
                self._raise_if_timed_out(bridge, timeout_event, legacy_trace)
                yield StreamEventV3(
                    type="started",
                    request_id=bridge.request_id,
                    sequence=sequence,
                    execution_id=trace.execution_id,
                    evidence=evidence,
                )
                adapter_events = iter(
                    selected.stream_v3(normalized_request, effective_options, cancel_event, prepared=prepared)
                )
                for event in adapter_events:
                    self._raise_if_timed_out(bridge, timeout_event, legacy_trace)
                    if event.type == "started":
                        continue
                    sequence += 1
                    if event.type == "completed" and event.result is not None:
                        result = dict(event.result)
                        usage = dict(result.get("usage") or {})
                        usage.pop("contract_version", None)
                        metrics = dict(result.get("metrics") or {})
                        metrics.pop("contract_version", None)
                        metrics.setdefault("load_duration_ms", self._last_load_duration_ms)
                        result_payload = {
                            "contract_version": CONTRACT_V3_VERSION,
                            "schema_version": "runtime-foundation.generation-result.v3",
                            "request_id": bridge.request_id,
                            "model_artifact_id": normalized_request.model_artifact_id,
                            "engine": selected.name,
                            "text": result.get("text", ""),
                            "finish_reason": result.get("finish_reason", "unknown"),
                            "usage": usage,
                            "metrics": metrics,
                            "execution_id": trace.execution_id,
                            "execution_evidence": evidence.to_dict(),
                        }
                        trace.status = "completed"
                        trace.metrics = metrics
                        trace.finish_reason = result_payload["finish_reason"]
                        trace.finished_at = utc_now()
                        event_payload = StreamEventV3(
                            type="completed",
                            request_id=bridge.request_id,
                            sequence=sequence,
                            execution_id=trace.execution_id,
                            evidence=evidence,
                            done=True,
                            result=result_payload,
                        )
                        terminal = True
                        close_adapter_events()
                        self._save_v3_trace(trace)
                        self._end(bridge.request_id)
                        yield event_payload
                        return
                    if event.type == "error":
                        error = dict(event.error or {})
                        error.setdefault("details", {})["execution_id"] = trace.execution_id
                        if timeout_event.is_set() and error.get("code") != "runtime_timeout":
                            normalized_error = self._timeout_error(bridge, legacy_trace)
                            event_payload = error_event(normalized_error)
                        else:
                            trace.status = "cancelled" if error.get("code") == "cancelled" else "error"
                            trace.error = error
                            trace.finished_at = utc_now()
                            self._last_error = error
                            event_payload = StreamEventV3(
                                type="error",
                                request_id=bridge.request_id,
                                sequence=sequence,
                                execution_id=trace.execution_id,
                                evidence=evidence,
                                done=True,
                                error=error,
                            )
                        terminal = True
                        close_adapter_events()
                        self._save_v3_trace(trace)
                        self._end(bridge.request_id)
                        yield event_payload
                        return
                    yield StreamEventV3(
                        type=event.type,
                        request_id=bridge.request_id,
                        sequence=sequence,
                        execution_id=trace.execution_id,
                        evidence=evidence,
                        delta=event.delta,
                        done=event.done,
                        result=event.result,
                        error=event.error,
                        timestamp=event.timestamp,
                    )
                self._raise_if_timed_out(bridge, timeout_event, legacy_trace)
                raise EngineRuntimeError("engine stream ended without a terminal event", details={"engine": selected.name})
            except RuntimeFoundationError as exc:
                if timeout_event.is_set() and exc.code != "runtime_timeout":
                    exc = self._timeout_error(bridge, legacy_trace)
                terminal = True
                close_adapter_events()
                event_payload = error_event(exc)
                self._save_v3_trace(trace)
                self._end(bridge.request_id)
                yield event_payload
            except Exception as exc:
                wrapped = EngineRuntimeError("engine streaming failed", details={"engine": selected.name})
                terminal = True
                close_adapter_events()
                event_payload = error_event(wrapped)
                self._save_v3_trace(trace)
                self._end(bridge.request_id)
                yield event_payload
            finally:
                close_adapter_events()
                if not terminal:
                    if trace.status == "running":
                        trace.status = "error" if timeout_event.is_set() else "cancelled" if cancel_event.is_set() else "abandoned"
                        trace.finished_at = utc_now()
                    self._save_v3_trace(trace)
                    self._end(bridge.request_id)

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
        engine_build_identity = _build_identity_payload(selected.build_identity() if selected is not None else None)
        payload = HealthResult(
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
            foundation_build_identity=(
                self.foundation_build_identity.to_dict() if self.foundation_build_identity is not None else None
            ),
            engine_build_identity=engine_build_identity,
        ).to_dict()
        with self._lock:
            loaded_input = self._loaded_execution_input
        if loaded_input is not None:
            payload["loaded_execution_input_kind"] = loaded_input.kind
            payload["loaded_execution_input_fingerprint"] = loaded_input.fingerprint
        return payload

    def health_v3(self) -> dict[str, Any]:
        payload = self.health()
        payload["contract_version"] = CONTRACT_V3_VERSION
        payload["supported_contract_versions"] = list(SUPPORTED_CONTRACT_VERSIONS_V3)
        payload["host_execution_capabilities"] = self.host_v3()["execution_capabilities"]
        return payload

    def runtime_metrics(self) -> dict[str, Any]:
        with self._lock:
            loaded = self._loaded_artifact
            selected = self._loaded_adapter
            loaded_input = self._loaded_execution_input
            active = list(self._active)
            state = self._state
        engine_build_identity = _build_identity_payload(selected.build_identity() if selected is not None else None)
        payload = {
            "contract_version": CONTRACT_VERSION,
            "foundation_version": self.foundation_version,
            "foundation_build_identity": (
                self.foundation_build_identity.to_dict() if self.foundation_build_identity is not None else None
            ),
            "engine_build_identity": engine_build_identity,
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
        if loaded_input is not None:
            payload["loaded_execution_input_kind"] = loaded_input.kind
            payload["loaded_execution_input_fingerprint"] = loaded_input.fingerprint
        return payload

    def get_execution(self, execution_id: str) -> dict[str, Any] | None:
        with self._lock:
            trace = self._traces.get(execution_id)
        return trace.to_dict() if trace else None


# Migration-friendly spelling for consumers that used the PR #10 name.
InferenceManager = RuntimeCore
