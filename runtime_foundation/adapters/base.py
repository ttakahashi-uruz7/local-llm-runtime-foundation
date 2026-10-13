"""Engine adapter boundary."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator

from ..contracts import (
    EngineCapability,
    EngineIdentity,
    GenerationRequest,
    GenerationResult,
    ModelArtifactBinding,
    RuntimeOptions,
    RuntimeSettingsResolution,
    StreamEvent,
)
from ..contracts_v2 import ArtifactBindingV2, EngineBindingV2, ExecutionInputV1, GenerationRequestV2
from ..contracts_v3 import (
    GenerationOptions,
    GenerationRequestV3,
    LoadOptions,
    LoadOptionsResolutionV1,
    OptionResolutionV3,
    SettingsEvidenceV3,
)
from ..capabilities_v3 import EngineCapabilityV3
from ..errors import (
    ThinkingResolutionError,
    UnsupportedExecutionInputError,
    UnsupportedGenerationSettingError,
    UnsupportedRuntimeOptionError,
)


@dataclass(frozen=True)
class PreparedGenerationV3:
    """One adapter-rendered prompt reused for both preflight and execution."""

    prompt_tokens: int | None
    adapter_state: Any = None


def off_without_template_control_resolution(control_names: tuple[str, ...]) -> OptionResolutionV3:
    """Record that Thinking OFF ran on a template that has no thinking control."""

    return OptionResolutionV3(
        path="thinking_intent.template_control",
        requested=list(control_names),
        resolved=None,
        effective=None,
        status="observed",
        reason=(
            "the loaded chat template references none of the listed thinking controls, so Thinking OFF "
            "is rendered without a thinking flag; the consumer must check outputs for reasoning text"
        ),
    )


class EngineAdapter(ABC):
    """The only execution interface Core may use for a model engine."""

    name: str

    def build_identity(self) -> Any | None:
        """Return structured observed build identity, or unknown."""

        return None

    def identity(self) -> EngineIdentity:
        capability = self.discover_capability()
        return capability.identity

    @abstractmethod
    def discover_capability(self) -> EngineCapability:
        raise NotImplementedError

    def discover_capability_v3(
        self,
        *,
        artifact: ArtifactBindingV2 | None = None,
        host_observation: dict[str, Any] | None = None,
        load_identity_fingerprint: str | None = None,
    ) -> EngineCapabilityV3:
        """Return option-level v3 capabilities without changing the v1 payload."""

        return EngineCapabilityV3.from_legacy(
            self.discover_capability(),
            engine_binding=EngineBindingV2.from_legacy(
                self.identity(),
                adapter_id=self.name,
                build_identity=self.build_identity(),
            ),
            host_observation=host_observation,
            artifact=artifact,
            load_identity_fingerprint=load_identity_fingerprint,
        )

    @abstractmethod
    def load(self, artifact: ModelArtifactBinding) -> dict[str, Any]:
        raise NotImplementedError

    def validate_execution_input(self, execution_input: ExecutionInputV1) -> None:
        """Preflight a composition before any engine model load is attempted."""

        if execution_input.adapters:
            raise UnsupportedExecutionInputError(
                "this engine adapter does not support direct LoRA execution",
                details={"engine": self.name, "adapter_count": len(execution_input.adapters)},
            )

    def load_execution_input(self, execution_input: ExecutionInputV1) -> dict[str, Any]:
        """Load a base-only composition through the existing single-artifact path."""

        self.validate_execution_input(execution_input)
        return self.load(execution_input.base.to_legacy())

    def resolve_load_options(self, artifact: ModelArtifactBinding, options: LoadOptions) -> LoadOptionsResolutionV1:
        """Validate v3 load settings; adapters opt in to each non-default setting."""

        del artifact
        default = LoadOptions()
        requested_payload = options.to_dict()
        default_payload = default.to_dict()
        unsupported = [
            key
            for key in requested_payload
            if key not in {"contract_version", "schema_version"} and requested_payload[key] != default_payload[key]
        ]
        if unsupported:
            raise UnsupportedRuntimeOptionError(
                "the selected engine adapter has no declared v3 LoadOptions implementation",
                details={"engine": self.name, "unsupported_paths": unsupported},
            )
        return LoadOptionsResolutionV1(requested=options, resolved=options, effective=options)

    def resolve_load_options_with_observation(
        self,
        artifact: ModelArtifactBinding,
        options: LoadOptions,
        artifact_observation: Any,
    ) -> LoadOptionsResolutionV1:
        """Optional validated-artifact metadata path used by artifact-specific adapters."""

        del artifact_observation
        return self.resolve_load_options(artifact, options)

    def load_with_options(
        self,
        artifact: ModelArtifactBinding,
        resolution: LoadOptionsResolutionV1,
    ) -> dict[str, Any]:
        """Load through the legacy primitive after the v3 options were validated."""

        raw = self.load(artifact)
        return {**raw, "effective_load_options": resolution.effective.to_dict()}

    def load_execution_input_with_options(
        self,
        execution_input: ExecutionInputV1,
        resolution: LoadOptionsResolutionV1,
    ) -> dict[str, Any]:
        if resolution.requested != LoadOptions():
            raise UnsupportedRuntimeOptionError(
                "this adapter does not implement v3 LoadOptions for composed execution inputs",
                details={"engine": self.name, "execution_input_kind": execution_input.kind},
            )
        raw = self.load_execution_input(execution_input)
        return {**raw, "effective_load_options": resolution.effective.to_dict()}

    def resolve_generation_options_v3(self, request: GenerationRequestV3) -> SettingsEvidenceV3:
        """Validate v3 generation settings before mapping the representable v1/v2 subset."""

        options = request.generation_options
        unsupported_fields = {
            "top_k": options.top_k,
            "repetition_penalty": options.repetition_penalty,
            "repetition_window": options.repetition_window,
            "stop": list(options.stop) if options.stop else None,
            "seed": options.seed,
        }
        unsupported = [name for name, value in unsupported_fields.items() if value is not None]
        if unsupported:
            raise UnsupportedGenerationSettingError(
                "the adapter's common v1/v2 generation entry does not implement every requested v3 setting",
                details={"engine": self.name, "unsupported_options": unsupported},
            )

        capability = self.discover_capability().generation_options
        for name, value in (("max_tokens", options.max_tokens), ("temperature", options.temperature)):
            if capability.get(name, {}).get("status") != "supported":
                raise UnsupportedGenerationSettingError(
                    f"the selected engine does not report support for generation_options.{name}",
                    details={"engine": self.name, "option": name, "requested": value},
                )
        if options.top_p is not None and capability.get("top_p", {}).get("status") != "supported":
            raise UnsupportedGenerationSettingError(
                "the selected engine does not report support for generation_options.top_p",
                details={"engine": self.name, "option": "top_p", "requested": options.top_p},
            )

        intent = options.thinking_intent
        if intent.mode.value == "AUTO":
            resolved_intent = request.studio_resolved_thinking
            if resolved_intent is None or resolved_intent.mode.value not in {"OFF", "ON"}:
                raise ThinkingResolutionError(
                    "AUTO thinking requests require an explicit Studio ON/OFF resolution",
                    details={"requested": intent.to_dict(), "studio_resolved": None},
                )
        else:
            resolved_intent = intent
        if capability.get("thinking_enabled", {}).get("status") != "supported":
            raise UnsupportedGenerationSettingError(
                "the selected adapter cannot represent the requested Thinking Intent",
                details={"engine": self.name, "requested": intent.to_dict()},
            )
        if intent.effort is not None and capability.get("thinking_effort", {}).get("status") != "supported":
            raise UnsupportedGenerationSettingError(
                "the selected adapter cannot represent Thinking effort",
                details={"engine": self.name, "requested": intent.to_dict()},
            )
        if intent.budget_tokens is not None and capability.get("thinking_budget_tokens", {}).get("status") != "supported":
            raise UnsupportedGenerationSettingError(
                "the selected adapter cannot represent Thinking token budget",
                details={"engine": self.name, "requested": intent.to_dict()},
            )
        effective = GenerationOptions(
            max_tokens=options.max_tokens,
            temperature=options.temperature,
            top_p=options.top_p,
            thinking_intent=resolved_intent,
        )
        resolutions: tuple[OptionResolutionV3, ...] = ()
        resolved = effective
        if intent != resolved_intent:
            resolutions = (
                OptionResolutionV3(
                    path="thinking_intent.mode",
                    requested=intent.to_dict()["mode"],
                    resolved=resolved_intent.to_dict()["mode"],
                    effective=resolved_intent.to_dict()["mode"],
                    status="resolved",
                    reason="Foundation applied the explicit Studio ON/OFF resolution for AUTO intent",
                ),
            )
            resolved = GenerationOptions(
                max_tokens=options.max_tokens,
                temperature=options.temperature,
                top_p=options.top_p,
                thinking_intent=resolved_intent,
            )
        return SettingsEvidenceV3.from_options(
            scope="GENERATION",
            requested=options,
            resolved=resolved,
            effective=effective,
            resolutions=resolutions,
        )

    def count_prompt_tokens_v3(self, request: GenerationRequestV3) -> int | None:
        del request
        return None

    def prepare_generation_v3(
        self,
        request: GenerationRequestV3,
        effective_options: GenerationOptions,
    ) -> PreparedGenerationV3:
        """Prepare execution input once so a budget and generation share one prompt."""

        del effective_options
        return PreparedGenerationV3(prompt_tokens=self.count_prompt_tokens_v3(request))

    @staticmethod
    def _legacy_v2_request(request: GenerationRequestV3, options: GenerationOptions) -> GenerationRequestV2:
        thinking_enabled = options.thinking_intent.to_legacy_enabled()
        return GenerationRequestV2(
            model_artifact_id=request.model_artifact_id,
            messages=[dict(item) for item in request.messages],
            request_id=request.request_id or "",
            consumer_id=request.consumer_id,
            lease_id=request.lease_id,
            max_tokens=options.max_tokens,
            temperature=options.temperature,
            top_p=options.top_p,
            thinking_enabled=thinking_enabled,
            timeout_ms=request.execution_constraints.timeout_ms,
            metadata=dict(request.metadata),
            thinking_intent=options.thinking_intent,
            studio_resolved_thinking=request.studio_resolved_thinking,
            execution_input=request.execution_input,
        )

    def generate_v3(
        self,
        request: GenerationRequestV3,
        effective_options: GenerationOptions,
        cancel_event: threading.Event,
        prepared: PreparedGenerationV3 | None = None,
    ) -> GenerationResult:
        del prepared
        return self.generate(self._legacy_v2_request(request, effective_options), cancel_event)

    def stream_v3(
        self,
        request: GenerationRequestV3,
        effective_options: GenerationOptions,
        cancel_event: threading.Event,
        prepared: PreparedGenerationV3 | None = None,
    ) -> Iterator[StreamEvent]:
        del prepared
        yield from self.stream(self._legacy_v2_request(request, effective_options), cancel_event)

    @abstractmethod
    def unload(self, artifact_id: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def resolve_runtime_options(self, options: RuntimeOptions) -> RuntimeSettingsResolution:
        """Validate and resolve settings without silently dropping a field."""

        raise NotImplementedError

    @abstractmethod
    def generate(self, request: GenerationRequest, cancel_event: threading.Event) -> GenerationResult:
        raise NotImplementedError

    @abstractmethod
    def stream(self, request: GenerationRequest, cancel_event: threading.Event) -> Iterator[StreamEvent]:
        raise NotImplementedError

    @abstractmethod
    def cancel(self, request_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def health(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def runtime_metrics(self) -> dict[str, Any]:
        raise NotImplementedError
