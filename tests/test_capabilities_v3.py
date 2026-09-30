from __future__ import annotations

from runtime_foundation import (
    CapabilityApplicabilityV3,
    CapabilityDependency,
    CapabilityScope,
    CapabilityStatus,
    EngineBindingV2,
    EngineCapability,
    EngineCapabilityV3,
    EngineIdentity,
    HostExecutionCapabilitiesV3,
    HostProfile,
    OptionCapabilityV3,
    RuntimeCore,
)
from runtime_foundation.adapters.llama_cpp import LlamaCppAdapter
from runtime_foundation.adapters.mock import MockAdapter


def _available_llama_capability() -> EngineCapabilityV3:
    build = LlamaCppAdapter()
    legacy = EngineCapability(
        identity=EngineIdentity("llama.cpp", "0.3", "native-build"),
        available=True,
        streaming=True,
        cancellation=True,
        load_unload=True,
        artifact_formats=["gguf"],
        runtime_options={
            "acceleration.backend": {
                "status": "supported",
                "supported_values": ["cpu", "metal"],
                "reason": "native llama.cpp system info reports Metal",
            },
            "load.model_context_size": {"status": "unknown"},
        },
        generation_options={"temperature": {"status": "supported", "minimum": 0, "maximum": 2}},
    )
    return EngineCapabilityV3.from_legacy(
        legacy,
        engine_binding=EngineBindingV2.from_legacy(
            legacy.identity,
            adapter_id=build.name,
            build_identity=build.build_identity(),
        ),
        host_observation={"platform": "darwin", "architecture": "arm64"},
    )


def test_option_capability_v3_preserves_status_scope_and_applicability() -> None:
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"mock": MockAdapter()})
    payload = core.capability_v3("mock")

    assert payload["schema_version"] == "runtime-foundation.engine-capability.v3"
    assert payload["status"] == "supported"
    option = payload["options"]["context.context_length"]
    assert option["status"] == "supported"
    assert option["scope"] == "LOAD"
    assert option["requires_reload"] is True
    assert option["applicability"]["depends_on"] == ["artifact", "engine", "engine_build", "host", "load"]
    assert option["applicability"]["engine_identity"]["family"] == "mock"


def test_unknown_unsupported_and_engine_unavailable_are_distinct() -> None:
    legacy = EngineCapability(
        identity=EngineIdentity("example"),
        available=True,
        streaming=False,
        cancellation=False,
        load_unload=True,
        runtime_options={
            "known-unsupported": {"status": "unsupported", "reason": "not implemented"},
            "known-unavailable": {"status": "unavailable", "reason": "missing library"},
        },
    )
    binding = EngineBindingV2.from_legacy(legacy.identity, adapter_id="example")
    capability = EngineCapabilityV3.from_legacy(legacy, engine_binding=binding)
    assert capability.options["known-unsupported"].status == CapabilityStatus.UNSUPPORTED
    assert capability.options["known-unavailable"].status == CapabilityStatus.UNAVAILABLE
    unknown = OptionCapabilityV3.from_legacy(
        value=None,
        scope=CapabilityScope.LOAD,
        engine_available=True,
        reason=None,
        applicability=CapabilityApplicabilityV3(),
    )
    assert unknown.status == CapabilityStatus.UNKNOWN

    unavailable_legacy = EngineCapability(
        identity=EngineIdentity("example"),
        available=False,
        streaming=False,
        cancellation=False,
        load_unload=False,
        runtime_options={"cpu": {"status": "unsupported"}},
    )
    unavailable = EngineCapabilityV3.from_legacy(
        unavailable_legacy,
        engine_binding=EngineBindingV2.from_legacy(unavailable_legacy.identity, adapter_id="example"),
    )
    assert unavailable.status == CapabilityStatus.UNAVAILABLE
    assert unavailable.options["cpu"].status == CapabilityStatus.UNSUPPORTED
    assert OptionCapabilityV3.from_legacy(
        value=None,
        scope=CapabilityScope.LOAD,
        engine_available=False,
        reason="llama.cpp binding is missing",
        applicability=CapabilityApplicabilityV3(),
    ).status == CapabilityStatus.UNAVAILABLE


def test_capability_shape_supports_constraint_scope_and_ranges() -> None:
    capability = OptionCapabilityV3(
        status=CapabilityStatus.SUPPORTED,
        scope=CapabilityScope.CONSTRAINT,
        minimum=1,
        maximum=4096,
        unit="tokens",
        evidence={"enforcement": "Foundation preflight"},
        applicability=CapabilityApplicabilityV3(depends_on=(CapabilityDependency.ARTIFACT, CapabilityDependency.HOST)),
    ).to_dict()
    assert capability["scope"] == "CONSTRAINT"
    assert capability["range"] == {"minimum": 1, "maximum": 4096, "unit": "tokens"}
    assert capability["evidence"]["enforcement"] == "Foundation preflight"


def test_host_metal_mlx_and_llama_build_observations_are_independent() -> None:
    capabilities = HostExecutionCapabilitiesV3.observe(
        platform="darwin",
        architecture="arm64",
        host_metal=True,
        host_metal_reason="system profiler reports Metal",
        mlx_available=False,
        mlx_default_device_metal=False,
        llama_cpp_capability=_available_llama_capability(),
    ).to_dict()
    assert capabilities["host_apple_silicon"] == "supported"
    assert capabilities["host_metal"] == "supported"
    assert capabilities["mlx_availability"] == "unavailable"
    assert capabilities["mlx_metal"] == "unavailable"
    assert capabilities["llama_cpp_availability"] == "supported"
    assert capabilities["llama_cpp_build_metal"] == "supported"


def test_mac_host_metal_does_not_imply_llama_build_metal() -> None:
    legacy = EngineCapability(
        identity=EngineIdentity("llama.cpp"),
        available=True,
        streaming=True,
        cancellation=True,
        load_unload=True,
    )
    capability = EngineCapabilityV3.from_legacy(
        legacy,
        engine_binding=EngineBindingV2.from_legacy(legacy.identity, adapter_id="llama.cpp"),
    )
    host = HostExecutionCapabilitiesV3.observe(
        platform="darwin",
        architecture="arm64",
        host_metal=True,
        host_metal_reason="Metal host observed",
        mlx_available=False,
        mlx_default_device_metal=False,
        llama_cpp_capability=capability,
    ).to_dict()
    assert host["host_metal"] == "supported"
    assert host["llama_cpp_build_metal"] == "unknown"


def test_legacy_host_metal_field_keeps_its_existing_meaning() -> None:
    legacy = HostProfile.mock_windows().to_dict()
    assert legacy["metal_available"] is False
    assert "host_metal" not in legacy
