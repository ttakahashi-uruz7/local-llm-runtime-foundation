from __future__ import annotations

import pytest

from runtime_foundation import (
    CONTRACT_V2_VERSION,
    CONTRACT_V3_VERSION,
    ArtifactBindingV2,
    ArtifactLocator,
    BuildIdentityV1,
    EngineBindingV2,
    EngineIdentity,
    ExecutionBindingV3,
    ExecutionConstraints,
    ExecutionOptionsEvidenceV3,
    FoundationBindingV2,
    GenerationOptions,
    GenerationRequest,
    GenerationRequestV2,
    GenerationRequestV3,
    LoadOptions,
    LoadIdentityV1,
    ModelArtifactBinding,
    OptionResolutionV3,
    RuntimeOptions,
    SettingsEvidenceV3,
    ThinkingIntent,
    ThinkingMode,
)
from runtime_foundation.errors import UnsupportedRuntimeOptionError


def test_v3_load_options_and_execution_constraints_are_independent() -> None:
    load = LoadOptions.from_payload(
        {
            "contract_version": CONTRACT_V3_VERSION,
            "model_context_size": 32768,
            "batch": 512,
            "ubatch": 128,
            "threads": 8,
            "kv_cache": {"key_type": "q8_0", "value_type": "f16"},
            "acceleration": {"backend": "metal", "gpu_offload_layers": 24},
        }
    )
    constraints = ExecutionConstraints.from_payload(
        {"contract_version": CONTRACT_V3_VERSION, "max_context_tokens": 8192, "timeout_ms": 5000}
    )
    assert load.model_context_size == 32768
    assert constraints.max_context_tokens == 8192
    assert "max_context_tokens" not in load.to_dict()
    assert "model_context_size" not in constraints.to_dict()
    assert LoadOptions.from_payload(load.to_dict()) == load
    assert ExecutionConstraints.from_payload(constraints.to_dict()) == constraints


def test_v3_options_validate_ranges_and_batch_relationship() -> None:
    with pytest.raises(ValueError, match="ubatch must be <="):
        LoadOptions(batch=128, ubatch=256)
    with pytest.raises(ValueError, match="model_context_size"):
        LoadOptions(model_context_size=0)
    with pytest.raises(ValueError, match="backend must be"):
        LoadOptions.from_payload({"acceleration": {"backend": "cuda"}})
    assert GenerationOptions(top_k=0).top_k == 0


def test_generation_options_are_one_authoritative_v3_source() -> None:
    request = GenerationRequestV3.from_payload(
        {
            "contract_version": CONTRACT_V3_VERSION,
            "schema_version": "runtime-foundation.generation-request.v3",
            "model_artifact_id": "model",
            "messages": [{"role": "user", "content": "hello"}],
            "generation_options": {
                "max_tokens": 40,
                "temperature": 0.5,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
                "repetition_window": 64,
                "stop": ["END"],
                "seed": 42,
                "thinking_intent": {"mode": "ON", "effort": "LOW", "budget_tokens": 256},
            },
            "execution_constraints": {"max_context_tokens": 4096},
        }
    )
    payload = request.to_dict()
    assert payload["generation_options"]["max_tokens"] == 40
    assert payload["generation_options"]["top_k"] == 50
    assert payload["generation_options"]["thinking_intent"]["budget_tokens"] == 256
    assert payload["execution_constraints"]["max_context_tokens"] == 4096
    assert "max_tokens" not in payload
    assert GenerationRequestV3.from_payload(payload) == request


def test_v3_rejects_legacy_duplicate_generation_fields() -> None:
    with pytest.raises(ValueError, match="unsupported fields"):
        GenerationRequestV3.from_payload(
            {
                "contract_version": CONTRACT_V3_VERSION,
                "model_artifact_id": "model",
                "messages": [{"role": "user", "content": "hello"}],
                "generation_options": {"max_tokens": 10},
                "max_tokens": 20,
            }
        )


def test_explicit_v1_conversion_preserves_context_as_execution_budget() -> None:
    request = GenerationRequest.from_payload(
        {
            "model_artifact_id": "legacy",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 20,
            "temperature": 0.4,
            "thinking_enabled": False,
            "timeout_ms": 1000,
            "runtime_options": {"context": {"context_length": 2048}},
        }
    )
    converted = GenerationRequestV3.from_legacy_v1(request)
    assert converted.generation_options.max_tokens == 20
    assert converted.generation_options.temperature == 0.4
    assert converted.generation_options.thinking_intent.mode == ThinkingMode.OFF
    assert converted.execution_constraints.max_context_tokens == 2048
    assert converted.execution_constraints.timeout_ms == 1000


def test_explicit_legacy_conversion_does_not_silently_drop_runtime_options() -> None:
    legacy_options = RuntimeOptions.from_payload({"kv_cache": {"mode": "quantized", "bits": 8}})
    with pytest.raises(UnsupportedRuntimeOptionError) as caught:
        ExecutionConstraints.from_legacy_v1(legacy_options)
    assert caught.value.details["fields"] == ["kv_cache"]


def test_v2_conversion_rejects_conflicting_thinking_fields() -> None:
    request = GenerationRequestV2.from_payload(
        {
            "contract_version": CONTRACT_V2_VERSION,
            "model_artifact_id": "legacy-v2",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking_enabled": True,
            "thinking_intent": {"mode": "OFF"},
        }
    )
    with pytest.raises(ValueError, match="conflicts with thinking_intent"):
        GenerationOptions.from_legacy_v2(request)


def test_v2_conversion_preserves_explicit_thinking_intent() -> None:
    request = GenerationRequestV2.from_payload(
        {
            "contract_version": CONTRACT_V2_VERSION,
            "model_artifact_id": "legacy-v2",
            "messages": [{"role": "user", "content": "hello"}],
            "thinking_intent": {"mode": "ON", "effort": "MEDIUM"},
        }
    )
    converted = GenerationRequestV3.from_legacy_v2(request)
    assert converted.generation_options.thinking_intent == ThinkingIntent(
        mode=ThinkingMode.ON, effort="MEDIUM"
    )


def _v3_binding(
    *,
    path: str = "models/fixture.gguf",
    load_options: LoadOptions | None = None,
    generation_options: GenerationOptions | None = None,
    constraints: ExecutionConstraints | None = None,
    engine_build: BuildIdentityV1 | None = None,
    artifact_hash: str = "a" * 64,
) -> ExecutionBindingV3:
    artifact = ArtifactBindingV2(
        artifact_id="fixture",
        locator=ArtifactLocator("filesystem", path),
        content_identity=ArtifactBindingV2.from_legacy(
            ModelArtifactBinding("fixture", path, "gguf", artifact_hash=f"sha256:{artifact_hash}")
        ).content_identity,
        format="gguf",
        quantization="Q4_K_M",
        revision="r1",
    )
    engine = EngineBindingV2.from_legacy(
        EngineIdentity(engine="llama.cpp", version="0.3.0"),
        adapter_id="llama.cpp",
        build_identity=engine_build,
    )
    foundation = FoundationBindingV2(
        contract_version=CONTRACT_V2_VERSION,
        foundation_version="0.3.0",
        build_identity=BuildIdentityV1.from_components(
            kind="test-foundation-v1", components={"package": {"version": "fixture"}}
        ),
        adapter_id="llama.cpp",
    )
    return ExecutionBindingV3.create(
        artifact=artifact,
        engine=engine,
        foundation=foundation,
        effective_load_options=load_options or LoadOptions(),
        effective_generation_options=generation_options or GenerationOptions(),
        effective_constraints=constraints or ExecutionConstraints(),
    )


def test_v3_load_identity_is_content_engine_build_and_load_setting_bound() -> None:
    build = BuildIdentityV1.from_components(
        kind="llama-cpp-native-build-v1",
        components={"library": {"version": "0.3.0", "metal": False}},
    )
    base = _v3_binding(path="/models/one.gguf", engine_build=build)
    relocated = _v3_binding(path="/models/two.gguf", engine_build=build)
    changed_load = _v3_binding(
        path="/models/one.gguf",
        engine_build=build,
        load_options=LoadOptions(model_context_size=8192),
    )
    changed_artifact = _v3_binding(path="/models/one.gguf", engine_build=build, artifact_hash="b" * 64)
    changed_build = _v3_binding(
        path="/models/one.gguf",
        engine_build=BuildIdentityV1.from_components(
            kind="llama-cpp-native-build-v1",
            components={"library": {"version": "0.3.1", "metal": False}},
        ),
    )
    assert base.load_identity.fingerprint == relocated.load_identity.fingerprint
    assert base.fingerprint == relocated.fingerprint
    assert base.load_identity.fingerprint != changed_load.load_identity.fingerprint
    assert base.load_identity.fingerprint != changed_artifact.load_identity.fingerprint
    assert base.load_identity.fingerprint != changed_build.load_identity.fingerprint


def test_v3_execution_binding_fingerprint_includes_each_effective_option_group() -> None:
    base = _v3_binding()
    changed_load = _v3_binding(load_options=LoadOptions(batch=256))
    changed_generation = _v3_binding(generation_options=GenerationOptions(top_k=40))
    changed_constraints = _v3_binding(constraints=ExecutionConstraints(max_context_tokens=4096))
    assert base.fingerprint != changed_load.fingerprint
    assert base.fingerprint != changed_generation.fingerprint
    assert base.fingerprint != changed_constraints.fingerprint
    assert ExecutionBindingV3.from_payload(base.to_dict()).fingerprint == base.fingerprint


def test_v3_evidence_requires_explicit_reason_for_requested_resolved_effective_change() -> None:
    requested = LoadOptions(model_context_size=8192)
    resolved = LoadOptions(model_context_size=8192)
    effective = LoadOptions(model_context_size=4096)
    with pytest.raises(ValueError, match="require explicit resolution records"):
        SettingsEvidenceV3.from_options(
            scope="LOAD", requested=requested, resolved=resolved, effective=effective
        )

    resolution = OptionResolutionV3(
        path="model_context_size",
        requested=8192,
        resolved=8192,
        effective=4096,
        status="observed",
        reason="native runtime reported its effective context allocation",
    )
    load = SettingsEvidenceV3.from_options(
        scope="LOAD", requested=requested, resolved=resolved, effective=effective, resolutions=(resolution,)
    )
    generation = SettingsEvidenceV3.from_options(
        scope="GENERATION", requested=GenerationOptions(), resolved=GenerationOptions(), effective=GenerationOptions()
    )
    constraints = SettingsEvidenceV3.from_options(
        scope="CONSTRAINT",
        requested=ExecutionConstraints(max_context_tokens=4096),
        resolved=ExecutionConstraints(max_context_tokens=4096),
        effective=ExecutionConstraints(max_context_tokens=4096),
    )
    evidence = ExecutionOptionsEvidenceV3(load=load, generation=generation, constraints=constraints)
    assert evidence.load.to_dict()["requested"]["model_context_size"] == 8192
    assert evidence.load.to_dict()["effective"]["model_context_size"] == 4096
    assert ExecutionOptionsEvidenceV3.from_payload(evidence.to_dict()) == evidence


def test_v3_load_identity_and_binding_reject_internal_inconsistency() -> None:
    binding = _v3_binding()
    payload = binding.to_dict()
    payload["effective_load_options"]["batch"] = 128
    payload.pop("execution_binding_fingerprint")
    with pytest.raises(ValueError, match="load_identity options do not match"):
        ExecutionBindingV3.from_payload(payload)

    wrong_identity = LoadIdentityV1.create(
        artifact=binding.artifact_binding,
        engine=binding.engine_binding,
        effective_load_options=LoadOptions(batch=64),
    )
    assert wrong_identity.fingerprint != binding.load_identity.fingerprint
