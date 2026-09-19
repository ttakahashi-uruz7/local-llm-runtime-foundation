from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from runtime_foundation import (
    CONTRACT_V2_VERSION,
    CONTRACT_VERSION,
    ArtifactBindingV2,
    ArtifactLocator,
    ContentIdentity,
    EngineBindingV2,
    EngineIdentity,
    ExecutionBindingV2,
    ExecutionGuardV1,
    FoundationBindingV2,
    GenerationRequestV2,
    HostProfile,
    ModelArtifactBinding,
    RuntimeCore,
    RuntimeOptions,
    RuntimeSettingsBindingV2,
    ThinkingEffort,
    ThinkingIntent,
    ThinkingMode,
    canonical_fingerprint,
)
from runtime_foundation.errors import (
    ThinkingResolutionError,
    UnsupportedArtifactLocatorError,
    UnsupportedGenerationSettingError,
)
from runtime_foundation.service import create_app


def test_artifact_v2_separates_registry_content_and_locator(tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"artifact content")
    content = ContentIdentity.from_file(model)
    # Constructing via the explicit locator keeps the path out of content identity.
    binding = ArtifactBindingV2.from_payload(
        {
            "contract_version": CONTRACT_V2_VERSION,
            "registry_identity": {"artifact_id": "registry-model"},
            "content_identity": content.to_dict(),
            "locator": {"type": "filesystem", "value": str(model)},
            "format": "bin",
        }
    )
    payload = binding.to_dict()
    assert payload["registry_identity"] == {"artifact_id": "registry-model"}
    assert payload["content_identity"]["digest"] == content.digest
    assert payload["locator"]["value"] == str(model)
    assert str(model) not in payload["content_identity"]["digest"]
    assert len(content.digest) == 64
    assert ContentIdentity.from_file(model, scheme="fast").scheme == "fast"


def test_engine_binding_separates_mlx_family_implementation_and_unknown_build() -> None:
    binding = EngineBindingV2.from_legacy(
        EngineIdentity(engine="mlx", version="0.28.0", build="0.28.0"),
        adapter_id="mlx",
    )
    assert binding.family == "mlx"
    assert binding.implementation.id == "mlx-lm"
    assert binding.implementation.version == "0.28.0"
    assert binding.build_identity is None
    assert binding.adapter_id == "mlx"


def test_execution_binding_fingerprint_is_canonical_and_sensitive() -> None:
    left = {"engine": {"family": "mlx", "implementation": {"id": "mlx-lm"}}, "artifact": {"id": "a"}}
    right = {"artifact": {"id": "a"}, "engine": {"implementation": {"id": "mlx-lm"}, "family": "mlx"}}
    assert canonical_fingerprint(left) == canonical_fingerprint(right)
    assert canonical_fingerprint(left) != canonical_fingerprint({**left, "engine": {"family": "mock"}})


def _execution_binding(artifact: ArtifactBindingV2) -> ExecutionBindingV2:
    settings = RuntimeSettingsBindingV2.from_effective(RuntimeOptions())
    assert settings is not None
    return ExecutionBindingV2(
        artifact_binding=artifact,
        engine_binding=EngineBindingV2.from_legacy(
            EngineIdentity(engine="mock", version="0.1-mock", build="foundation-mock"),
            adapter_id="mock",
        ),
        foundation_binding=FoundationBindingV2(
            contract_version=CONTRACT_V2_VERSION,
            foundation_version="0.2.0",
            build_identity=None,
            adapter_id="mock",
        ),
        runtime_settings_binding=settings,
    )


def test_locator_only_change_does_not_change_execution_binding_fingerprint(tmp_path: Path) -> None:
    first_path = tmp_path / "first.bin"
    second_path = tmp_path / "second.bin"
    first_path.write_bytes(b"same content")
    second_path.write_bytes(b"same content")
    content = ContentIdentity.from_file(first_path)
    first = ArtifactBindingV2("model", ArtifactLocator("filesystem", str(first_path)), content, format="bin")
    second = ArtifactBindingV2("model", ArtifactLocator("filesystem", str(second_path)), content, format="bin")
    first_binding = _execution_binding(first)
    second_binding = _execution_binding(second)
    assert first_binding.fingerprint == second_binding.fingerprint
    assert ExecutionBindingV2.from_payload(first_binding.to_dict()).fingerprint == first_binding.fingerprint


def test_content_change_changes_execution_binding_fingerprint(tmp_path: Path) -> None:
    first_path = tmp_path / "first.bin"
    second_path = tmp_path / "second.bin"
    first_path.write_bytes(b"first content")
    second_path.write_bytes(b"second content")
    first = ArtifactBindingV2(
        "model", ArtifactLocator("filesystem", str(first_path)), ContentIdentity.from_file(first_path), format="bin"
    )
    second = ArtifactBindingV2(
        "model", ArtifactLocator("filesystem", str(first_path)), ContentIdentity.from_file(second_path), format="bin"
    )
    assert _execution_binding(first).fingerprint != _execution_binding(second).fingerprint


def test_complete_directory_hash_is_versioned_deterministic_and_framed(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "weights").mkdir(parents=True)
    (second / "weights").mkdir(parents=True)
    (first / "weights" / "b.bin").write_bytes(b"b")
    (first / "weights" / "a.bin").write_bytes(b"a")
    (second / "weights" / "a.bin").write_bytes(b"a")
    (second / "weights" / "b.bin").write_bytes(b"b")
    first_identity = ContentIdentity.from_file(first)
    second_identity = ContentIdentity.from_file(second)
    assert first_identity.digest == second_identity.digest
    assert first_identity.canonicalization_scheme == "complete-directory-manifest-v1"

    (second / "weights" / "b.bin").write_bytes(b"changed")
    assert ContentIdentity.from_file(second).digest != first_identity.digest
    (second / "weights" / "b.bin").rename(second / "weights" / "renamed.bin")
    assert ContentIdentity.from_file(second).digest != first_identity.digest


def test_legacy_64_hex_hash_is_not_inferred_as_complete() -> None:
    binding = ArtifactBindingV2.from_legacy(
        ModelArtifactBinding("historical", "C:/historical.bin", "bin", artifact_hash="sha256:" + "a" * 64)
    )
    assert binding.content_identity is not None
    assert binding.content_identity.scheme == "legacy"
    assert binding.content_identity.canonicalization_scheme is None


def test_exact_runtime_settings_fingerprint_uses_effective_settings() -> None:
    first = RuntimeOptions.from_payload({"context": {"context_length": 4096}})
    second = RuntimeOptions.from_payload({"context": {"context_length": 8192}})
    first_binding = RuntimeSettingsBindingV2.from_effective(first)
    second_binding = RuntimeSettingsBindingV2.from_effective(second)
    assert first_binding is not None and second_binding is not None
    assert first_binding.exact_settings_fingerprint != second_binding.exact_settings_fingerprint
    assert first_binding.exact_settings_fingerprint == canonical_fingerprint(first.to_dict())


@pytest.mark.parametrize(
    ("mode", "expected"),
    [(ThinkingMode.OFF, False), (ThinkingMode.AUTO, None), (ThinkingMode.ON, True)],
)
def test_thinking_v2_modes_and_legacy_mapping(mode: ThinkingMode, expected: bool | None) -> None:
    intent = ThinkingIntent(mode=mode)
    assert intent.to_legacy_enabled() is expected
    assert ThinkingIntent.from_payload(intent.to_dict()) == intent


@pytest.mark.parametrize("effort", list(ThinkingEffort))
def test_thinking_v2_efforts_round_trip(effort: ThinkingEffort) -> None:
    intent = ThinkingIntent(mode=ThinkingMode.ON, effort=effort, budget_tokens=1024)
    assert ThinkingIntent.from_payload(intent.to_dict()) == intent


def test_generation_request_v2_and_guard_round_trip() -> None:
    request = GenerationRequestV2(
        model_artifact_id="model",
        messages=[{"role": "user", "content": "hello"}],
        thinking_intent=ThinkingIntent(mode=ThinkingMode.ON, effort=ThinkingEffort.MEDIUM),
        execution_guard=ExecutionGuardV1(
            expected_execution_binding_fingerprint="sha256:binding",
            expected_runtime_settings_fingerprint="sha256:settings",
        ),
    )
    payload = request.to_dict()
    restored = GenerationRequestV2.from_payload(payload)
    assert payload["contract_version"] == CONTRACT_V2_VERSION
    assert restored.thinking_intent == request.thinking_intent
    assert restored.execution_guard == request.execution_guard
    assert restored.to_dict() == payload


def test_core_emits_v2_binding_trace_and_health_discovery(tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"v2 fixture")
    artifact = ModelArtifactBinding("fixture", str(model), "bin", artifact_hash="sha256:" + "a" * 64)
    core = RuntimeCore(host_profile=HostProfile.mock_windows())
    core.load(artifact, adapter="mock")
    result = core.generate(
        GenerationRequestV2(
            model_artifact_id=artifact.artifact_id,
            messages=[{"role": "user", "content": "v2"}],
            thinking_intent=ThinkingIntent(mode=ThinkingMode.ON),
        )
    )
    payload = result.to_dict()
    trace_v2 = payload["generation_v2"]["trace"]
    assert payload["contract_version"] == CONTRACT_VERSION
    assert trace_v2["contract_version"] == CONTRACT_V2_VERSION
    assert trace_v2["execution_binding_fingerprint"].startswith("sha256:")
    assert trace_v2["artifact_binding"]["content_identity"]["scheme"] == "legacy"
    assert trace_v2["artifact_binding"]["content_identity"]["canonicalization_scheme"] is None
    assert trace_v2["engine_binding"]["family"] == "mock"
    assert trace_v2["thinking_foundation_effective"]["mode"] == "ON"
    health = core.health()
    assert health["supported_contract_versions"] == [CONTRACT_VERSION, CONTRACT_V2_VERSION]


def test_service_accepts_v2_artifact_and_generation_request(tmp_path: Path) -> None:
    model = tmp_path / "service-v2.bin"
    model.write_bytes(b"service v2 fixture")
    content = ContentIdentity.from_file(model)
    client = TestClient(create_app(RuntimeCore(host_profile=HostProfile.mock_windows())))
    artifact = {
        "contract_version": CONTRACT_V2_VERSION,
        "registry_identity": {"artifact_id": "service-v2"},
        "content_identity": content.to_dict(),
        "locator": {"type": "filesystem", "value": str(model)},
        "format": "bin",
    }
    loaded = client.post("/models/load", json={"artifact": artifact, "engine": "mock"})
    assert loaded.status_code == 200
    generated = client.post(
        "/generate",
        json={
            "contract_version": CONTRACT_V2_VERSION,
            "schema_version": "runtime-foundation.generation-request.v2",
            "model_artifact_id": "service-v2",
            "lease_id": loaded.json()["lease_id"],
            "messages": [{"role": "user", "content": "service v2"}],
            "thinking_intent": {"mode": "OFF"},
        },
    )
    assert generated.status_code == 200
    assert generated.json()["generation_v2"]["contract_version"] == CONTRACT_V2_VERSION


def test_thinking_effort_cannot_silently_downgrade(tmp_path: Path) -> None:
    model = tmp_path / "model.bin"
    model.write_bytes(b"thinking fixture")
    artifact = ModelArtifactBinding("fixture", str(model), "bin")
    core = RuntimeCore(host_profile=HostProfile.mock_windows())
    core.load(artifact, adapter="mock")
    with pytest.raises(UnsupportedGenerationSettingError) as caught:
        core.generate(
            GenerationRequestV2(
                model_artifact_id=artifact.artifact_id,
                messages=[{"role": "user", "content": "high"}],
                thinking_intent=ThinkingIntent(mode=ThinkingMode.ON, effort=ThinkingEffort.HIGH),
            )
        )
    execution_id = caught.value.details["execution_id"]
    execution = core.get_execution(execution_id)
    assert execution is not None
    assert execution["trace_v2"]["thinking_resolution"]["status"] == "unsupported"
    assert execution["trace_v2"]["raw_execution_error"]["code"] == "unsupported_generation_setting"


def test_auto_without_studio_resolution_fails_explicitly(tmp_path: Path) -> None:
    model = tmp_path / "auto.bin"
    model.write_bytes(b"auto fixture")
    artifact = ModelArtifactBinding("auto", str(model), "bin")
    core = RuntimeCore(host_profile=HostProfile.mock_windows())
    core.load(artifact, adapter="mock")
    with pytest.raises(ThinkingResolutionError) as caught:
        core.generate(
            GenerationRequestV2(
                model_artifact_id=artifact.artifact_id,
                messages=[{"role": "user", "content": "auto"}],
                thinking_intent=ThinkingIntent(mode=ThinkingMode.AUTO),
            )
        )
    execution_id = caught.value.details["execution_id"]
    execution = core.get_execution(execution_id)
    assert execution is not None
    assert execution["trace_v2"]["thinking_resolution"]["status"] == "unresolved"
    assert execution["trace_v2"]["raw_execution_error"]["code"] == "thinking_resolution_error"


@pytest.mark.parametrize("resolved_mode", [ThinkingMode.ON, ThinkingMode.OFF])
def test_auto_uses_explicit_studio_resolution(tmp_path: Path, resolved_mode: ThinkingMode) -> None:
    model = tmp_path / f"auto-{resolved_mode.value}.bin"
    model.write_bytes(b"auto resolved fixture")
    artifact = ModelArtifactBinding("auto", str(model), "bin")
    core = RuntimeCore(host_profile=HostProfile.mock_windows())
    core.load(artifact, adapter="mock")
    result = core.generate(
        GenerationRequestV2(
            model_artifact_id=artifact.artifact_id,
            messages=[{"role": "user", "content": "auto"}],
            thinking_intent=ThinkingIntent(mode=ThinkingMode.AUTO),
            studio_resolved_thinking=ThinkingIntent(mode=resolved_mode),
        )
    )
    trace = result.to_dict()["generation_v2"]["trace"]
    assert trace["thinking_studio_resolved"]["mode"] == resolved_mode.value
    assert trace["thinking_foundation_effective"]["mode"] == resolved_mode.value


def test_fast_identity_stays_fast_through_load_generate_trace(tmp_path: Path) -> None:
    model = tmp_path / "fast.bin"
    model.write_bytes(b"fast fixture")
    content = ContentIdentity.from_file(model, scheme="fast")
    artifact = ArtifactBindingV2(
        "fast",
        ArtifactLocator("filesystem", str(model)),
        content,
        format="bin",
    )
    core = RuntimeCore(host_profile=HostProfile.mock_windows())
    core.load(artifact, adapter="mock")
    result = core.generate(
        GenerationRequestV2(
            model_artifact_id="fast",
            messages=[{"role": "user", "content": "fast"}],
            thinking_intent=ThinkingIntent(mode=ThinkingMode.OFF),
        )
    )
    content_payload = result.to_dict()["generation_v2"]["trace"]["artifact_binding"]["content_identity"]
    assert content_payload["scheme"] == "fast"
    assert content_payload["canonicalization_scheme"] == "fast-file-v1"


def test_non_filesystem_locator_is_explicitly_unsupported(tmp_path: Path) -> None:
    model = tmp_path / "unsupported.bin"
    model.write_bytes(b"unsupported locator")
    artifact = ArtifactBindingV2(
        "remote",
        ArtifactLocator("registry", "registry://remote/model"),
        ContentIdentity.from_file(model),
        format="bin",
    )
    with pytest.raises(UnsupportedArtifactLocatorError):
        RuntimeCore(host_profile=HostProfile.mock_windows()).load(artifact, adapter="mock")


def test_v1_trace_serialization_is_not_upgraded_without_v2_evidence() -> None:
    artifact = ModelArtifactBinding("model", "C:/model.bin", "bin")
    legacy = artifact.to_dict()
    assert legacy["contract_version"] == CONTRACT_VERSION
