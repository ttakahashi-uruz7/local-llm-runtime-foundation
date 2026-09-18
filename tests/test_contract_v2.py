from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from runtime_foundation import (
    CONTRACT_V2_VERSION,
    CONTRACT_VERSION,
    ArtifactBindingV2,
    ContentIdentity,
    EngineBindingV2,
    EngineIdentity,
    ExecutionGuardV1,
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
from runtime_foundation.errors import UnsupportedGenerationSettingError
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
    assert trace_v2["artifact_binding"]["content_identity"]["scheme"] == "complete"
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


def test_v1_trace_serialization_is_not_upgraded_without_v2_evidence() -> None:
    artifact = ModelArtifactBinding("model", "C:/model.bin", "bin")
    legacy = artifact.to_dict()
    assert legacy["contract_version"] == CONTRACT_VERSION
