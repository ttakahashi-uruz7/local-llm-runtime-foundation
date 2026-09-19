from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from runtime_foundation import (
    CONTRACT_V2_VERSION,
    ArtifactBindingV2,
    ArtifactLocator,
    BuildIdentityV1,
    ContentIdentity,
    EngineBindingV2,
    EngineIdentity,
    ExecutionBindingV2,
    ExecutionGuardV1,
    FoundationBindingV2,
    GenerationRequestV2,
    HostProfile,
    RuntimeCore,
    RuntimeOptions,
    RuntimeSettingsBindingV2,
    ThinkingIntent,
    ThinkingMode,
)
from runtime_foundation.adapters.mlx import MLXAdapter
from runtime_foundation.adapters.mock import MockAdapter
from runtime_foundation.build_identity import (
    FOUNDATION_BUILD_IDENTITY_KIND,
    build_identity_from_package_tree,
    foundation_package_tree_components,
)
from runtime_foundation.errors import ExecutionBindingUnresolvableError, ExecutionGuardMismatchError

TEST_FOUNDATION_BUILD = BuildIdentityV1.from_components(
    kind="test-foundation-build-v1",
    components={"test": {"version": "1"}},
)


def _complete_artifact(tmp_path: Path) -> ArtifactBindingV2:
    model = tmp_path / "model.bin"
    model.write_bytes(b"complete build identity fixture")
    return ArtifactBindingV2(
        artifact_id="guarded",
        locator=ArtifactLocator("filesystem", str(model)),
        content_identity=ContentIdentity.from_file(model),
        format="bin",
    )


def _guarded_request(artifact: ArtifactBindingV2, fingerprint: str) -> GenerationRequestV2:
    return GenerationRequestV2(
        model_artifact_id=artifact.artifact_id,
        messages=[{"role": "user", "content": "guarded build identity"}],
        thinking_intent=ThinkingIntent(mode=ThinkingMode.OFF),
        execution_guard=ExecutionGuardV1(expected_execution_binding_fingerprint=fingerprint),
    )


def _baseline(tmp_path: Path) -> tuple[ArtifactBindingV2, str]:
    artifact = _complete_artifact(tmp_path)
    core = RuntimeCore(
        host_profile=HostProfile.mock_windows(),
        adapters={"mock": MockAdapter()},
        foundation_build_identity=TEST_FOUNDATION_BUILD,
    )
    core.load(artifact, adapter="mock")
    result = core.generate(
        GenerationRequestV2(
            model_artifact_id=artifact.artifact_id,
            messages=[{"role": "user", "content": "baseline"}],
            thinking_intent=ThinkingIntent(mode=ThinkingMode.OFF),
        )
    )
    assert result.trace is not None and result.trace.trace_v2 is not None
    return artifact, result.trace.trace_v2.execution_binding_fingerprint


def test_build_identity_is_canonical_and_validated() -> None:
    first = BuildIdentityV1.from_components(kind="test-build-v1", components={"z": 1, "a": {"version": "1.0"}})
    second = BuildIdentityV1.from_components(kind="test-build-v1", components={"a": {"version": "1.0"}, "z": 1})
    assert first.fingerprint == second.fingerprint
    assert BuildIdentityV1.from_payload(first.to_dict()) == first
    changed = BuildIdentityV1.from_components(kind="test-build-v1", components={"a": {"version": "1.1"}, "z": 1})
    assert changed.fingerprint != first.fingerprint
    with pytest.raises(ValueError, match="does not match canonical"):
        BuildIdentityV1(kind=first.kind, fingerprint="sha256:" + "0" * 64, components=first.components)


def test_foundation_tree_identity_is_path_independent_and_content_sensitive(tmp_path: Path) -> None:
    first_root = tmp_path / "first" / "runtime_foundation"
    second_root = tmp_path / "second" / "runtime_foundation"
    for root in (first_root, second_root):
        (root / "nested").mkdir(parents=True)
        (root / "__pycache__").mkdir()
        (root / "alpha.py").write_text("ALPHA = 1\n", encoding="utf-8")
        (root / "nested" / "beta.py").write_text("BETA = 2\n", encoding="utf-8")
        (root / "__pycache__" / "ignored.py").write_text("ignored = True\n", encoding="utf-8")

    first = build_identity_from_package_tree(first_root)
    second = build_identity_from_package_tree(second_root)
    assert first.kind == FOUNDATION_BUILD_IDENTITY_KIND
    assert first.fingerprint == second.fingerprint
    assert all("__pycache__" not in item["path"] for item in first.components["files"])

    (second_root / "nested" / "beta.py").write_text("BETA = 3\n", encoding="utf-8")
    assert build_identity_from_package_tree(second_root).fingerprint != first.fingerprint

    components = foundation_package_tree_components(first_root)
    assert [item["path"] for item in components["files"]] == ["alpha.py", "nested/beta.py"]


def test_runtime_core_observes_and_exposes_foundation_and_engine_build_identities(tmp_path: Path) -> None:
    core = RuntimeCore()
    assert core.foundation_build_identity is not None
    assert core.foundation_build_identity.kind == FOUNDATION_BUILD_IDENTITY_KIND
    assert core.foundation_build_identity.fingerprint != core.foundation_version
    assert core.health()["foundation_build_identity"]["fingerprint"] == core.foundation_build_identity.fingerprint

    artifact = _complete_artifact(tmp_path)
    core.load(artifact, adapter="mock")
    health = core.health()
    assert health["engine_build_identity"]["kind"] == "mock-runtime-build-v1"
    assert core.runtime_metrics()["engine_build_identity"]["kind"] == "mock-runtime-build-v1"
    capability = core.adapters["mock"].discover_capability().to_dict()
    assert capability["build_identity"]["kind"] == "mock-runtime-build-v1"


def test_foundation_build_identity_override_is_structured_and_rejects_version_strings() -> None:
    core = RuntimeCore(
        host_profile=HostProfile.mock_windows(),
        adapters={"mock": MockAdapter()},
        foundation_build_identity=TEST_FOUNDATION_BUILD,
    )
    assert core.foundation_build_identity == TEST_FOUNDATION_BUILD
    with pytest.raises(ValueError, match="validated BuildIdentityV1"):
        RuntimeCore(  # type: ignore[arg-type]
            host_profile=HostProfile.mock_windows(),
            adapters={"mock": MockAdapter()},
            foundation_build_identity="0.2.0",
        )
    malformed = TEST_FOUNDATION_BUILD.to_dict()
    malformed["fingerprint"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="does not match canonical"):
        RuntimeCore(
            host_profile=HostProfile.mock_windows(),
            adapters={"mock": MockAdapter()},
            foundation_build_identity=malformed,
        )


def test_mlx_build_identity_observes_both_distributions_and_changes_with_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions: dict[str, str | None] = {"mlx-lm": "0.20.0", "mlx": "0.28.1"}
    adapter = MLXAdapter()
    monkeypatch.setattr(
        adapter, "_modules", lambda: (SimpleNamespace(__version__=None), SimpleNamespace(__version__=None), None)
    )
    monkeypatch.setattr(adapter, "_version", lambda distribution: versions[distribution])

    identity = adapter.build_identity()
    assert identity is not None
    assert identity.kind == "python-distribution-set-v1"
    assert identity.components == {"mlx-lm": {"version": "0.20.0"}, "mlx": {"version": "0.28.1"}}
    assert adapter.identity().version == "0.20.0"
    assert adapter.identity().build == "0.28.1"

    versions["mlx-lm"] = "0.20.1"
    changed_mlx_lm = adapter.build_identity()
    assert changed_mlx_lm is not None and changed_mlx_lm.fingerprint != identity.fingerprint
    versions["mlx-lm"] = "0.20.0"
    versions["mlx"] = "0.28.2"
    changed_mlx = adapter.build_identity()
    assert changed_mlx is not None and changed_mlx.fingerprint != identity.fingerprint
    versions["mlx"] = None
    assert adapter.build_identity() is None


def test_execution_binding_fingerprint_is_sensitive_to_engine_and_foundation_build_identity(tmp_path: Path) -> None:
    artifact = _complete_artifact(tmp_path)
    settings = RuntimeSettingsBindingV2.from_effective(RuntimeOptions())
    assert settings is not None
    engine_a = BuildIdentityV1.from_components(kind="engine-v1", components={"version": "a"})
    engine_b = BuildIdentityV1.from_components(kind="engine-v1", components={"version": "b"})
    foundation_a = BuildIdentityV1.from_components(kind="foundation-v1", components={"version": "a"})
    foundation_b = BuildIdentityV1.from_components(kind="foundation-v1", components={"version": "b"})

    def binding(engine_identity: BuildIdentityV1, foundation_identity: BuildIdentityV1) -> ExecutionBindingV2:
        return ExecutionBindingV2(
            artifact_binding=artifact,
            engine_binding=EngineBindingV2.from_legacy(
                EngineIdentity(engine="mock", version="0.1-mock"),
                adapter_id="mock",
                build_identity=engine_identity,
            ),
            foundation_binding=FoundationBindingV2(
                contract_version=CONTRACT_V2_VERSION,
                foundation_version="0.2.0",
                build_identity=foundation_identity,
                adapter_id="mock",
            ),
            runtime_settings_binding=settings,
        )

    baseline = binding(engine_a, foundation_a)
    assert binding(engine_b, foundation_a).fingerprint != baseline.fingerprint
    assert binding(engine_a, foundation_b).fingerprint != baseline.fingerprint


class _VariantMockAdapter(MockAdapter):
    def __init__(self, identity: BuildIdentityV1 | None) -> None:
        super().__init__()
        self._identity = identity

    def build_identity(self) -> BuildIdentityV1 | None:
        return self._identity


class _CountingMockAdapter(MockAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.generate_calls = 0

    def generate(self, request, cancel_event):  # type: ignore[no-untyped-def]
        self.generate_calls += 1
        return super().generate(request, cancel_event)


def test_guard_matching_complete_artifact_and_build_identities_then_rejects_mismatch(tmp_path: Path) -> None:
    artifact, expected = _baseline(tmp_path)
    counting = _CountingMockAdapter()
    matching = RuntimeCore(
        host_profile=HostProfile.mock_windows(),
        adapters={"mock": counting},
        foundation_build_identity=TEST_FOUNDATION_BUILD,
    )
    matching.load(artifact, adapter="mock")
    result = matching.generate(_guarded_request(artifact, expected))
    assert result.trace is not None and result.trace.trace_v2 is not None
    trace = result.trace.trace_v2.to_dict()
    assert trace["guard_verification"]["status"] == "MATCH"
    assert trace["generation_started"] is True
    assert counting.generate_calls == 1
    assert trace["engine_binding"]["build_identity"]["kind"] == "mock-runtime-build-v1"
    assert trace["foundation_binding"]["build_identity"]["kind"] == "test-foundation-build-v1"

    changed_engine = BuildIdentityV1.from_components(
        kind="mock-runtime-build-v1", components={"mock-runtime": {"version": "0.2-mock"}}
    )
    mismatched = RuntimeCore(
        host_profile=HostProfile.mock_windows(),
        adapters={"mock": _VariantMockAdapter(changed_engine)},
        foundation_build_identity=TEST_FOUNDATION_BUILD,
    )
    mismatched.load(artifact, adapter="mock")
    with pytest.raises(ExecutionGuardMismatchError):
        mismatched.generate(_guarded_request(artifact, expected))

    changed_foundation = BuildIdentityV1.from_components(
        kind="test-foundation-build-v1", components={"test": {"version": "2"}}
    )
    mismatched_foundation = RuntimeCore(
        host_profile=HostProfile.mock_windows(),
        adapters={"mock": MockAdapter()},
        foundation_build_identity=changed_foundation,
    )
    mismatched_foundation.load(artifact, adapter="mock")
    with pytest.raises(ExecutionGuardMismatchError):
        mismatched_foundation.generate(_guarded_request(artifact, expected))


def test_guard_unknown_engine_or_foundation_build_is_unresolvable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact, expected = _baseline(tmp_path)
    unknown_engine = RuntimeCore(
        host_profile=HostProfile.mock_windows(),
        adapters={"mock": _VariantMockAdapter(None)},
        foundation_build_identity=TEST_FOUNDATION_BUILD,
    )
    unknown_engine.load(artifact, adapter="mock")
    with pytest.raises(ExecutionBindingUnresolvableError) as engine_error:
        unknown_engine.generate(_guarded_request(artifact, expected))
    assert "engine_binding.build_identity" in engine_error.value.details["unresolvable_fields"]

    monkeypatch.setattr("runtime_foundation.core.observe_foundation_build_identity", lambda: None)
    unknown_foundation = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"mock": MockAdapter()})
    unknown_foundation.load(artifact, adapter="mock")
    with pytest.raises(ExecutionBindingUnresolvableError) as foundation_error:
        unknown_foundation.generate(_guarded_request(artifact, expected))
    assert "foundation_binding.build_identity" in foundation_error.value.details["unresolvable_fields"]
