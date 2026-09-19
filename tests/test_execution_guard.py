from __future__ import annotations

from pathlib import Path

import pytest

from runtime_foundation import (
    ArtifactBindingV2,
    ArtifactLocator,
    BuildIdentityV1,
    ContentIdentity,
    ExecutionGuardV1,
    GenerationRequestV2,
    HostProfile,
    ModelArtifactBinding,
    RuntimeCore,
    RuntimeOptions,
    RuntimeSettingsBindingV2,
    ThinkingIntent,
    ThinkingMode,
)
from runtime_foundation.adapters.mock import MockAdapter
from runtime_foundation.errors import (
    ExecutionBindingUnresolvableError,
    ExecutionGuardMismatchError,
    InvalidRequestError,
)

TEST_FOUNDATION_BUILD_IDENTITY = BuildIdentityV1.from_components(
    kind="test-foundation-build-v1",
    components={"test": {"version": "1"}},
)


class CountingMockAdapter(MockAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.generate_calls = 0
        self.stream_calls = 0

    def generate(self, request, cancel_event):  # type: ignore[no-untyped-def]
        self.generate_calls += 1
        return super().generate(request, cancel_event)

    def stream(self, request, cancel_event):  # type: ignore[no-untyped-def]
        self.stream_calls += 1
        yield from super().stream(request, cancel_event)


def _complete_artifact(tmp_path: Path, *, artifact_id: str = "guarded") -> ArtifactBindingV2:
    model = tmp_path / f"{artifact_id}.bin"
    model.write_bytes(b"complete guard fixture")
    return ArtifactBindingV2(
        artifact_id=artifact_id,
        locator=ArtifactLocator("filesystem", str(model)),
        content_identity=ContentIdentity.from_file(model),
        format="bin",
    )


def _loaded_core(
    tmp_path: Path,
    artifact: ArtifactBindingV2 | ModelArtifactBinding,
    *,
    adapter: MockAdapter | None = None,
    foundation_build_identity: BuildIdentityV1 = TEST_FOUNDATION_BUILD_IDENTITY,
) -> tuple[RuntimeCore, MockAdapter]:
    selected = adapter or MockAdapter()
    core = RuntimeCore(
        host_profile=HostProfile.mock_windows(),
        adapters={"mock": selected},
        foundation_build_identity=foundation_build_identity,
    )
    core.load(artifact, adapter="mock")
    return core, selected


def _baseline_and_fingerprints(tmp_path: Path) -> tuple[RuntimeCore, ArtifactBindingV2, str, str]:
    artifact = _complete_artifact(tmp_path)
    core, _ = _loaded_core(tmp_path, artifact)
    baseline = core.generate(
        GenerationRequestV2(
            model_artifact_id=artifact.artifact_id,
            messages=[{"role": "user", "content": "baseline"}],
            thinking_intent=ThinkingIntent(mode=ThinkingMode.OFF),
        )
    )
    result_trace = baseline.trace
    assert result_trace is not None
    trace = result_trace.trace_v2
    assert trace is not None
    assert trace.runtime_settings_binding is not None
    assert trace.execution_binding_fingerprint is not None
    assert trace.runtime_settings_binding.exact_settings_fingerprint is not None
    return (
        core,
        artifact,
        trace.execution_binding_fingerprint,
        trace.runtime_settings_binding.exact_settings_fingerprint,
    )


def _request(
    artifact: ArtifactBindingV2, guard: ExecutionGuardV1, *, request_id: str = "guarded-request"
) -> GenerationRequestV2:
    return GenerationRequestV2(
        model_artifact_id=artifact.artifact_id,
        request_id=request_id,
        messages=[{"role": "user", "content": "guarded"}],
        thinking_intent=ThinkingIntent(mode=ThinkingMode.OFF),
        execution_guard=guard,
    )


def test_guard_match_allows_generate_and_records_pass_evidence(tmp_path: Path) -> None:
    core, artifact, execution_fingerprint, settings_fingerprint = _baseline_and_fingerprints(tmp_path)
    result = core.generate(
        _request(
            artifact,
            ExecutionGuardV1(
                expected_execution_binding_fingerprint=execution_fingerprint,
                expected_runtime_settings_fingerprint=settings_fingerprint,
            ),
        )
    )

    trace = result.to_dict()["generation_v2"]["trace"]
    assert trace["guard_verification"]["status"] == "MATCH"
    assert trace["guard_verification"]["generation_started"] is True
    assert trace["generation_started"] is True
    assert trace["guard_verification"]["actual_execution_binding_fingerprint"] == execution_fingerprint
    assert trace["guard_verification"]["actual_runtime_settings_fingerprint"] == settings_fingerprint


def test_guard_match_allows_stream_without_duplicate_guard_logic(tmp_path: Path) -> None:
    core, artifact, execution_fingerprint, settings_fingerprint = _baseline_and_fingerprints(tmp_path)
    events = list(
        core.stream(
            _request(
                artifact,
                ExecutionGuardV1(
                    expected_execution_binding_fingerprint=execution_fingerprint,
                    expected_runtime_settings_fingerprint=settings_fingerprint,
                ),
                request_id="guarded-stream",
            )
        )
    )
    assert events[0].type == "started"
    assert events[-1].type == "completed"
    assert all(event.type == "delta" for event in events[1:-1])
    completed = events[-1].result
    assert completed is not None
    assert completed["trace"]["trace_v2"]["guard_verification"]["status"] == "MATCH"


def test_runtime_settings_guard_uses_effective_settings_after_normalization(tmp_path: Path) -> None:
    artifact = _complete_artifact(tmp_path)
    core, _ = _loaded_core(tmp_path, artifact)
    requested = RuntimeOptions()
    requested_binding = RuntimeSettingsBindingV2.from_effective(requested)
    assert requested_binding is not None
    with pytest.raises(ExecutionGuardMismatchError) as caught:
        core.generate(
            GenerationRequestV2(
                model_artifact_id=artifact.artifact_id,
                messages=[{"role": "user", "content": "normalized"}],
                runtime_options=requested,
                thinking_intent=ThinkingIntent(mode=ThinkingMode.OFF),
                execution_guard=ExecutionGuardV1(
                    expected_runtime_settings_fingerprint=requested_binding.exact_settings_fingerprint,
                ),
            )
        )
    assert caught.value.details["mismatch_category"] == "runtime_settings_mismatch"
    assert caught.value.details["generation_started"] is False
    execution = core.get_execution(caught.value.details["execution_id"])
    assert execution is not None
    assert execution["trace_v2"]["guard_verification"]["status"] == "MISMATCH"
    assert execution["trace_v2"]["generation_started"] is False


def test_execution_binding_mismatch_blocks_generate_before_adapter(tmp_path: Path) -> None:
    artifact = _complete_artifact(tmp_path)
    adapter = CountingMockAdapter()
    core, _ = _loaded_core(tmp_path, artifact, adapter=adapter)
    with pytest.raises(ExecutionGuardMismatchError) as caught:
        core.generate(
            _request(
                artifact,
                ExecutionGuardV1(expected_execution_binding_fingerprint="sha256:" + "0" * 64),
            )
        )
    assert caught.value.details["mismatch_category"] == "execution_binding_mismatch"
    assert caught.value.details["generation_started"] is False
    assert adapter.generate_calls == 0
    assert adapter.stream_calls == 0
    execution = core.get_execution(caught.value.details["execution_id"])
    assert execution is not None
    assert execution["trace_v2"]["raw_execution_error"]["code"] == "execution_guard_mismatch"
    assert execution["trace_v2"]["guard_verification"]["generation_started"] is False


def test_stream_guard_failure_emits_only_terminal_error_and_never_calls_adapter(tmp_path: Path) -> None:
    artifact = _complete_artifact(tmp_path)
    adapter = CountingMockAdapter()
    core, _ = _loaded_core(tmp_path, artifact, adapter=adapter)
    events = list(
        core.stream(
            _request(
                artifact,
                ExecutionGuardV1(expected_execution_binding_fingerprint="sha256:" + "0" * 64),
                request_id="blocked-stream",
            )
        )
    )
    assert [event.type for event in events] == ["error"]
    assert events[0].error is not None
    assert events[0].error["code"] == "execution_guard_mismatch"
    assert adapter.generate_calls == 0
    assert adapter.stream_calls == 0
    execution_id = events[0].error["details"]["execution_id"]
    execution = core.get_execution(execution_id)
    assert execution is not None
    assert execution["trace_v2"]["generation_started"] is False


@pytest.mark.parametrize("identity_kind", ["fast", "legacy", "missing"])
def test_non_complete_artifact_identity_is_unresolvable_for_strict_execution_guard(
    tmp_path: Path, identity_kind: str
) -> None:
    model = tmp_path / f"{identity_kind}.bin"
    model.write_bytes(b"identity safety fixture")
    if identity_kind == "fast":
        artifact: ArtifactBindingV2 | ModelArtifactBinding = ArtifactBindingV2(
            "guarded",
            ArtifactLocator("filesystem", str(model)),
            ContentIdentity.from_file(model, scheme="fast"),
            format="bin",
        )
    elif identity_kind == "legacy":
        artifact = ModelArtifactBinding("guarded", str(model), "bin", artifact_hash="sha256:" + "a" * 64)
    else:
        artifact = ModelArtifactBinding("guarded", str(model), "bin")
    core, _ = _loaded_core(tmp_path, artifact)
    with pytest.raises(ExecutionBindingUnresolvableError) as caught:
        core.generate(
            GenerationRequestV2(
                model_artifact_id="guarded",
                messages=[{"role": "user", "content": "identity"}],
                thinking_intent=ThinkingIntent(mode=ThinkingMode.OFF),
                execution_guard=ExecutionGuardV1(expected_execution_binding_fingerprint="sha256:" + "b" * 64),
            )
        )
    assert caught.value.details["mismatch_category"] == "execution_binding_unresolvable"
    assert "artifact_binding.content_identity.complete" in caught.value.details["unresolvable_fields"]


def test_malformed_guard_is_invalid_request_and_preserved_in_trace(tmp_path: Path) -> None:
    artifact = _complete_artifact(tmp_path)
    core, _ = _loaded_core(tmp_path, artifact)
    with pytest.raises(InvalidRequestError) as caught:
        core.generate(
            _request(
                artifact,
                ExecutionGuardV1(expected_runtime_settings_fingerprint="sha256:not-a-fingerprint"),
            )
        )
    assert caught.value.details["mismatch_category"] == "invalid_expectation"
    execution = core.get_execution(caught.value.details["execution_id"])
    assert execution is not None
    verification = execution["trace_v2"]["guard_verification"]
    assert verification["status"] == "INVALID_EXPECTATION"
    assert verification["error"]["code"] == "invalid_request"
    assert execution["trace_v2"]["generation_started"] is False
