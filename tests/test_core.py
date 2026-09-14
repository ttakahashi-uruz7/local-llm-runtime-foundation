from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from runtime_foundation import GenerationRequest, HostProfile, ModelArtifactBinding, RuntimeCore, RuntimeOptions
from runtime_foundation.adapters.mock import MockAdapter
from runtime_foundation.errors import (
    ArtifactNotFoundError,
    ContextLengthExceededError,
    EngineNotFoundError,
    EngineUnavailableError,
    LoadConflictError,
    RuntimeTimeoutError,
    RequestCancelledError,
    UnsupportedRuntimeOptionError,
)


class CoreTimerCancellationAdapter(MockAdapter):
    """Adapter double that only reports cancellation after Core's timer fires."""

    def generate(self, request, cancel_event):
        del request
        assert cancel_event.wait(1.0), "Core timeout did not request adapter cancellation"
        raise RequestCancelledError("adapter observed Core timeout cancellation")


def build_core(tmp_path: Path) -> tuple[RuntimeCore, ModelArtifactBinding]:
    model = tmp_path / "mock-model.bin"
    model.write_bytes(b"foundation fixture")
    artifact = ModelArtifactBinding(artifact_id="fixture", local_path=str(model), format="bin")
    return RuntimeCore(host_profile=HostProfile.mock_windows()), artifact


def test_mock_lifecycle_generate_trace_and_unload(tmp_path: Path) -> None:
    core, artifact = build_core(tmp_path)
    loaded = core.load(artifact, adapter="mock")
    assert loaded["lifecycle_state"] == "LOADED"
    assert loaded["engine"]["engine"] == "mock"

    result = core.generate(
        GenerationRequest(
            model_artifact_id=artifact.artifact_id,
            messages=[{"role": "user", "content": "hello foundation"}],
        )
    )
    assert result.engine == "mock"
    assert result.adapter == "mock"
    assert result.text
    assert result.execution_id
    assert result.trace is not None
    assert result.trace.status == "completed"
    assert result.requested_runtime_settings["acceleration"]["backend"] == "auto"
    assert result.effective_runtime_settings["acceleration"]["backend"] == "cpu"
    assert result.runtime_settings_resolution is not None
    assert result.runtime_settings_resolution.to_dict()["option_status"]
    assert result.runtime_settings_resolution.to_dict()["warnings"]
    assert result.trace.to_dict()["runtime_settings_resolution"] == result.runtime_settings_resolution.to_dict()
    assert result.metrics.measurement_provenance == "simulated"
    assert core.get_execution(result.execution_id)["engine"]["engine"] == "mock"
    assert core.health()["lifecycle_state"] == "LOADED"

    unloaded = core.unload(artifact.artifact_id)
    assert unloaded["unloaded"] is True
    assert unloaded["raw"]["cleanup_status"] == "clean"
    assert core.health()["lifecycle_state"] == "UNLOADED"


def test_missing_artifact_and_engine_unavailable_are_explicit(tmp_path: Path) -> None:
    core, artifact = build_core(tmp_path)
    missing = ModelArtifactBinding("missing", str(tmp_path / "missing.bin"), "bin")
    with pytest.raises(ArtifactNotFoundError):
        core.load(missing, adapter="mock")
    with pytest.raises(EngineUnavailableError):
        core.load(artifact, adapter="mlx")
    assert core.health()["loaded_artifact_id"] is None
    assert core.health()["status"] == "error"


def test_omitted_engine_never_falls_back_to_mock(tmp_path: Path) -> None:
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"mlx-shaped fixture")
    mlx_artifact = ModelArtifactBinding("mlx-artifact", str(model), "safetensors")
    core = RuntimeCore(host_profile=HostProfile.mock_windows())

    with pytest.raises(EngineUnavailableError) as unavailable:
        core.load(mlx_artifact)
    assert unavailable.value.details["engine"] == "mlx"
    assert unavailable.value.details["engine"] != "mock"

    unknown = replace(mlx_artifact, artifact_id="unknown", format="unknown")
    with pytest.raises(EngineNotFoundError):
        core.load(unknown)

    explicit_mock = RuntimeCore(host_profile=HostProfile.mock_windows())
    loaded = explicit_mock.load(mlx_artifact, adapter="mock")
    assert loaded["engine"]["engine"] == "mock"


def test_artifact_reuse_requires_full_execution_identity(tmp_path: Path) -> None:
    core, artifact = build_core(tmp_path)
    core.load(artifact, adapter="mock")
    alternate_path = tmp_path / "other-model.bin"
    alternate_path.write_bytes(b"different fixture")

    variants = (
        replace(artifact, artifact_hash="sha256:other"),
        replace(artifact, local_path=str(alternate_path)),
        replace(artifact, revision="revision-2"),
    )
    for variant in variants:
        with pytest.raises(LoadConflictError):
            core.load(variant, adapter="mock")

    loaded_binding = replace(artifact, metadata={"authority": "loaded"})
    fresh = RuntimeCore(host_profile=HostProfile.mock_windows())
    fresh.load(loaded_binding, adapter="mock")
    requested_with_new_metadata = replace(loaded_binding, metadata={"authority": "request"})
    reused = fresh.load(requested_with_new_metadata, adapter="mock")
    assert reused["reused"] is True
    assert reused["artifact"] == loaded_binding.to_dict()
    assert reused["raw"]["loaded_artifact_identity"] == loaded_binding.execution_identity()


def test_context_length_budget_is_enforced(tmp_path: Path) -> None:
    core, artifact = build_core(tmp_path)
    core.load(artifact, adapter="mock")
    request = GenerationRequest(
        model_artifact_id=artifact.artifact_id,
        messages=[{"role": "user", "content": "one two"}],
        max_tokens=3,
        runtime_options=RuntimeOptions.from_payload({"context": {"context_length": 4}}),
    )
    with pytest.raises(ContextLengthExceededError) as caught:
        core.generate(request)
    assert caught.value.details["requested_total_tokens"] == 5
    assert caught.value.details["enforcement"] == "preflight"


def test_timeout_is_cooperative_and_reaches_runtime_timeout(tmp_path: Path) -> None:
    core, artifact = build_core(tmp_path)
    core.load(artifact, adapter="mock")
    request = GenerationRequest(
        model_artifact_id=artifact.artifact_id,
        messages=[{"role": "user", "content": "timeout this generated response"}],
        timeout_ms=5,
        runtime_options=RuntimeOptions.from_payload({"engine_options": {"mock.chunk_delay_ms": 20}}),
    )
    with pytest.raises(RuntimeTimeoutError) as caught:
        core.generate(request)
    assert caught.value.details["timeout_semantics"] == "cooperative"
    assert core.health()["lifecycle_state"] == "LOADED"


def test_core_timeout_authority_overrides_adapter_cancellation(tmp_path: Path) -> None:
    model = tmp_path / "timeout-double.bin"
    model.write_bytes(b"timeout fixture")
    artifact = ModelArtifactBinding("timeout-double", str(model), "bin")
    core = RuntimeCore(
        host_profile=HostProfile.mock_windows(),
        adapters={"mock": CoreTimerCancellationAdapter()},
    )
    core.load(artifact, adapter="mock")

    with pytest.raises(RuntimeTimeoutError) as caught:
        core.generate(
            GenerationRequest(
                model_artifact_id=artifact.artifact_id,
                messages=[{"role": "user", "content": "Core timeout authority"}],
                timeout_ms=5,
            )
        )

    assert caught.value.code == "runtime_timeout"
    assert caught.value.__cause__ is not None
    assert caught.value.__cause__.code == "cancelled"
    execution_id = caught.value.details["execution_id"]
    execution = core.get_execution(execution_id)
    assert execution is not None
    assert execution["error"]["code"] == "runtime_timeout"
    assert core.health()["last_error"]["code"] == "runtime_timeout"


def test_requested_and_effective_options_and_unsupported_option(tmp_path: Path) -> None:
    core, artifact = build_core(tmp_path)
    core.load(artifact, adapter="mock")
    options = RuntimeOptions.from_payload({"context": {"context_length": 8192}})
    result = core.generate(
        GenerationRequest(
            model_artifact_id=artifact.artifact_id,
            messages=[{"role": "user", "content": "options"}],
            runtime_options=options,
        )
    )
    assert result.requested_runtime_settings["context"]["context_length"] == 8192
    assert result.effective_runtime_settings["context"]["context_length"] == 8192
    assert result.effective_runtime_settings["prefill"]["chunk_size"] == 128

    bad = RuntimeOptions.from_payload({"acceleration": {"backend": "metal"}})
    with pytest.raises(UnsupportedRuntimeOptionError) as caught:
        core.generate(
            GenerationRequest(
                model_artifact_id=artifact.artifact_id,
                messages=[{"role": "user", "content": "bad"}],
                runtime_options=bad,
            )
        )
    assert caught.value.details["execution_id"]
    failed_trace = core.get_execution(caught.value.details["execution_id"])
    assert failed_trace["runtime_settings_resolution"] is None
    assert failed_trace["effective_runtime_settings"] is None
    assert core.health()["lifecycle_state"] == "LOADED"

    with pytest.raises(UnsupportedRuntimeOptionError) as prompt_cache_error:
        core.generate(
            GenerationRequest(
                model_artifact_id=artifact.artifact_id,
                messages=[{"role": "user", "content": "prompt cache"}],
                runtime_options=RuntimeOptions.from_payload({"prompt_cache": {"enabled": True}}),
            )
        )
    assert prompt_cache_error.value.details["path"] == "prompt_cache.enabled"


def test_stream_and_cancel_leave_loaded_state_consistent(tmp_path: Path) -> None:
    core, artifact = build_core(tmp_path)
    core.load(artifact, adapter="mock")
    request = GenerationRequest(
        model_artifact_id=artifact.artifact_id,
        request_id="cancel-me",
        messages=[{"role": "user", "content": "cancel this stream"}],
        runtime_options=RuntimeOptions.from_payload({"engine_options": {"mock.chunk_delay_ms": 10}}),
    )
    events = core.stream(request)
    assert next(events).type == "started"
    assert core.cancel(request.request_id)["cancelled"] is True
    remainder = list(events)
    assert remainder[-1].type == "error"
    assert remainder[-1].error["code"] == "cancelled"
    assert core.health()["lifecycle_state"] == "LOADED"
    execution_id = remainder[-1].error["details"]["execution_id"]
    assert core.get_execution(execution_id)["status"] == "cancelled"


def test_stream_completed_result_exposes_same_runtime_resolution_as_trace(tmp_path: Path) -> None:
    core, artifact = build_core(tmp_path)
    core.load(artifact, adapter="mock")
    events = list(
        core.stream(
            GenerationRequest(
                model_artifact_id=artifact.artifact_id,
                request_id="completed-stream",
                messages=[{"role": "user", "content": "stream resolution"}],
            )
        )
    )
    completed = next(event for event in events if event.type == "completed")
    resolution = completed.result["runtime_settings_resolution"]
    assert resolution["option_status"]
    assert resolution["warnings"]
    assert completed.result["trace"]["runtime_settings_resolution"] == resolution


def test_consumer_leases_prevent_cross_consumer_unload_and_switch(tmp_path: Path) -> None:
    core, artifact = build_core(tmp_path)
    first = core.load(artifact, adapter="mock", consumer_id="benchmark")
    second = core.load(artifact, adapter="mock", consumer_id="novel")
    assert second["reused"] is True
    released = core.unload(artifact.artifact_id, consumer_id="benchmark", lease_id=first["lease_id"])
    assert released["unloaded"] is False
    with pytest.raises(Exception):
        core.unload(artifact.artifact_id, consumer_id="unknown")
    unloaded = core.unload(artifact.artifact_id, consumer_id="novel", lease_id=second["lease_id"])
    assert unloaded["unloaded"] is True
