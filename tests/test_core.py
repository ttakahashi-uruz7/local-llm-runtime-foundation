from __future__ import annotations

from pathlib import Path

import pytest

from runtime_foundation import GenerationRequest, HostProfile, ModelArtifactBinding, RuntimeCore, RuntimeOptions
from runtime_foundation.errors import (
    ArtifactNotFoundError,
    EngineUnavailableError,
    UnsupportedRuntimeOptionError,
)


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
    assert result.metrics.measurement_provenance == "simulated"
    assert core.get_execution(result.execution_id)["engine"]["engine"] == "mock"
    assert core.health()["lifecycle_state"] == "LOADED"

    unloaded = core.unload(artifact.artifact_id)
    assert unloaded["unloaded"] is True
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
    assert core.health()["lifecycle_state"] == "LOADED"


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
