from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from runtime_foundation import (
    ExecutionConstraints,
    GenerationOptions,
    GenerationRequestV3,
    HostProfile,
    LoadOptions,
    LocalRuntimeClient,
    ModelArtifactBinding,
    RuntimeCore,
    ThinkingIntent,
    ThinkingMode,
)
from runtime_foundation.adapters.mock import MockAdapter
from runtime_foundation.errors import ContextLengthExceededError
from runtime_foundation.service import create_app


def _core_and_artifact(tmp_path):
    artifact_path = tmp_path / "test-model.bin"
    artifact_path.write_bytes(b"test-only-runtime-fixture")
    artifact = ModelArtifactBinding("test-model", str(artifact_path), "bin")
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"mock": MockAdapter()})
    return core, artifact


def _request(artifact, *, request_id="request-v3", max_context_tokens=None):
    return GenerationRequestV3(
        model_artifact_id=artifact.artifact_id,
        messages=[{"role": "user", "content": "hello there"}],
        generation_options=GenerationOptions(
            max_tokens=2,
            temperature=0,
            thinking_intent=ThinkingIntent(mode=ThinkingMode.OFF),
        ),
        execution_constraints=ExecutionConstraints(max_context_tokens=max_context_tokens),
        request_id=request_id,
        consumer_id="test-consumer",
    )


def test_v3_generation_returns_separate_evidence_and_versioned_fingerprint(tmp_path) -> None:
    core, artifact = _core_and_artifact(tmp_path)
    loaded = core.load(artifact, adapter="mock", load_options=LoadOptions(), consumer_id="test-consumer")

    result = core.generate_v3(
        _request(artifact),
    ).to_dict()

    assert loaded["raw"]["load_identity"]["fingerprint"].startswith("sha256:")
    assert result["contract_version"] == "runtime-foundation.contract.v3"
    evidence = result["execution_evidence"]
    assert evidence["execution_binding"]["schema_version"] == "runtime-foundation.execution-binding.v3"
    assert evidence["execution_binding_fingerprint"].startswith("sha256:")
    assert evidence["options"]["load"]["requested"] == LoadOptions().to_dict()
    assert evidence["options"]["generation"]["requested"]["max_tokens"] == 2
    assert evidence["options"]["generation"]["effective"]["thinking_intent"]["mode"] == "OFF"
    persisted = core.get_execution_v3(result["execution_id"])
    assert persisted is not None
    assert persisted["execution_evidence"]["execution_binding_fingerprint"] == evidence[
        "execution_binding_fingerprint"
    ]


def test_v3_context_budget_fails_before_generation_when_prompt_and_output_do_not_fit(tmp_path) -> None:
    core, artifact = _core_and_artifact(tmp_path)
    core.load(artifact, adapter="mock", load_options=LoadOptions(), consumer_id="test-consumer")

    with pytest.raises(ContextLengthExceededError) as caught:
        core.generate_v3(_request(artifact, request_id="too-large-v3", max_context_tokens=3))

    assert caught.value.details["generation_started"] is False
    assert caught.value.details["required_tokens"] > caught.value.details["max_context_tokens"]
    trace = core.get_execution_v3(caught.value.details["execution_id"])
    assert trace is not None and trace["status"] == "error"


def test_v3_streaming_events_carry_evidence_and_stable_ids(tmp_path) -> None:
    core, artifact = _core_and_artifact(tmp_path)
    core.load(artifact, adapter="mock", load_options=LoadOptions(), consumer_id="test-consumer")

    events = list(core.stream_v3(_request(artifact, request_id="stream-v3")))

    assert events[0].type == "started"
    assert events[0].evidence is not None
    assert all(event.request_id == "stream-v3" for event in events)
    assert all(event.execution_id == events[0].execution_id for event in events)
    terminal = events[-1].to_dict()
    assert terminal["type"] == "completed"
    assert terminal["done"] is True
    assert terminal["contract_version"] == "runtime-foundation.contract.v3"
    assert terminal["result"]["execution_evidence"]["execution_binding_fingerprint"] == events[0].evidence.execution_binding.fingerprint


def test_client_uses_v3_load_generate_and_stream_routes(tmp_path) -> None:
    core, artifact = _core_and_artifact(tmp_path)
    client = LocalRuntimeClient(http_client=TestClient(create_app(core)))

    loaded = client.load(artifact, engine="mock", consumer_id="test-consumer", load_options=LoadOptions())
    result = client.generate(_request(artifact))
    events = list(client.stream(_request(artifact, request_id="service-stream-v3")))

    assert loaded["contract_version"] == "runtime-foundation.contract.v3"
    assert loaded["load_options_evidence"]["effective"] == LoadOptions().to_dict()
    assert result["contract_version"] == "runtime-foundation.contract.v3"
    assert result["execution_evidence"]["execution_binding"]["schema_version"].endswith("execution-binding.v3")
    assert events[-1]["contract_version"] == "runtime-foundation.contract.v3"
