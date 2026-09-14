from __future__ import annotations

import pytest

from runtime_foundation import (
    CONTRACT_VERSION,
    GenerationRequest,
    ModelArtifactBinding,
    RuntimeErrorRecord,
    RuntimeOptions,
)
from runtime_foundation.errors import RuntimeOptionsAliasConflictError, RuntimeOptionsSchemaMismatchError


def test_versioned_runtime_options_round_trip_and_effective_fields_are_separate() -> None:
    options = RuntimeOptions.from_payload(
        {
            "schema_version": "runtime-foundation.runtime-options.v1",
            "context": {"context_length": 32768},
            "kv_cache": {"mode": "quantized", "bits": 8, "group_size": 64},
            "prefill": {"chunk_size": 512},
            "acceleration": {"backend": "auto"},
        }
    )

    payload = options.to_dict()
    assert payload["contract_version"] == CONTRACT_VERSION
    assert RuntimeOptions.from_payload(payload).to_dict() == payload
    assert payload["kv_cache"]["mode"] == "quantized"


def test_contract_rejects_schema_alias_conflicts_and_unknown_fields() -> None:
    with pytest.raises(RuntimeOptionsSchemaMismatchError):
        RuntimeOptions.from_payload({"schema_version": "runtime-foundation.runtime-options.v0"})
    with pytest.raises(RuntimeOptionsAliasConflictError):
        RuntimeOptions.from_payload({"kv": {}, "kv_cache": {}})
    with pytest.raises(ValueError, match="unsupported fields"):
        RuntimeOptions.from_payload({"unexpected": True})


def test_generation_request_and_artifact_binding_are_json_safe() -> None:
    artifact = ModelArtifactBinding(
        artifact_id="fixture-model",
        local_path="C:/fixtures/model.safetensors",
        format="safetensors",
        quantization="int8",
        artifact_hash="sha256:fixture",
        revision="rev-1",
    )
    request = GenerationRequest(
        model_artifact_id=artifact.artifact_id,
        messages=[{"role": "user", "content": "hello"}],
        consumer_id="benchmark",
        runtime_options=RuntimeOptions.from_payload({"context": {"context_length": 2048}}),
    )
    assert ModelArtifactBinding.from_payload(artifact.to_dict()).to_dict() == artifact.to_dict()
    round_trip = GenerationRequest.from_payload(request.to_dict())
    assert round_trip.to_dict() == request.to_dict()
    assert round_trip.consumer_id == "benchmark"


def test_runtime_error_contract_has_no_policy_fields() -> None:
    error = RuntimeErrorRecord(code="engine_unavailable", message="not installed", details={"engine": "mlx"})
    payload = error.to_dict()
    assert payload["contract_version"] == CONTRACT_VERSION
    assert "production_eligible" not in payload
    assert "score" not in payload
