"""Immutable wire/fingerprint baseline captured before the dual-runtime work."""

from runtime_foundation import (
    CONTRACT_V2_VERSION,
    ArtifactBindingV2,
    EngineBindingV2,
    EngineIdentity,
    ExecutionBindingV2,
    FoundationBindingV2,
    GenerationRequest,
    ModelArtifactBinding,
    RuntimeOptions,
    RuntimeSettingsBindingV2,
    canonical_fingerprint,
)


def test_v1_runtime_options_and_generation_payload_baseline() -> None:
    options = RuntimeOptions.from_payload(
        {
            "context": {"context_length": 2048},
            "kv_cache": {"mode": "full_precision"},
            "prefill": {"chunk_size": 256, "batch_size": 1},
            "acceleration": {"backend": "auto"},
        }
    )
    assert options.to_dict() == {
        "contract_version": "runtime-foundation.contract.v1",
        "schema_version": "runtime-foundation.runtime-options.v1",
        "context": {"context_length": 2048, "sliding_window": None},
        "kv_cache": {
            "mode": "full_precision",
            "precision": None,
            "bits": None,
            "group_size": None,
            "quantization_start": None,
            "max_size_tokens": None,
            "cache_limit_bytes": None,
        },
        "prefill": {"chunk_size": 256, "batch_size": 1},
        "prompt_cache": {"enabled": False, "max_entries": None, "max_size_tokens": None},
        "acceleration": {"backend": "auto", "device": None, "threads": None},
        "engine_options": {},
    }
    assert canonical_fingerprint(options.to_dict()) == (
        "sha256:46c5d0129a24720d5cdb7586861138d80c2368702af75b9eec63cfae1d4e56b0"
    )

    request = GenerationRequest.from_payload(
        {
            "model_artifact_id": "baseline-model",
            "messages": [{"role": "user", "content": "hello"}],
            "request_id": "baseline-request",
        }
    )
    assert request.to_dict() == {
        "contract_version": "runtime-foundation.contract.v1",
        "schema_version": "runtime-foundation.generation-request.v1",
        "model_artifact_id": "baseline-model",
        "messages": [{"role": "user", "content": "hello"}],
        "request_id": "baseline-request",
        "consumer_id": None,
        "lease_id": None,
        "max_tokens": 64,
        "temperature": 0.0,
        "top_p": None,
        "thinking_enabled": None,
        "timeout_ms": None,
        "runtime_options": RuntimeOptions().to_dict(),
        "metadata": {},
    }


def test_v2_execution_binding_v1_fingerprint_baseline() -> None:
    options = RuntimeOptions.from_payload(
        {
            "context": {"context_length": 2048},
            "kv_cache": {"mode": "full_precision"},
            "prefill": {"chunk_size": 256, "batch_size": 1},
            "acceleration": {"backend": "auto"},
        }
    )
    artifact = ArtifactBindingV2.from_legacy(
        ModelArtifactBinding(
            artifact_id="baseline-model",
            local_path="baseline/model.gguf",
            format="gguf",
            quantization="Q4_K_M",
            artifact_hash="sha256:" + "a" * 64,
            revision="r1",
            metadata={"source": "fixture"},
        )
    )
    engine = EngineBindingV2.from_legacy(
        EngineIdentity(engine="mock", version="0.1-mock", build="foundation-mock"),
        adapter_id="mock",
    )
    foundation = FoundationBindingV2(
        contract_version=CONTRACT_V2_VERSION,
        foundation_version="0.2.0",
        build_identity=None,
        adapter_id="mock",
    )
    binding = ExecutionBindingV2(
        artifact_binding=artifact,
        engine_binding=engine,
        foundation_binding=foundation,
        runtime_settings_binding=RuntimeSettingsBindingV2.from_effective(options),
    )
    assert binding.fingerprint == "sha256:18cdf3db4164a185dedf1b37e0392601e16ede2ac8f88a3c738e14c54b279090"
    assert ExecutionBindingV2.from_payload(binding.to_dict()).fingerprint == binding.fingerprint
