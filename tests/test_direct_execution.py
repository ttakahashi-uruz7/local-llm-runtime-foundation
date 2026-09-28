from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from runtime_foundation import (
    ADAPTER_LINEAGE_METADATA_KEY,
    ArtifactBindingV2,
    ArtifactLocator,
    AdapterLineageV1,
    AdapterTargetBaseV1,
    ContentIdentity,
    ExecutionInputV1,
    GenerationRequest,
    GenerationRequestV2,
    HostProfile,
    LocalRuntimeClient,
    ModelArtifactBinding,
    RuntimeCore,
    ThinkingIntent,
    ThinkingMode,
)
from runtime_foundation.client import RemoteRuntimeError
from runtime_foundation.adapters.mlx import MLXAdapter
from runtime_foundation.adapters.mock import MockAdapter
from runtime_foundation.errors import (
    ArtifactNotFoundError,
    ArtifactCompatibilityError,
    ExecutionInputMismatchError,
    LoadConflictError,
    UnsupportedExecutionInputError,
)
from runtime_foundation.service import create_app


MODULE = "self_attn.q_proj"
MODULE_SHAPE = {"in_features": 4, "out_features": 6}
FACTOR_SHAPES = {MODULE: {"lora_a": [4, 2], "lora_b": [2, 6]}}


def _base(root: Path, name: str = "base", contents: bytes = b"base fixture") -> ArtifactBindingV2:
    path = root / f"{name}.safetensors"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
    return ArtifactBindingV2(
        artifact_id=name,
        locator=ArtifactLocator("filesystem", str(path)),
        content_identity=ContentIdentity.from_file(path),
        format="mlx",
        quantization="8-bit-g64",
        revision="base-revision-1",
        metadata={
            "model_identity": "vendor/model-family-v1",
            "tokenizer_identity": "vendor/tokenizer-v1",
            "target_module_shapes": {MODULE: MODULE_SHAPE},
        },
    )


def _adapter(
    root: Path,
    base: ArtifactBindingV2,
    name: str = "adapter-a",
    *,
    target_base: AdapterTargetBaseV1 | None = None,
    rank: int = 2,
    target_modules: tuple[str, ...] = (MODULE,),
    tensor_shapes: dict[str, dict[str, list[int]]] | None = None,
) -> ArtifactBindingV2:
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(
        json.dumps({"fine_tune_type": "lora", "lora_parameters": {"rank": rank, "keys": list(target_modules)}}),
        encoding="utf-8",
    )
    (path / "adapters.safetensors").write_bytes(f"{name} tensor fixture".encode())
    target = target_base or AdapterTargetBaseV1(
        artifact_id=base.artifact_id,
        content_identity=base.content_identity,
        revision=base.revision,
        format=base.format,
        quantization=base.quantization,
        model_identity=base.metadata["model_identity"],
        tokenizer_identity=base.metadata["tokenizer_identity"],
        locator=base.locator,
    )
    lineage = AdapterLineageV1(
        target_base=target,
        rank=rank,
        target_modules=target_modules,
        tensor_shapes=tensor_shapes or FACTOR_SHAPES,
    )
    return ArtifactBindingV2(
        artifact_id=name,
        locator=ArtifactLocator("filesystem", str(path)),
        content_identity=ContentIdentity.from_file(path),
        format="mlx-lora",
        quantization=None,
        revision=f"{name}-revision-1",
        metadata={ADAPTER_LINEAGE_METADATA_KEY: lineage.to_dict()},
    )


def _input(root: Path, name: str = "adapter-a", *, base_contents: bytes = b"base fixture") -> ExecutionInputV1:
    base = _base(root, contents=base_contents)
    return ExecutionInputV1(base=base, adapters=(_adapter(root, base, name),))


def _generate_request(
    execution_input: ExecutionInputV1, *, consumer_id: str | None = None, lease_id: str | None = None
) -> GenerationRequestV2:
    return GenerationRequestV2(
        model_artifact_id=execution_input.base.artifact_id,
        messages=[{"role": "user", "content": "direct composition fixture"}],
        consumer_id=consumer_id,
        lease_id=lease_id,
        execution_input=execution_input,
        thinking_intent=ThinkingIntent(mode=ThinkingMode.OFF),
    )


def test_execution_input_contract_round_trips_and_allows_zero_or_one_adapter(tmp_path: Path) -> None:
    direct = _input(tmp_path)
    restored = ExecutionInputV1.from_payload(direct.to_dict())
    assert restored.fingerprint == direct.fingerprint
    assert len(restored.adapters) == 1

    base_only = ExecutionInputV1(base=direct.base, adapters=())
    assert ExecutionInputV1.from_payload(base_only.to_dict()).fingerprint == base_only.fingerprint
    assert base_only.fingerprint != direct.fingerprint
    base_metadata = {key: value for key, value in direct.base.metadata.items() if key != "target_module_shapes"}
    base_without_adapter_manifest = ArtifactBindingV2(
        artifact_id=direct.base.artifact_id,
        locator=direct.base.locator,
        content_identity=direct.base.content_identity,
        format=direct.base.format,
        quantization=direct.base.quantization,
        revision=direct.base.revision,
        metadata=base_metadata,
    )
    assert ExecutionInputV1(base_without_adapter_manifest).adapters == ()


def test_execution_input_rejects_unsupported_adapter_count_and_malformed_payload(tmp_path: Path) -> None:
    direct = _input(tmp_path)
    two = direct.to_dict()
    two["adapters"].append(direct.adapters[0].to_dict())
    with pytest.raises(UnsupportedExecutionInputError):
        ExecutionInputV1.from_payload(two)

    malformed = direct.to_dict()
    malformed["schema_version"] = "runtime-foundation.execution-input.v9"
    with pytest.raises(ValueError, match="schema_version"):
        ExecutionInputV1.from_payload(malformed)

    missing_lineage_version = direct.to_dict()
    del missing_lineage_version["adapters"][0]["metadata"][ADAPTER_LINEAGE_METADATA_KEY]["schema_version"]
    with pytest.raises(ValueError, match="adapter_lineage.schema_version"):
        ExecutionInputV1.from_payload(missing_lineage_version)


def test_execution_input_fingerprint_excludes_locators_but_includes_roles_and_content(tmp_path: Path) -> None:
    first = _input(tmp_path / "first")
    base = ArtifactBindingV2(
        artifact_id=first.base.artifact_id,
        locator=ArtifactLocator("filesystem", "/other/path/base.safetensors"),
        content_identity=first.base.content_identity,
        format=first.base.format,
        quantization=first.base.quantization,
        revision=first.base.revision,
        metadata=first.base.metadata,
    )
    first_adapter = first.adapters[0]
    lineage_data = dict(first_adapter.metadata[ADAPTER_LINEAGE_METADATA_KEY])
    lineage_data["target_base"] = {
        **lineage_data["target_base"],
        "locator": {"type": "filesystem", "value": "/other/path/base.safetensors"},
    }
    adapter = ArtifactBindingV2(
        artifact_id=first_adapter.artifact_id,
        locator=ArtifactLocator("filesystem", "/other/path/adapter"),
        content_identity=first_adapter.content_identity,
        format=first_adapter.format,
        revision=first_adapter.revision,
        metadata={ADAPTER_LINEAGE_METADATA_KEY: lineage_data},
    )
    moved = ExecutionInputV1(base=base, adapters=(adapter,))
    assert moved.fingerprint == first.fingerprint

    changed_adapter = ArtifactBindingV2(
        artifact_id=first_adapter.artifact_id,
        locator=first_adapter.locator,
        content_identity=ContentIdentity.from_file(first_adapter.local_path),
        format=first_adapter.format,
        revision="adapter-revision-2",
        metadata=first_adapter.metadata,
    )
    assert ExecutionInputV1(base=first.base, adapters=(changed_adapter,)).fingerprint != first.fingerprint

    changed_base = _input(tmp_path / "changed-base", base_contents=b"different base fixture")
    assert changed_base.fingerprint != first.fingerprint


@pytest.mark.parametrize("mismatch", ["hash", "revision", "model", "module", "shape"])
def test_execution_input_compatibility_guard_rejects_lineage_mismatches(tmp_path: Path, mismatch: str) -> None:
    base = _base(tmp_path)
    target = AdapterTargetBaseV1(
        artifact_id=base.artifact_id,
        content_identity=ContentIdentity(
            algorithm="sha256",
            digest="a" * 64,
            scheme="complete",
            canonicalization_scheme="complete-file-v1",
        ) if mismatch == "hash" else base.content_identity,
        revision="wrong-revision" if mismatch == "revision" else base.revision,
        format=base.format,
        quantization=base.quantization,
        model_identity="wrong-model" if mismatch == "model" else base.metadata["model_identity"],
        tokenizer_identity=base.metadata["tokenizer_identity"],
    )
    modules = ("self_attn.k_proj",) if mismatch == "module" else (MODULE,)
    if mismatch == "module":
        shapes = {"self_attn.k_proj": {"lora_a": [4, 2], "lora_b": [2, 6]}}
    elif mismatch == "shape":
        shapes = {MODULE: {"lora_a": [4, 2], "lora_b": [2, 5]}}
    else:
        shapes = FACTOR_SHAPES
    adapter = _adapter(
        tmp_path,
        base,
        target_base=target,
        target_modules=modules,
        tensor_shapes=shapes,
    )
    with pytest.raises(ArtifactCompatibilityError):
        ExecutionInputV1(base=base, adapters=(adapter,))


def test_core_direct_load_generate_mismatch_trace_unload_reload_and_lease_reuse(tmp_path: Path) -> None:
    direct_a = _input(tmp_path / "a", "adapter-a")
    direct_b = ExecutionInputV1(
        base=direct_a.base,
        adapters=(_adapter(tmp_path / "b", direct_a.base, "adapter-b"),),
    )
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"mock": MockAdapter()})

    loaded = core.load(execution_input=direct_a, adapter="mock", consumer_id="test")
    assert loaded["raw"]["execution_input_fingerprint"] == direct_a.fingerprint
    reused = core.load(execution_input=direct_a, adapter="mock", consumer_id="test")
    assert reused["reused"] is True
    assert reused["lease_id"] == loaded["lease_id"]
    with pytest.raises(LoadConflictError):
        core.load(execution_input=direct_b, adapter="mock", consumer_id="test")
    with pytest.raises(ExecutionInputMismatchError):
        core.generate(GenerationRequest(direct_a.base.artifact_id, [{"role": "user", "content": "missing input"}]))
    with pytest.raises(ExecutionInputMismatchError):
        core.generate(_generate_request(direct_b))

    result = core.generate(_generate_request(direct_a, consumer_id="test", lease_id=loaded["lease_id"]))
    trace = result.to_dict()["generation_v2"]["trace"]
    assert trace["execution_input"]["execution_input_fingerprint"] == direct_a.fingerprint
    assert trace["execution_binding"]["execution_binding_fingerprint"] == trace["execution_binding_fingerprint"]
    assert trace["execution_binding"]["execution_input"]["execution_input_fingerprint"] == direct_a.fingerprint
    assert trace["execution_binding_fingerprint"].startswith("sha256:")

    unloaded = core.unload(direct_a.base.artifact_id, consumer_id="test", lease_id=loaded["lease_id"])
    assert unloaded["unloaded"] is True
    reloaded = core.load(execution_input=direct_b, adapter="mock", consumer_id="test")
    assert reloaded["reused"] is False
    assert core.unload(direct_b.base.artifact_id, consumer_id="test", lease_id=reloaded["lease_id"])["unloaded"]


def test_core_base_only_execution_input_uses_legacy_engine_path_with_distinct_trace(tmp_path: Path) -> None:
    original = _input(tmp_path)
    base_metadata = {key: value for key, value in original.base.metadata.items() if key != "target_module_shapes"}
    base = ArtifactBindingV2(
        artifact_id=original.base.artifact_id,
        locator=original.base.locator,
        content_identity=original.base.content_identity,
        format=original.base.format,
        quantization=original.base.quantization,
        revision=original.base.revision,
        metadata=base_metadata,
    )
    direct_base = ExecutionInputV1(base=base)
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"mock": MockAdapter()})
    loaded = core.load(execution_input=direct_base, adapter="mock", consumer_id="base-only")
    result = core.generate(
        _generate_request(direct_base, consumer_id="base-only", lease_id=loaded["lease_id"])
    )
    trace = result.to_dict()["generation_v2"]["trace"]
    assert trace["execution_input"]["adapters"] == []
    assert trace["execution_binding_fingerprint"].startswith("sha256:")
    assert core.unload(direct_base.base.artifact_id, consumer_id="base-only", lease_id=loaded["lease_id"])["unloaded"]


def test_core_verifies_complete_local_content_identity_before_engine_load(tmp_path: Path) -> None:
    direct = _input(tmp_path)
    wrong_base = ArtifactBindingV2(
        artifact_id=direct.base.artifact_id,
        locator=direct.base.locator,
        content_identity=ContentIdentity(
            algorithm="sha256", digest="b" * 64, scheme="complete", canonicalization_scheme="complete-file-v1"
        ),
        format=direct.base.format,
        quantization=direct.base.quantization,
        revision=direct.base.revision,
        metadata=direct.base.metadata,
    )
    target = AdapterTargetBaseV1(
        artifact_id=wrong_base.artifact_id,
        content_identity=wrong_base.content_identity,
        revision=wrong_base.revision,
        format=wrong_base.format,
        quantization=wrong_base.quantization,
        model_identity=wrong_base.metadata["model_identity"],
        tokenizer_identity=wrong_base.metadata["tokenizer_identity"],
    )
    adapter = _adapter(tmp_path / "wrong-id", wrong_base, target_base=target)
    input_with_wrong_digest = ExecutionInputV1(base=wrong_base, adapters=(adapter,))
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"mock": MockAdapter()})
    with pytest.raises(ArtifactCompatibilityError, match="does not match"):
        core.load(execution_input=input_with_wrong_digest, adapter="mock")


def test_core_rejects_missing_adapter_before_engine_load(tmp_path: Path) -> None:
    direct = _input(tmp_path)
    adapter = direct.adapters[0]
    missing = ArtifactBindingV2(
        artifact_id=adapter.artifact_id,
        locator=ArtifactLocator("filesystem", str(tmp_path / "missing-adapter")),
        content_identity=adapter.content_identity,
        format=adapter.format,
        revision=adapter.revision,
        metadata=adapter.metadata,
    )
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"mock": MockAdapter()})
    with pytest.raises(ArtifactNotFoundError) as caught:
        core.load(execution_input=ExecutionInputV1(direct.base, (missing,)), adapter="mock")
    assert caught.value.details["role"] == "adapter"


def test_single_base_and_fused_artifact_remain_on_legacy_load_path(tmp_path: Path) -> None:
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"mock": MockAdapter()})
    base_path = tmp_path / "legacy-base.bin"
    base_path.write_bytes(b"legacy base artifact")
    base = ModelArtifactBinding("legacy-base", str(base_path), "mlx", quantization="8-bit")
    base_load = core.load(base, adapter="mock", consumer_id="legacy")
    base_result = core.generate(
        GenerationRequest(
            model_artifact_id=base.artifact_id,
            consumer_id="legacy",
            lease_id=base_load["lease_id"],
            messages=[{"role": "user", "content": "legacy base"}],
        )
    )
    assert "execution_input" not in base_result.to_dict()["generation_v2"]["trace"]
    assert core.unload(base.artifact_id, consumer_id="legacy", lease_id=base_load["lease_id"])["unloaded"]

    fused_path = tmp_path / "materialized-fused.safetensors"
    fused_path.write_bytes(b"preexisting fused artifact")
    fused = ArtifactBindingV2(
        artifact_id="fused-materialized",
        locator=ArtifactLocator("filesystem", str(fused_path)),
        content_identity=ContentIdentity.from_file(fused_path),
        format="safetensors",
        quantization="8-bit-fused",
        revision="fused-revision-1",
    )
    fused_load = core.load(fused, adapter="mock", consumer_id="fused")
    fused_result = core.generate(
        GenerationRequestV2(
            model_artifact_id=fused.artifact_id,
            consumer_id="fused",
            lease_id=fused_load["lease_id"],
            messages=[{"role": "user", "content": "fused artifact"}],
            thinking_intent=ThinkingIntent(mode=ThinkingMode.OFF),
        )
    )
    trace = fused_result.to_dict()["generation_v2"]["trace"]
    assert "execution_input" not in trace
    assert trace["artifact_binding"]["artifact_id"] == fused.artifact_id
    assert core.unload(fused.artifact_id, consumer_id="fused", lease_id=fused_load["lease_id"])["unloaded"]


def test_service_and_local_client_direct_execution_round_trip(tmp_path: Path) -> None:
    direct = _input(tmp_path)
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"mock": MockAdapter()})
    transport = TestClient(create_app(core))
    with LocalRuntimeClient(http_client=transport, base_url="http://127.0.0.1") as client:
        loaded = client.load(execution_input=direct, engine="mock", consumer_id="direct-client")
        assert loaded["raw"]["execution_input_fingerprint"] == direct.fingerprint
        result = client.generate(
            _generate_request(direct, consumer_id="direct-client", lease_id=loaded["lease_id"])
        )
        assert result["generation_v2"]["trace"]["execution_input"]["execution_input_fingerprint"] == direct.fingerprint
        with pytest.raises(RemoteRuntimeError) as caught:
            client.generate(
                _generate_request(
                    ExecutionInputV1(base=direct.base, adapters=()),
                    consumer_id="direct-client",
                    lease_id=loaded["lease_id"],
                )
            )
        assert caught.value.code == "execution_input_mismatch"
        stream_events = list(
            client.stream(_generate_request(direct, consumer_id="direct-client", lease_id=loaded["lease_id"]))
        )
        assert stream_events[0]["type"] == "started"
        assert stream_events[-1]["type"] == "completed"
        assert stream_events[-1]["result"]["generation_v2"]["trace"]["execution_input"][
            "execution_input_fingerprint"
        ] == direct.fingerprint
        assert client.unload(direct.base.artifact_id, consumer_id="direct-client", lease_id=loaded["lease_id"])["unloaded"]


def test_service_rejects_ambiguous_or_malformed_direct_load(tmp_path: Path) -> None:
    direct = _input(tmp_path)
    client = TestClient(create_app(RuntimeCore(host_profile=HostProfile.mock_windows())))
    ambiguous = client.post(
        "/models/load",
        json={"artifact": direct.base.to_dict(), "execution_input": direct.to_dict(), "engine": "mock"},
    )
    assert ambiguous.status_code == 400
    assert ambiguous.json()["error"]["code"] == "invalid_request"
    unsupported = direct.to_dict()
    unsupported["adapters"].append(direct.adapters[0].to_dict())
    rejected = client.post("/models/load", json={"execution_input": unsupported, "engine": "mock"})
    assert rejected.status_code == 400
    assert rejected.json()["error"]["code"] == "unsupported_execution_input"


def test_mlx_direct_load_passes_adapter_path_only_for_composed_load(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    direct = _input(tmp_path)
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    load_thread_ids: list[int] = []

    def fake_load(path_or_hf_repo: str, adapter_path: str | None = None):
        calls.append(((path_or_hf_repo,), {"adapter_path": adapter_path}))
        load_thread_ids.append(threading.get_ident())
        return object(), object()

    module = SimpleNamespace(load=fake_load, stream_generate=lambda *args, **kwargs: None)
    adapter = MLXAdapter()
    monkeypatch.setattr(adapter, "_available", lambda: (object(), module))
    monkeypatch.setattr(adapter, "validate_execution_input", lambda execution_input: None)
    result = adapter.load_execution_input(direct)
    assert calls == [((direct.base.local_path,), {"adapter_path": direct.adapters[0].local_path})]
    assert load_thread_ids[0] == adapter._mlx_thread_id
    assert load_thread_ids[0] != threading.get_ident()
    assert result["load_mode"] == "direct_base_plus_adapter"
    assert result["execution_input_fingerprint"] == direct.fingerprint

    generation_thread_ids: list[int] = []

    def fake_items():
        generation_thread_ids.append(threading.get_ident())
        yield "token"
        generation_thread_ids.append(threading.get_ident())

    assert list(adapter._iter_on_mlx_thread(fake_items)) == ["token"]
    assert generation_thread_ids == [load_thread_ids[0], load_thread_ids[0]]


def test_core_cancel_terminal_finalizes_before_consumer_stops_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    direct = _input(tmp_path)
    load_thread_ids: list[int] = []
    generation_thread_ids: list[int] = []
    generator_close_thread_ids: list[int] = []
    cache_clear_thread_ids: list[int] = []

    class FakeTokenizer:
        @staticmethod
        def apply_chat_template(messages: list[dict[str, str]], **_: object) -> str:
            return "\n".join(message["content"] for message in messages)

        @staticmethod
        def encode(prompt: str) -> list[int]:
            return prompt.split()

    tokenizer = FakeTokenizer()

    def fake_load(path_or_hf_repo: str, adapter_path: str | None = None) -> tuple[object, FakeTokenizer]:
        assert path_or_hf_repo == direct.base.local_path
        assert adapter_path == direct.adapters[0].local_path
        load_thread_ids.append(threading.get_ident())
        return object(), tokenizer

    def fake_stream_generate(
        _model: object,
        _tokenizer: FakeTokenizer,
        _prompt: str,
        max_tokens: int = 64,
        prefill_step_size: int | None = None,
    ):
        del max_tokens
        del prefill_step_size

        def items():
            try:
                for index in range(8):
                    generation_thread_ids.append(threading.get_ident())
                    yield {
                        "text": f"{' ' if index else ''}token-{index}",
                        "prompt_tokens": 3,
                        "generation_tokens": index + 1,
                        "finish_reason": "stop",
                    }
            finally:
                generator_close_thread_ids.append(threading.get_ident())

        return items()

    fake_mlx = SimpleNamespace(clear_cache=lambda: cache_clear_thread_ids.append(threading.get_ident()))
    fake_mlx_lm = SimpleNamespace(load=fake_load, stream_generate=fake_stream_generate, __version__="test")
    adapter = MLXAdapter()
    monkeypatch.setattr(adapter, "_modules", lambda: (fake_mlx, fake_mlx_lm, None))
    monkeypatch.setattr(adapter, "validate_execution_input", lambda _: None)
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"mlx": adapter})
    consumer_id = "mlx-cancel-regression"
    loaded = core.load(execution_input=direct, adapter="mlx", consumer_id=consumer_id)

    request = _generate_request(direct, consumer_id=consumer_id, lease_id=loaded["lease_id"])
    events = core.stream(request)
    first_event = next(events)
    assert first_event.type == "started", first_event.to_dict()
    assert next(events).type == "delta"
    assert core.cancel(request.request_id)["cancelled"] is True
    terminal = next(events)
    assert terminal.type == "error"
    assert (terminal.error or {}).get("code") == "cancelled"

    # The simulated HTTP client consumes done=true then stops immediately; it
    # neither asks for another event nor explicitly closes the response iterator.
    assert core.health()["lifecycle_state"] == "LOADED"
    assert core.health()["active_request_ids"] == []
    assert adapter.runtime_metrics()["active_request_ids"] == []
    assert generator_close_thread_ids == [adapter._mlx_thread_id]

    next_request = _generate_request(direct, consumer_id=consumer_id, lease_id=loaded["lease_id"])
    generated = core.generate(next_request)
    assert generated.text == "token-0 token-1 token-2 token-3 token-4 token-5 token-6 token-7"
    assert core.unload(direct.base.artifact_id, consumer_id=consumer_id, lease_id=loaded["lease_id"])["unloaded"]
    assert cache_clear_thread_ids == [adapter._mlx_thread_id]
    assert load_thread_ids == [adapter._mlx_thread_id]
    assert generation_thread_ids
    assert set(generation_thread_ids) == {adapter._mlx_thread_id}
    events.close()
    adapter._mlx_executor.shutdown(wait=True)


def test_mlx_preflight_checks_actual_adapter_and_base_tensor_shapes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    direct = _input(tmp_path)
    base_dir = Path(direct.base.local_path).parent / "mlx-base"
    base_dir.mkdir()
    (base_dir / "config.json").write_text(
        json.dumps({"quantization_config": {"bits": 8, "group_size": 64, "mode": "affine"}}), encoding="utf-8"
    )
    (base_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {
            f"model.layers.0.{MODULE}.scales": "weights.safetensors",
            f"model.layers.0.{MODULE}.weight": "weights.safetensors",
        }}),
        encoding="utf-8",
    )
    (base_dir / "weights.safetensors").write_bytes(b"fixture header placeholder")
    adapter_binding = direct.adapters[0]
    actual_adapter_path = Path(adapter_binding.local_path)
    (actual_adapter_path / "adapter_config.json").write_text(
        json.dumps({"fine_tune_type": "lora", "lora_parameters": {"rank": 2, "keys": [MODULE]}}), encoding="utf-8"
    )
    (actual_adapter_path / "adapters.safetensors").write_bytes(b"fixture header placeholder")
    # Keep the input and actual fixtures content-bound after their test setup writes.
    actual_base_metadata = {
        **direct.base.metadata,
        "target_module_shapes": {MODULE: {"in_features": 64, "out_features": 6}},
    }
    actual_base = ArtifactBindingV2(
        artifact_id=direct.base.artifact_id,
        locator=ArtifactLocator("filesystem", str(base_dir)),
        content_identity=ContentIdentity.from_file(base_dir),
        format=direct.base.format,
        quantization="mlx:bits=8;group_size=64;mode=affine",
        revision=direct.base.revision,
        metadata=actual_base_metadata,
    )
    original_lineage = AdapterLineageV1.from_payload(adapter_binding.metadata[ADAPTER_LINEAGE_METADATA_KEY])
    target = AdapterTargetBaseV1(
        artifact_id=actual_base.artifact_id,
        content_identity=actual_base.content_identity,
        revision=actual_base.revision,
        format=actual_base.format,
        quantization=actual_base.quantization,
        model_identity=actual_base.metadata["model_identity"],
        tokenizer_identity=actual_base.metadata["tokenizer_identity"],
        locator=actual_base.locator,
    )
    mlx_factor_shapes = {MODULE: {"lora_a": [64, 2], "lora_b": [2, 6]}}
    new_lineage = AdapterLineageV1(target, original_lineage.rank, original_lineage.target_modules, mlx_factor_shapes)
    adapter_binding = ArtifactBindingV2(
        artifact_id=adapter_binding.artifact_id,
        locator=adapter_binding.locator,
        content_identity=ContentIdentity.from_file(actual_adapter_path),
        format=adapter_binding.format,
        revision=adapter_binding.revision,
        metadata={ADAPTER_LINEAGE_METADATA_KEY: new_lineage.to_dict()},
    )
    direct = ExecutionInputV1(actual_base, (adapter_binding,))

    adapter = MLXAdapter()
    monkeypatch.setattr(adapter, "_available", lambda: (object(), object()))

    def fake_shapes(files: list[Path], *, role: str) -> dict[str, list[int]]:
        if role == "Adapter":
            return {
                f"language_model.model.layers.0.{MODULE}.lora_a": [64, 2],
                f"language_model.model.layers.0.{MODULE}.lora_b": [2, 6],
            }
        return {
            f"model.layers.0.{MODULE}.scales": [6, 1],
            f"model.layers.0.{MODULE}.weight": [6, 64],
        }

    monkeypatch.setattr(MLXAdapter, "_safetensors_shapes", staticmethod(fake_shapes))
    adapter.validate_execution_input(direct)

    wrong_quant_base = ArtifactBindingV2(
        artifact_id=actual_base.artifact_id,
        locator=actual_base.locator,
        content_identity=actual_base.content_identity,
        format=actual_base.format,
        quantization="mlx:bits=4;group_size=64;mode=affine",
        revision=actual_base.revision,
        metadata=actual_base.metadata,
    )
    wrong_quant_target = AdapterTargetBaseV1(
        artifact_id=wrong_quant_base.artifact_id,
        content_identity=wrong_quant_base.content_identity,
        revision=wrong_quant_base.revision,
        format=wrong_quant_base.format,
        quantization=wrong_quant_base.quantization,
        model_identity=wrong_quant_base.metadata["model_identity"],
        tokenizer_identity=wrong_quant_base.metadata["tokenizer_identity"],
    )
    wrong_quant_lineage = AdapterLineageV1(
        wrong_quant_target, original_lineage.rank, original_lineage.target_modules, mlx_factor_shapes
    )
    wrong_quant_adapter = ArtifactBindingV2(
        artifact_id=adapter_binding.artifact_id,
        locator=adapter_binding.locator,
        content_identity=adapter_binding.content_identity,
        format=adapter_binding.format,
        revision=adapter_binding.revision,
        metadata={ADAPTER_LINEAGE_METADATA_KEY: wrong_quant_lineage.to_dict()},
    )
    wrong_quant_input = ExecutionInputV1(wrong_quant_base, (wrong_quant_adapter,))
    with pytest.raises(ArtifactCompatibilityError, match="quantization"):
        adapter.validate_execution_input(wrong_quant_input)

    def incompatible_base_shapes(files: list[Path], *, role: str) -> dict[str, list[int]]:
        result = fake_shapes(files, role=role)
        if role == "Base":
            result[f"model.layers.0.{MODULE}.scales"] = [6, 2]
            result[f"model.layers.0.{MODULE}.weight"] = [6, 128]
        return result

    monkeypatch.setattr(MLXAdapter, "_safetensors_shapes", staticmethod(incompatible_base_shapes))
    with pytest.raises(ArtifactCompatibilityError, match="shape"):
        adapter.validate_execution_input(direct)


def test_base_only_direct_load_uses_existing_single_path_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    direct_adapter = _input(tmp_path)
    direct = ExecutionInputV1(base=direct_adapter.base, adapters=())
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_load(path_or_hf_repo: str):
        calls.append(((path_or_hf_repo,), {}))
        return object(), object()

    module = SimpleNamespace(load=fake_load, stream_generate=lambda *args, **kwargs: None)
    adapter = MLXAdapter()
    monkeypatch.setattr(adapter, "_available", lambda: (object(), module))
    monkeypatch.setattr(adapter, "validate_execution_input", lambda execution_input: None)
    adapter.load_execution_input(direct)
    assert calls == [((direct.base.local_path,), {})]
