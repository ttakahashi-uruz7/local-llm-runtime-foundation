from __future__ import annotations

import hashlib
import os
import struct
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from runtime_foundation import HostExecutionCapabilitiesV3, HostProfile, RuntimeCore
from runtime_foundation.adapters.llama_cpp import (
    LlamaCppAdapter,
    NativeBinding,
    _metal_marker,
    _native_api_string,
    _native_metal_load_observation,
)
from runtime_foundation.contracts import GenerationRequest, RuntimeOptions
from runtime_foundation.contracts_v2 import (
    ArtifactBindingV2,
    ArtifactLocator,
    BuildIdentityV1,
    ContentIdentity,
    ThinkingIntent,
    ThinkingMode,
)
from runtime_foundation.contracts_v3 import (
    AccelerationConfiguration,
    ExecutionConstraints,
    GenerationOptions,
    GenerationRequestV3,
    KVCacheConfiguration,
    LoadOptions,
)
from runtime_foundation.errors import (
    ArtifactCompatibilityError,
    ContextLengthExceededError,
    EngineRuntimeError,
    LoadConflictError,
    UnsupportedGenerationSettingError,
    UnsupportedRuntimeOptionError,
)
from runtime_foundation.gguf import VerifiedGGUFArtifact


_TEMPLATE = (
    "{% if enable_thinking %}<think>{% endif %}"
    "{% for message in messages %}{{ message['role'] }}:{{ message['content'] }};{% endfor %}"
)


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf(path: Path, *, context: int = 64) -> bytes:
    metadata = [
        ("general.architecture", 8, "tiny"),
        ("general.file_type", 4, 0),
        ("tiny.context_length", 4, context),
        ("tokenizer.chat_template", 8, _TEMPLATE),
        ("tiny.block_count", 4, 2),
        ("general.alignment", 4, 32),
    ]
    encoded = bytearray(b"GGUF")
    encoded.extend(struct.pack("<IQQ", 3, 1, len(metadata)))
    for key, value_type, value in metadata:
        encoded.extend(_string(key))
        encoded.extend(struct.pack("<I", value_type))
        encoded.extend(_string(value) if value_type == 8 else struct.pack("<I", value))
    encoded.extend(_string("weight"))
    encoded.extend(struct.pack("<I Q I Q", 1, 1, 0, 0))
    encoded.extend(b"\0" * ((32 - len(encoded) % 32) % 32))
    encoded.extend(struct.pack("<f", 0.25))
    path.write_bytes(encoded)
    return bytes(encoded)


class _FakeLlama:
    instances: list[_FakeLlama] = []
    pause_before_first_chunk = 0.0
    completion_started = threading.Event()
    loaded_file_sha256: str | None = None
    formatter_init_args: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.n_threads = kwargs["n_threads"]
        self.context_params = SimpleNamespace(type_k=kwargs["type_k"], type_v=kwargs["type_v"])
        self.model_params = SimpleNamespace(
            n_gpu_layers=0x7FFFFFFF if kwargs["n_gpu_layers"] == -1 else kwargs["n_gpu_layers"]
        )
        self.metadata = {
            "general.architecture": "tiny",
            "general.file_type": 0,
            "tiny.context_length": 64,
            "tokenizer.chat_template": _TEMPLATE,
        }
        self.template_calls: list[dict[str, Any]] = []
        self.completion_calls: list[dict[str, Any]] = []
        self.tokenize_calls: list[dict[str, Any]] = []
        self.closed = False
        with open(kwargs["model_path"], "rb") as handle:
            loaded_file = handle.read()
        assert loaded_file.startswith(b"GGUF")
        self.loaded_file_sha256 = hashlib.sha256(loaded_file).hexdigest()
        self._model = SimpleNamespace(token_get_text=lambda token_id: {1: "<bos>", 2: "<eot>"}[token_id])
        self._chat_handlers = {"chat_template.default": self._forbidden_chat_handler}
        self.__class__.instances.append(self)

    @staticmethod
    def _forbidden_chat_handler(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("generation chat handler must not be used as a formatter")

    def token_eos(self) -> int:
        return 2

    def token_bos(self) -> int:
        return 1

    def n_ctx(self) -> int:
        return self.kwargs["n_ctx"]

    def n_batch(self) -> int:
        return self.kwargs["n_batch"]

    def n_ubatch(self) -> int:
        return self.kwargs["n_ubatch"]

    def tokenize(self, value: bytes, *, add_bos: bool, special: bool) -> list[int]:
        self.tokenize_calls.append({"text": value.decode("utf-8"), "add_bos": add_bos, "special": special})
        return list(range(max(len(value.decode("utf-8").split()), 1)))

    def create_completion(
        self,
        *,
        prompt: Any,
        max_tokens: int,
        temperature: float,
        stream: bool,
        stop: list[str] | None = None,
        stopping_criteria: Any = None,
        top_p: float = 0.95,
        top_k: int = 40,
        repeat_penalty: float = 1.0,
        seed: int | None = None,
    ):
        self.completion_calls.append(
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": stream,
                "stop": stop,
                "stopping_criteria": stopping_criteria,
                "top_p": top_p,
                "top_k": top_k,
                "repeat_penalty": repeat_penalty,
                "seed": seed,
            }
        )
        self.__class__.completion_started.set()

        def _chunks():
            if self.pause_before_first_chunk:
                time.sleep(self.pause_before_first_chunk)
            yield {"choices": [{"text": "hello world", "finish_reason": None}]}
            yield {"choices": [{"text": "", "finish_reason": "stop"}]}

        return _chunks()

    def close(self) -> None:
        self.closed = True


class _FakeJinja2ChatFormatter:
    def __init__(
        self,
        *,
        template: str,
        eos_token: str,
        bos_token: str,
        stop_token_ids: list[int] | None,
    ) -> None:
        self.template = template
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.stop_token_ids = stop_token_ids
        from jinja2 import Environment

        self._environment = Environment().from_string(template)
        _FakeLlama.formatter_init_args.append(
            {
                "template": template,
                "eos_token": eos_token,
                "bos_token": bos_token,
                "stop_token_ids": stop_token_ids,
            }
        )

    def __call__(self, *, messages: list[dict[str, str]], **kwargs: Any) -> Any:
        llama = _FakeLlama.instances[-1]
        llama.template_calls.append({"messages": messages, **kwargs})
        prefix = "<think>" if kwargs.get("enable_thinking") else ""
        prompt = prefix + "".join(f"{item['role']}:{item['content']};" for item in messages)
        return SimpleNamespace(prompt=prompt, added_special=True, stop=[self.eos_token], stopping_criteria=None)


def _native(*, metal: bool | None = False, gpu_offload: bool | None = False) -> NativeBinding:
    system_info = f"Metal = {int(metal)}" if metal is not None else "Metal support unknown"
    api = SimpleNamespace(
        GGML_TYPE_F16=1,
        GGML_TYPE_Q8_0=8,
        llama_print_system_info=lambda: system_info,
        llama_supports_gpu_offload=lambda: gpu_offload,
    )
    return NativeBinding(
        llama_class=_FakeLlama,
        api=api,
        package_version="0.3.test",
        system_info=system_info,
        library_path=None,
        library_sha256=None,
        gpu_offload=gpu_offload,
        metal_build=metal,
        chat_format_module=SimpleNamespace(Jinja2ChatFormatter=_FakeJinja2ChatFormatter),
    )


def _artifact(path: Path) -> ArtifactBindingV2:
    return ArtifactBindingV2(
        artifact_id="local-test-gguf",
        locator=ArtifactLocator("filesystem", str(path)),
        format="gguf",
    )


def _load_options(context: int = 48) -> LoadOptions:
    return LoadOptions(
        model_context_size=context,
        batch=24,
        ubatch=12,
        threads=2,
        kv_cache=KVCacheConfiguration(key_type="f16", value_type="f16"),
        acceleration=AccelerationConfiguration(backend="cpu", gpu_offload_layers=0),
    )


def _request(
    artifact_id: str,
    lease_id: str,
    *,
    request_id: str = "llama-test-request",
    max_tokens: int = 8,
    max_context_tokens: int | None = 40,
    thinking: ThinkingIntent | None = None,
) -> GenerationRequestV3:
    return GenerationRequestV3(
        model_artifact_id=artifact_id,
        messages=({"role": "user", "content": "hello"},),
        request_id=request_id,
        lease_id=lease_id,
        generation_options=GenerationOptions(
            max_tokens=max_tokens,
            temperature=0.25,
            top_p=0.8,
            top_k=12,
            repetition_penalty=1.1,
            stop=("END",),
            seed=7,
            thinking_intent=thinking or ThinkingIntent(mode=ThinkingMode.OFF),
        ),
        execution_constraints=ExecutionConstraints(max_context_tokens=max_context_tokens),
    )


def _runtime(path: Path) -> tuple[RuntimeCore, LlamaCppAdapter]:
    _FakeLlama.instances.clear()
    _FakeLlama.completion_started = threading.Event()
    _FakeLlama.formatter_init_args.clear()
    adapter = LlamaCppAdapter(native=_native(), host_profile=HostProfile.mock_windows())
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"llama.cpp": adapter})
    return core, adapter


def test_gguf_default_engine_loads_through_core_and_records_v3_execution_evidence(tmp_path: Path) -> None:
    path = tmp_path / "test-only.gguf"
    _gguf(path)
    core, adapter = _runtime(path)
    load_options = _load_options()

    loaded = core.load(_artifact(path), load_options=load_options)
    assert loaded["engine"]["engine"] == "llama.cpp"
    assert loaded["raw"]["effective_load_options"]["model_context_size"] == 48
    assert loaded["raw"]["observed_effective_load_state"]["acceleration"]["backend"] == "cpu"
    assert loaded["raw"]["load_identity"]["fingerprint"].startswith("sha256:")
    observed_load = loaded["raw"]["observed_effective_load_state"]
    assert observed_load["batch"]["value"] == 24
    assert observed_load["ubatch"]["value"] == 12
    assert observed_load["threads"]["value"] == 2
    assert observed_load["kv_cache"]["key_type"]["value"] == "f16"
    assert observed_load["kv_cache"]["value_type"]["native_type_id"] == 1
    assert observed_load["acceleration"]["native_n_gpu_layers"] == 0
    native_identity = loaded["raw"]["engine_build_identity"]["components"]
    assert native_identity["python_binding"]["version"] == "0.3.test"
    assert native_identity["native_library"]["system_info"] == "Metal = 0"
    assert adapter._verified_gguf is not None
    assert adapter._verified_gguf.fd >= 0

    result = core.generate_v3(_request("local-test-gguf", loaded["lease_id"]))
    assert result.text == "hello world"
    evidence = result.evidence.to_dict()
    assert evidence["execution_binding"]["engine_binding"]["family"] == "llama.cpp"
    assert evidence["execution_binding"]["effective_load_options"]["model_context_size"] == 48
    assert evidence["options"]["generation"]["effective"]["top_k"] == 12
    assert evidence["options"]["constraints"]["effective"]["max_context_tokens"] == 40

    llama = _FakeLlama.instances[-1]
    call = llama.completion_calls[-1]
    assert sum(call["text"] == "user:hello;" for call in llama.tokenize_calls) == 1
    assert _FakeLlama.formatter_init_args[-1] == {
        "template": _TEMPLATE,
        "eos_token": "<eot>",
        "bos_token": "<bos>",
        "stop_token_ids": [2],
    }
    assert llama.tokenize_calls[0]["add_bos"] is False
    assert llama.tokenize_calls[0]["special"] is True
    assert call["prompt"] == [0]
    assert call["temperature"] == 0.25
    assert call["top_p"] == 0.8
    assert call["top_k"] == 12
    assert call["repeat_penalty"] == 1.1
    assert call["seed"] == 7
    assert call["stop"] == ["<eot>", "END"]
    assert llama.template_calls[-1]["enable_thinking"] is False

    stream = list(core.stream_v3(_request("local-test-gguf", loaded["lease_id"], request_id="stream-v3")))
    assert [event.type for event in stream] == ["started", "delta", "completed"]
    assert sum(call["text"] == "user:hello;" for call in llama.tokenize_calls) == 2
    assert stream[-1].result["text"] == "hello world"
    assert stream[-1].evidence.execution_binding.fingerprint == result.evidence.execution_binding.fingerprint

    core.unload(consumer_id="anonymous")
    assert adapter._verified_gguf is None
    assert llama.closed


def test_gguf_load_out_of_memory_is_normalized_and_does_not_leave_a_load(tmp_path: Path) -> None:
    path = tmp_path / "test-only.gguf"
    _gguf(path)

    class _OutOfMemoryLlama:
        def __init__(self, **_kwargs: Any) -> None:
            raise RuntimeError("failed to allocate context: out of memory")

    adapter = LlamaCppAdapter(
        native=replace(_native(), llama_class=_OutOfMemoryLlama),
        host_profile=HostProfile.mock_windows(),
    )
    with pytest.raises(EngineRuntimeError) as caught:
        adapter.load(_artifact(path).to_legacy())
    assert caught.value.details["failure_kind"] == "out_of_memory"
    assert adapter._loaded is None
    assert adapter._verified_gguf is None


def test_load_identity_reuse_requires_matching_effective_load_options(tmp_path: Path) -> None:
    path = tmp_path / "test-only.gguf"
    _gguf(path)
    core, adapter = _runtime(path)
    options = _load_options()
    loaded = core.load(_artifact(path), load_options=options)

    reused = core.load(_artifact(path), consumer_id="second-consumer", load_options=options)
    assert reused["reused"] is True
    assert len(_FakeLlama.instances) == 1
    with pytest.raises(LoadConflictError, match="explicitly unload"):
        core.load(_artifact(path), consumer_id="third-consumer", load_options=_load_options(32))
    assert len(_FakeLlama.instances) == 1

    adapter._build_identity = BuildIdentityV1.from_components(
        kind="llama-cpp-native-build-v1",
        components={"test_build_change": "new-native-build"},
    )
    with pytest.raises(LoadConflictError, match="explicitly unload"):
        core.load(_artifact(path), consumer_id="fourth-consumer", load_options=options)

    core.unload(consumer_id="second-consumer")
    core.unload(consumer_id="anonymous")
    assert adapter._loaded is None
    assert loaded["lease_id"]


def test_core_loads_the_verified_gguf_inode_after_path_replacement(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "test-only.gguf"
    expected_contents = _gguf(path, context=64)
    original_sha256 = hashlib.sha256(expected_contents).hexdigest()
    original_content_digest = ContentIdentity.from_file(path).digest
    replacement = tmp_path / "replacement.gguf"
    _gguf(replacement, context=32)
    core, adapter = _runtime(path)
    original_resolver = adapter.resolve_load_options_with_observation
    replaced = False

    def replace_after_observation(artifact, options, observation):
        nonlocal replaced
        resolution = original_resolver(artifact, options, observation)
        if not replaced:
            os.replace(replacement, path)
            replaced = True
        return resolution

    monkeypatch.setattr(adapter, "resolve_load_options_with_observation", replace_after_observation)
    loaded = core.load(_artifact(path), load_options=_load_options())

    assert replaced
    assert _FakeLlama.instances[-1].loaded_file_sha256 == original_sha256
    assert loaded["raw"]["gguf_validation"]["content_identity"]["digest"] == original_content_digest
    core.unload()


def test_load_options_change_while_generation_is_active_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "test-only.gguf"
    _gguf(path)
    core, adapter = _runtime(path)
    loaded = core.load(_artifact(path), load_options=_load_options())
    _FakeLlama.pause_before_first_chunk = 0.05
    worker_result: dict[str, Any] = {}

    def generate() -> None:
        try:
            worker_result["result"] = core.generate_v3(
                _request("local-test-gguf", loaded["lease_id"], request_id="active-generation")
            )
        except Exception as exc:  # noqa: BLE001 - assert the worker reports no runtime failure
            worker_result["error"] = exc

    worker = threading.Thread(target=generate)
    worker.start()
    assert _FakeLlama.completion_started.wait(timeout=2)
    with pytest.raises(LoadConflictError, match="explicitly unload"):
        core.load(_artifact(path), load_options=_load_options(32))
    worker.join(timeout=3)
    assert not worker.is_alive()
    assert "error" not in worker_result
    assert worker_result["result"].text == "hello world"
    _FakeLlama.pause_before_first_chunk = 0.0
    core.unload(consumer_id="anonymous")


def test_constraint_budget_and_template_thinking_are_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "test-only.gguf"
    _gguf(path)
    core, adapter = _runtime(path)
    loaded = core.load(_artifact(path), load_options=_load_options())

    with pytest.raises(ContextLengthExceededError, match="max_context_tokens"):
        core.generate_v3(
            _request(
                "local-test-gguf",
                loaded["lease_id"],
                request_id="too-small-budget",
                max_tokens=8,
                max_context_tokens=2,
            )
        )
    assert _FakeLlama.instances[-1].completion_calls == []

    with pytest.raises(ContextLengthExceededError, match="loaded llama.cpp context"):
        core.generate_v3(
            _request(
                "local-test-gguf",
                loaded["lease_id"],
                request_id="loaded-context-overflow",
                max_tokens=48,
                max_context_tokens=None,
            )
        )
    assert _FakeLlama.instances[-1].completion_calls == []

    effort_request = _request(
        "local-test-gguf",
        loaded["lease_id"],
        request_id="unsupported-thinking-effort",
        thinking=ThinkingIntent(mode=ThinkingMode.ON, effort="HIGH"),
    )
    with pytest.raises(UnsupportedGenerationSettingError, match="effort"):
        core.generate_v3(effort_request)
    assert _FakeLlama.instances[-1].completion_calls == []
    core.unload()


def test_v3_chat_template_runtime_failure_is_normalized_before_generation(tmp_path: Path) -> None:
    path = tmp_path / "test-only.gguf"
    _gguf(path)
    core, adapter = _runtime(path)
    loaded = core.load(_artifact(path), load_options=_load_options())

    def failed_formatter(**_kwargs: Any) -> Any:
        raise ValueError("template rendering failed")

    adapter._chat_formatter = failed_formatter
    with pytest.raises(EngineRuntimeError) as caught:
        core.generate_v3(_request("local-test-gguf", loaded["lease_id"], request_id="template-failure"))
    assert caught.value.details["failure_kind"] == "chat_template_failure"
    assert _FakeLlama.instances[-1].completion_calls == []
    core.unload()


def test_v3_native_defaults_resolve_explicitly_and_repetition_window_is_rejected(tmp_path: Path, monkeypatch) -> None:
    from runtime_foundation.adapters import llama_cpp as llama_module

    path = tmp_path / "test-only.gguf"
    _gguf(path)
    core, adapter = _runtime(path)
    loaded = core.load(_artifact(path), load_options=_load_options())
    monkeypatch.setattr(llama_module.secrets, "randbits", lambda _count: 9876)
    request = GenerationRequestV3(
        model_artifact_id="local-test-gguf",
        messages=({"role": "user", "content": "hello"},),
        lease_id=loaded["lease_id"],
        generation_options=GenerationOptions(max_tokens=4, temperature=0.2),
        execution_constraints=ExecutionConstraints(max_context_tokens=32),
    )
    settings = adapter.resolve_generation_options_v3(request).to_dict()
    assert settings["requested"]["top_p"] is None
    assert settings["resolved"]["top_p"] == 0.95
    assert settings["effective"]["top_k"] == 40
    assert settings["effective"]["repetition_penalty"] == 1.0
    assert settings["effective"]["seed"] == 9876
    resolutions = {item["path"]: item for item in settings["resolutions"]}
    assert resolutions["seed"]["requested"] is None
    assert resolutions["seed"]["effective"] == 9876

    zero_top_k = GenerationRequestV3(
        model_artifact_id="local-test-gguf",
        messages=({"role": "user", "content": "hello"},),
        lease_id=loaded["lease_id"],
        generation_options=GenerationOptions(max_tokens=4, temperature=0.2, top_k=0),
    )
    with pytest.raises(UnsupportedGenerationSettingError, match="top_k must be at least one"):
        adapter.resolve_generation_options_v3(zero_top_k)

    result = core.generate_v3(request)
    call = _FakeLlama.instances[-1].completion_calls[-1]
    assert call["top_p"] == 0.95
    assert call["top_k"] == 40
    assert call["repeat_penalty"] == 1.0
    assert call["seed"] == 9876
    assert result.evidence.options.generation.effective["seed"] == 9876

    unsupported = GenerationRequestV3(
        model_artifact_id="local-test-gguf",
        messages=({"role": "user", "content": "hello"},),
        lease_id=loaded["lease_id"],
        generation_options=GenerationOptions(
            max_tokens=4,
            temperature=0.2,
            repetition_window=12,
            thinking_intent=ThinkingIntent(mode=ThinkingMode.OFF),
        ),
    )
    with pytest.raises(UnsupportedGenerationSettingError, match="repetition_window"):
        adapter.resolve_generation_options_v3(unsupported)
    core.unload()


def test_metal_all_layer_request_matches_llama_cpp_native_sentinel(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "test-only.gguf"
    _gguf(path)
    adapter = LlamaCppAdapter(native=_native(metal=True, gpu_offload=True), host_profile=HostProfile.mock_windows())
    monkeypatch.setattr(adapter, "_host_metal_observation", lambda: (True, "fixture host Metal"))
    options = LoadOptions(
        model_context_size=48,
        batch=24,
        ubatch=12,
        threads=2,
        kv_cache=KVCacheConfiguration(key_type="f16", value_type="f16"),
        acceleration=AccelerationConfiguration(backend="metal", gpu_offload_layers="all"),
    )
    with VerifiedGGUFArtifact.open(_artifact(path)) as verified:
        resolution = adapter.resolve_load_options_with_observation(
            _artifact(path).to_legacy(), options, verified.observed
        )
        loaded = adapter.load_verified_gguf(_artifact(path).to_legacy(), resolution, verified)
    assert loaded["observed_effective_load_state"]["acceleration"]["native_n_gpu_layers"] == 0x7FFFFFFF
    adapter.unload("local-test-gguf")


def test_llama_load_options_reject_unknown_native_kv_cache_values(tmp_path: Path) -> None:
    path = tmp_path / "test-only.gguf"
    _gguf(path)
    adapter = LlamaCppAdapter(native=_native(), host_profile=HostProfile.mock_windows())
    with VerifiedGGUFArtifact.open(_artifact(path)) as verified:
        with pytest.raises(UnsupportedRuntimeOptionError, match="KV key type"):
            adapter.resolve_load_options_with_observation(
                _artifact(path).to_legacy(),
                LoadOptions(kv_cache=KVCacheConfiguration(key_type="q4_0")),
                verified.observed,
            )


def test_native_capability_keeps_cpu_and_metal_observations_independent(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "test-only.gguf"
    _gguf(path)
    adapter = LlamaCppAdapter(native=_native(metal=True, gpu_offload=False), host_profile=HostProfile.mock_windows())
    monkeypatch.setattr(adapter, "_host_metal_observation", lambda: (True, "fixture host"))
    capability = adapter.discover_capability_v3(
        host_observation={"host_metal_capability": True, "host_metal_reason": "fixture host"}
    )
    assert capability.options["repetition_window"].status.value == "unsupported"
    acceleration = capability.options["acceleration.backend"]
    gpu_layers = capability.options["acceleration.gpu_offload_layers"]
    assert acceleration.status.value == "supported"
    assert acceleration.allowed_values == ("auto", "cpu")
    assert acceleration.evidence["host_metal"] is True
    assert acceleration.evidence["native_build_metal"] is True
    assert acceleration.evidence["native_gpu_offload"] is False
    assert gpu_layers.status.value == "supported"
    assert gpu_layers.allowed_values == (0,)

    with VerifiedGGUFArtifact.open(_artifact(path)) as verified:
        with pytest.raises(UnsupportedRuntimeOptionError, match="GPU offload support"):
            adapter.resolve_load_options_with_observation(
                _artifact(path).to_legacy(),
                LoadOptions(acceleration=AccelerationConfiguration(backend="metal")),
                verified.observed,
            )


def test_explicit_metal_request_fails_closed_when_native_build_is_non_metal(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "test-only.gguf"
    _gguf(path)
    adapter = LlamaCppAdapter(native=_native(metal=False, gpu_offload=True), host_profile=HostProfile.mock_windows())
    monkeypatch.setattr(adapter, "_host_metal_observation", lambda: (True, "fixture host Metal"))

    with VerifiedGGUFArtifact.open(_artifact(path)) as verified:
        with pytest.raises(UnsupportedRuntimeOptionError, match="both a Metal-enabled") as caught:
            adapter.resolve_load_options_with_observation(
                _artifact(path).to_legacy(),
                LoadOptions(acceleration=AccelerationConfiguration(backend="metal")),
                verified.observed,
            )

    assert caught.value.details["native_build_metal"] is False
    assert caught.value.details["host_metal"] is True


def test_unobserved_build_and_host_metal_stay_unknown(monkeypatch) -> None:
    adapter = LlamaCppAdapter(native=_native(metal=None, gpu_offload=None))
    monkeypatch.setattr(adapter, "_host_capability", lambda _host: (None, "fixture host Metal unknown"))
    capability = adapter.discover_capability_v3(host_observation={"platform": "darwin"})
    host_capability = HostExecutionCapabilitiesV3.observe(
        platform="darwin",
        architecture="arm64",
        host_metal=None,
        host_metal_reason="fixture host Metal unknown",
        mlx_available=False,
        mlx_default_device_metal=False,
        llama_cpp_capability=capability,
    )
    assert capability.options["acceleration.backend"].evidence["metal_capability_status"] == "unknown"
    assert host_capability.llama_cpp_availability.value == "supported"
    assert host_capability.llama_cpp_build_metal.value == "unknown"


def test_native_build_probe_reads_ggml_registry_and_native_version() -> None:
    registry = ("CPU", "MTL")

    def count() -> int:
        return len(registry)

    def get(index: int) -> int:
        return index + 1

    def name(registration: int) -> bytes:
        return registry[registration - 1].encode()

    def version() -> bytes:
        return b"0.20.0"

    def commit() -> bytes:
        return b"4df29be-dirty"

    api = SimpleNamespace(
        _lib=SimpleNamespace(
            ggml_backend_reg_count=count,
            ggml_backend_reg_get=get,
            ggml_backend_reg_name=name,
            ggml_version=version,
            ggml_commit=commit,
        )
    )
    assert _metal_marker("MTL : EMBED_LIBRARY = 1", api) is True
    assert _native_api_string(api, "ggml_version") == "0.20.0"
    assert _native_api_string(api, "ggml_commit") == "4df29be-dirty"

    cpu_only = SimpleNamespace(_lib=SimpleNamespace(
        ggml_backend_reg_count=lambda: 1,
        ggml_backend_reg_get=lambda _index: 1,
        ggml_backend_reg_name=lambda _registration: b"CPU",
    ))
    assert _metal_marker(None, cpu_only) is False
    assert _metal_marker(None, SimpleNamespace()) is None


def test_native_metal_execution_evidence_requires_device_buffer_and_offloaded_layers() -> None:
    observation = _native_metal_load_observation(
        (
            "load_tensors: offloaded 29/29 layers to GPU",
            "ggml_metal_init: found device: Apple M5 Max",
            "sched_reserve: MTL0 compute buffer size = 37.34 MiB",
        )
    )
    assert observation["status"] == "confirmed"
    assert observation["all_model_layers_offloaded"] is True
    assert observation["offloaded_layers"] == 29
    assert observation["metal_device_observations"] == ["Apple M5 Max"]

    no_metal_device = _native_metal_load_observation(("load_tensors: offloaded 29/29 layers to GPU",))
    assert no_metal_device["status"] == "unobserved"

    adapter = LlamaCppAdapter(native=_native(metal=True, gpu_offload=True))
    adapter._selected_execution_acceleration = {
        "backend": "metal",
        "actual_kernel_execution": "pending_generation",
        "native_metal_load_observation": observation,
    }
    adapter._record_native_metal_generation()
    assert adapter._selected_execution_acceleration["actual_kernel_execution"] == (
        "confirmed_by_native_metal_offload_and_generated_token"
    )


def test_native_metadata_numeric_strings_match_gguf_numeric_observations() -> None:
    observed = SimpleNamespace(
        architecture="qwen3",
        file_type=2,
        context_length=40960,
        chat_template="{% for message in messages %}{{ message['content'] }}{% endfor %}",
    )
    native_metadata = {
        "general.architecture": "qwen3",
        "general.file_type": "2",
        "qwen3.context_length": "40960",
        "tokenizer.chat_template": observed.chat_template,
    }
    LlamaCppAdapter._validate_native_metadata(observed, native_metadata)

    native_metadata["general.file_type"] = "3"
    with pytest.raises(ArtifactCompatibilityError) as exc_info:
        LlamaCppAdapter._validate_native_metadata(observed, native_metadata)
    assert exc_info.value.details == {"metadata_key": "general.file_type", "observed": 2, "native": "3"}


def test_unavailable_binding_is_reported_as_unavailable_not_unsupported(monkeypatch) -> None:
    from runtime_foundation.adapters import llama_cpp as llama_module

    monkeypatch.setattr(llama_module, "_read_native_binding", lambda: (None, "fixture dependency missing"))
    adapter = LlamaCppAdapter()
    capability = adapter.discover_capability_v3()
    assert capability.status.value == "unavailable"
    assert capability.options["max_tokens"].status.value == "unavailable"
    assert capability.options["acceleration.backend"].status.value == "unavailable"


def test_legacy_cooperative_cancel_and_timeout_emit_normalized_events(tmp_path: Path) -> None:
    path = tmp_path / "test-only.gguf"
    _gguf(path)
    adapter = LlamaCppAdapter(native=_native(), host_profile=HostProfile.mock_windows())
    artifact = _artifact(path).to_legacy()
    adapter.load(artifact)
    request = GenerationRequest(
        model_artifact_id=artifact.artifact_id,
        request_id="cooperative-cancel",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=4,
        runtime_options=RuntimeOptions(),
    )
    cancel_event = threading.Event()
    events = adapter.stream(request, cancel_event)
    assert next(events).type == "started"
    cancel_event.set()
    cancelled = next(events)
    assert cancelled.type == "error"
    assert cancelled.error["code"] == "cancelled"
    assert cancelled.error["details"] == {}
    events.close()

    _FakeLlama.pause_before_first_chunk = 0.01
    timeout_request = GenerationRequest(
        model_artifact_id=artifact.artifact_id,
        request_id="cooperative-timeout",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=4,
        timeout_ms=1,
    )
    timed_out = list(adapter.stream(timeout_request, threading.Event()))
    assert timed_out[-1].type == "error"
    assert timed_out[-1].error["code"] == "runtime_timeout"
    assert timed_out[-1].error["details"]["timeout_semantics"] == "cooperative"
    _FakeLlama.pause_before_first_chunk = 0.0
    adapter.unload(artifact.artifact_id)


def test_v3_cooperative_cancel_and_timeout_emit_normalized_events(tmp_path: Path) -> None:
    path = tmp_path / "test-only.gguf"
    _gguf(path)
    core, adapter = _runtime(path)
    loaded = core.load(_artifact(path), load_options=_load_options())
    _FakeLlama.pause_before_first_chunk = 0.05
    cancel_events: list[Any] = []
    cancel_errors: list[BaseException] = []

    def collect_cancelled() -> None:
        try:
            cancel_events.extend(
                core.stream_v3(_request("local-test-gguf", loaded["lease_id"], request_id="v3-cancel"))
            )
        except BaseException as exc:  # noqa: BLE001 - captured for assertion in the test thread
            cancel_errors.append(exc)

    worker = threading.Thread(target=collect_cancelled)
    worker.start()
    assert _FakeLlama.completion_started.wait(timeout=2)
    assert core.cancel("v3-cancel")["cancelled"] is True
    worker.join(timeout=2)
    assert not worker.is_alive()
    assert not cancel_errors
    assert cancel_events[-1].type == "error"
    assert cancel_events[-1].error["code"] == "cancelled"

    timeout_request = GenerationRequestV3(
        model_artifact_id="local-test-gguf",
        messages=({"role": "user", "content": "hello"},),
        lease_id=loaded["lease_id"],
        request_id="v3-timeout",
        generation_options=GenerationOptions(max_tokens=4, temperature=0.2),
        execution_constraints=ExecutionConstraints(timeout_ms=5, max_context_tokens=32),
    )
    timed_out = list(core.stream_v3(timeout_request))
    assert timed_out[-1].type == "error"
    assert timed_out[-1].error["code"] == "runtime_timeout"
    assert timed_out[-1].error["details"]["timeout_semantics"] == "cooperative"

    _FakeLlama.pause_before_first_chunk = 0.0
    core.unload()
