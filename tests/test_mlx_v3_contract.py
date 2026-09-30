from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from runtime_foundation import (
    ExecutionConstraints,
    GenerationOptions,
    GenerationRequestV3,
    HostProfile,
    LoadOptions,
    ModelArtifactBinding,
    RuntimeCore,
    ThinkingIntent,
    ThinkingMode,
)
from runtime_foundation.adapters.mlx import MLXAdapter
from runtime_foundation.errors import ContextLengthExceededError, UnsupportedRuntimeOptionError


@pytest.fixture
def mlx_fixture(tmp_path, monkeypatch):
    artifact_path = tmp_path / "mlx-test-fixture.safetensors"
    artifact_path.write_bytes(b"TEST-only synthetic adapter fixture; no model weights")
    artifact = ModelArtifactBinding("mlx-test-fixture", str(artifact_path), "safetensors")
    calls: list[dict[str, object]] = []

    class Tokenizer:
        bos_token = "<s>"
        chat_template = "{% if enable_thinking %}thinking{% else %}plain{% endif %} {{ messages }}"

        def apply_chat_template(self, messages, *, add_generation_prompt, enable_thinking):
            assert add_generation_prompt is True
            return f"<s>{messages[0]['content']}:{'think' if enable_thinking else 'plain'}"

        def encode(self, prompt, *, add_special_tokens):
            calls.append({"encoded_prompt": prompt, "add_special_tokens": add_special_tokens})
            return [101, 102, 103]

    tokenizer = Tokenizer()

    def load(path_or_hf_repo: str):
        assert path_or_hf_repo == str(artifact_path)
        return object(), tokenizer

    def stream_generate(model, tokenizer_arg, prompt, *, max_tokens=256, **kwargs):
        assert tokenizer_arg is tokenizer
        calls.append({"stream_prompt": prompt, "max_tokens": max_tokens, **kwargs})
        yield SimpleNamespace(
            text="generated",
            prompt_tokens=len(prompt),
            generation_tokens=1,
            generation_tps=12.0,
            prompt_tps=24.0,
            finish_reason="stop",
        )

    def make_sampler(temp: float = 0.0, top_p: float = 0.0, top_k: int = 0):
        return {"temperature": temp, "top_p": top_p, "top_k": top_k}

    def make_logits_processors(
        repetition_penalty: float | None = None,
        repetition_context_size: int | None = 20,
    ):
        return [{"repetition_penalty": repetition_penalty, "repetition_context_size": repetition_context_size}]

    def generate_step(prompt, model, *, sampler=None, logits_processors=None):
        del prompt, model, sampler, logits_processors

    fake_mlx = SimpleNamespace(default_device=lambda: "gpu")
    fake_mlx_lm = SimpleNamespace(load=load, stream_generate=stream_generate)
    api = {
        "make_sampler": make_sampler,
        "sampler_parameters": inspect.signature(make_sampler).parameters,
        "make_logits_processors": make_logits_processors,
        "logits_parameters": inspect.signature(make_logits_processors).parameters,
        "generate_step_parameters": inspect.signature(generate_step).parameters,
    }
    adapter = MLXAdapter()
    monkeypatch.setattr(adapter, "_modules", lambda: (fake_mlx, fake_mlx_lm, None))
    monkeypatch.setattr(adapter, "_mlx_v3_generation_api", lambda: api)
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"mlx": adapter})
    yield core, adapter, artifact, calls
    adapter._mlx_executor.shutdown(wait=True)


def _request(artifact, *, max_context_tokens=16, options=None, request_id="mlx-v3"):
    return GenerationRequestV3(
        model_artifact_id=artifact.artifact_id,
        messages=[{"role": "user", "content": "hello"}],
        generation_options=options
        or GenerationOptions(
            max_tokens=2,
            temperature=0.7,
            top_p=0.8,
            top_k=7,
            repetition_penalty=1.1,
            repetition_window=12,
            thinking_intent=ThinkingIntent(mode=ThinkingMode.OFF),
        ),
        execution_constraints=ExecutionConstraints(max_context_tokens=max_context_tokens),
        request_id=request_id,
        consumer_id="mlx-contract-test",
    )


def test_mlx_v3_separates_load_options_and_reports_supported_generation_settings(mlx_fixture) -> None:
    core, adapter, artifact, calls = mlx_fixture
    capability_before = adapter.discover_capability_v3()
    assert capability_before.options["model_context_size"].status.value == "unsupported"
    assert capability_before.options["max_context_tokens"].status.value == "supported"
    assert capability_before.chat_template.status.value == "unknown"
    assert capability_before.options["top_k"].status.value == "supported"
    assert capability_before.options["repetition_window"].status.value == "supported"

    with pytest.raises(UnsupportedRuntimeOptionError, match="LoadOptions"):
        core.load(
            artifact,
            adapter="mlx",
            load_options=LoadOptions(model_context_size=64),
            consumer_id="mlx-contract-test",
        )

    loaded = core.load(
        artifact,
        adapter="mlx",
        load_options=LoadOptions(),
        consumer_id="mlx-contract-test",
    )
    assert loaded["raw"]["load_options_resolution"]["effective"]["acceleration"]["backend"] == "metal"
    assert loaded["raw"]["selected_execution_acceleration"]["backend"] == "metal"
    assert adapter.health()["selected_execution_acceleration"]["actual_kernel_execution"] == "not_independently_observed"
    assert adapter.discover_capability_v3().chat_template.status.value == "supported"

    result = core.generate_v3(_request(artifact)).to_dict()
    evidence = result["execution_evidence"]["options"]["generation"]
    assert evidence["requested"]["top_k"] == 7
    assert evidence["effective"]["top_k"] == 7
    assert evidence["effective"]["repetition_penalty"] == 1.1
    assert evidence["effective"]["repetition_window"] == 12
    assert evidence["effective"]["top_p"] == 0.8
    stream_call = next(call for call in calls if "stream_prompt" in call)
    assert stream_call["stream_prompt"] == [101, 102, 103]
    assert stream_call["sampler"] == {"temperature": 0.7, "top_p": 0.8, "top_k": 7}
    assert stream_call["logits_processors"] == [
        {"repetition_penalty": 1.1, "repetition_context_size": 12}
    ]
    encoded = next(call for call in calls if "encoded_prompt" in call)
    assert encoded["add_special_tokens"] is False
    assert len([call for call in calls if "encoded_prompt" in call]) == 1


def test_mlx_v3_context_budget_uses_the_same_ids_passed_to_mlx_lm(mlx_fixture) -> None:
    core, _, artifact, calls = mlx_fixture
    core.load(artifact, adapter="mlx", load_options=LoadOptions(), consumer_id="mlx-contract-test")
    before = len([call for call in calls if "stream_prompt" in call])

    with pytest.raises(ContextLengthExceededError) as caught:
        core.generate_v3(_request(artifact, max_context_tokens=4, request_id="mlx-too-large"))

    assert caught.value.details["generation_started"] is False
    assert caught.value.details["prompt_tokens"] == 3
    assert len([call for call in calls if "stream_prompt" in call]) == before


def test_mlx_v3_resolves_native_sampler_and_repetition_defaults(mlx_fixture) -> None:
    core, _, artifact, calls = mlx_fixture
    core.load(artifact, adapter="mlx", load_options=LoadOptions(), consumer_id="mlx-contract-test")
    request = _request(
        artifact,
        options=GenerationOptions(
            max_tokens=1,
            temperature=0.5,
            thinking_intent=ThinkingIntent(mode=ThinkingMode.OFF),
            repetition_penalty=1.05,
        ),
        request_id="mlx-defaults",
    )

    result = core.generate_v3(request).to_dict()
    evidence = result["execution_evidence"]["options"]["generation"]
    assert evidence["effective"]["top_p"] == 0.0
    assert evidence["effective"]["top_k"] == 0
    assert evidence["effective"]["repetition_window"] == 20
    resolution_paths = {item["path"] for item in evidence["resolutions"]}
    assert {"top_p", "top_k", "repetition_window"}.issubset(resolution_paths)
    stream_call = next(call for call in calls if "stream_prompt" in call)
    assert stream_call["sampler"] == {"temperature": 0.5, "top_p": 0.0, "top_k": 0}
    assert stream_call["logits_processors"] == [
        {"repetition_penalty": 1.05, "repetition_context_size": 20}
    ]


def test_mlx_v3_stop_sequence_is_trimmed_across_stream_chunks(mlx_fixture) -> None:
    core, adapter, artifact, _ = mlx_fixture
    core.load(artifact, adapter="mlx", load_options=LoadOptions(), consumer_id="mlx-contract-test")

    def stream_with_split_stop(model, tokenizer, prompt, *, max_tokens=256, **kwargs):
        del model, tokenizer, prompt, max_tokens, kwargs
        yield SimpleNamespace(text="answer<ST", prompt_tokens=3, generation_tokens=2, finish_reason=None)
        yield SimpleNamespace(text="OP>hidden", prompt_tokens=3, generation_tokens=4, finish_reason=None)
        yield SimpleNamespace(text="must-not-appear", prompt_tokens=3, generation_tokens=8, finish_reason="stop")

    adapter._stream_generate = stream_with_split_stop
    request = _request(
        artifact,
        options=GenerationOptions(
            max_tokens=10,
            temperature=0.7,
            thinking_intent=ThinkingIntent(mode=ThinkingMode.OFF),
            stop=("<STOP>",),
        ),
        request_id="mlx-stop",
    )

    events = list(core.stream_v3(request))
    deltas = [event.delta for event in events if event.type == "delta"]
    result = events[-1].to_dict()["result"]
    assert "".join(deltas) == "answer"
    assert result["text"] == "answer"
    assert result["finish_reason"] == "stop_sequence"
