from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from runtime_foundation import (
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
from runtime_foundation.adapters.mlx import MLXAdapter
from runtime_foundation.adapters.mock import MockAdapter
from runtime_foundation.errors import UnsupportedGenerationSettingError
from runtime_foundation.service import create_app

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
}
TOOL_TURNS = [
    {"role": "user", "content": "open the readme"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": {"path": "README.md"}}}],
    },
    {"role": "tool", "content": "# RSVP Reader", "tool_call_id": "call_1", "name": "read_file"},
]


def _tool_request(artifact_id, *, request_id="tools-v3", consumer_id="rsvp", tools=(READ_FILE_TOOL,), messages=TOOL_TURNS):
    return GenerationRequestV3(
        model_artifact_id=artifact_id,
        messages=messages,
        tools=tools,
        generation_options=GenerationOptions(max_tokens=2, thinking_intent=ThinkingIntent(mode=ThinkingMode.OFF)),
        request_id=request_id,
        consumer_id=consumer_id,
    )


def test_tool_fields_round_trip_and_legacy_payload_shape_is_unchanged() -> None:
    plain = GenerationRequestV3(model_artifact_id="m", messages=[{"role": "user", "content": "hi", "extra": 1}])
    assert plain.uses_chat_tools is False
    assert plain.messages == ({"role": "user", "content": "hi"},)
    assert "tools" not in plain.to_dict() and "scheduling" not in plain.to_dict()

    request = GenerationRequestV3.from_payload(_tool_request("m").to_dict())
    assert request.uses_chat_tools is True
    assert request.template_tools() == [READ_FILE_TOOL]
    assert request.template_messages()[1]["tool_calls"][0]["function"]["name"] == "read_file"
    assert request.template_messages()[2]["tool_call_id"] == "call_1"


@pytest.mark.parametrize(
    "tools, messages",
    [
        ([{"type": "function", "function": {}}], [{"role": "user", "content": "x"}]),
        ([{"type": "retrieval"}], [{"role": "user", "content": "x"}]),
        ([], [{"role": "assistant", "content": "", "tool_calls": []}]),
        ([], [{"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "f", "arguments": 3}}]}]),
        ([{"type": "function", "function": {"name": "f", "parameters": {"bad": float("nan")}}}], [{"role": "user", "content": "x"}]),
    ],
)
def test_invalid_tools_and_tool_calls_are_rejected(tools, messages) -> None:
    with pytest.raises(ValueError):
        GenerationRequestV3(model_artifact_id="m", messages=messages, tools=tools)


@pytest.fixture
def mlx_tools(tmp_path, monkeypatch):
    artifact_path = tmp_path / "mlx-tools-fixture.safetensors"
    artifact_path.write_bytes(b"TEST-only synthetic adapter fixture; no model weights")
    artifact = ModelArtifactBinding("mlx-tools-fixture", str(artifact_path), "safetensors")
    rendered: list[dict[str, object]] = []

    class Tokenizer:
        bos_token = "<s>"
        chat_template = "{% if tools %}{{ tools }}{% endif %}{% for m in messages %}{{ m.content }}{% endfor %}"

        def apply_chat_template(self, messages, *, add_generation_prompt, tools=None, enable_thinking=None):
            rendered.append({"messages": messages, "tools": tools, "enable_thinking": enable_thinking})
            return "<s>" + "|".join(message["content"] for message in messages)

        def encode(self, prompt, *, add_special_tokens):
            return [101, 102, 103]

    def stream_generate(model, tokenizer_arg, prompt, *, max_tokens=256, **kwargs):
        yield SimpleNamespace(
            text="<|tool_call>call:read_file{}", prompt_tokens=len(prompt), generation_tokens=1,
            generation_tps=12.0, prompt_tps=24.0, finish_reason="stop",
        )

    def make_sampler(temp: float = 0.0, top_p: float = 0.0, top_k: int = 0):
        return {"temperature": temp}

    def make_logits_processors(repetition_penalty: float | None = None, repetition_context_size: int | None = 20):
        return []

    def generate_step(prompt, model, *, sampler=None, logits_processors=None):
        del prompt, model, sampler, logits_processors

    tokenizer = Tokenizer()
    adapter = MLXAdapter()
    monkeypatch.setattr(
        adapter,
        "_modules",
        lambda: (
            SimpleNamespace(default_device=lambda: "gpu"),
            SimpleNamespace(load=lambda path: (object(), tokenizer), stream_generate=stream_generate),
            None,
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_mlx_v3_generation_api",
        lambda: {
            "make_sampler": make_sampler,
            "sampler_parameters": inspect.signature(make_sampler).parameters,
            "make_logits_processors": make_logits_processors,
            "logits_parameters": inspect.signature(make_logits_processors).parameters,
            "generate_step_parameters": inspect.signature(generate_step).parameters,
        },
    )
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"mlx": adapter})
    core.load(artifact, adapter="mlx", load_options=LoadOptions(), consumer_id="rsvp")
    yield core, adapter, tokenizer, artifact, rendered
    adapter._mlx_executor.shutdown(wait=True)


def test_mlx_passes_tools_and_tool_turns_to_the_chat_template(mlx_tools) -> None:
    core, adapter, _, artifact, rendered = mlx_tools

    result = core.generate_v3(_tool_request(artifact.artifact_id)).to_dict()

    assert result["text"] == "<|tool_call>call:read_file{}"
    assert rendered[-1]["tools"] == [READ_FILE_TOOL]
    assert rendered[-1]["messages"][1]["tool_calls"][0]["id"] == "call_1"
    assert rendered[-1]["messages"][2] == TOOL_TURNS[2]
    assert rendered[-1]["enable_thinking"] is False
    assert adapter.discover_capability_v3().to_dict()["features"]["chat_tools"]["status"] == "supported"


def test_mlx_plain_requests_still_render_without_tools(mlx_tools) -> None:
    core, _, _, artifact, rendered = mlx_tools

    core.generate_v3(_tool_request(artifact.artifact_id, tools=(), messages=[{"role": "user", "content": "hello"}]))

    assert rendered[-1]["tools"] is None
    assert rendered[-1]["messages"] == [{"role": "user", "content": "hello"}]


def test_mlx_template_without_tools_parameter_fails_closed(mlx_tools) -> None:
    core, _, tokenizer, artifact, _ = mlx_tools

    def no_tools_template(messages, *, add_generation_prompt, enable_thinking=None):
        return "<s>plain"

    tokenizer.apply_chat_template = no_tools_template
    with pytest.raises(UnsupportedGenerationSettingError, match="does not accept tools"):
        core.generate_v3(_tool_request(artifact.artifact_id, request_id="no-tools-template"))


class ToollessMockAdapter(MockAdapter):
    supports_chat_tools_v3 = False


def _mock_core(tmp_path, adapter):
    artifact_path = tmp_path / "tools-model.bin"
    artifact_path.write_bytes(b"test-only-runtime-fixture")
    artifact = ModelArtifactBinding("tools-model", str(artifact_path), "bin")
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"mock": adapter})
    return core, artifact


def test_core_refuses_tools_for_an_adapter_without_tool_support(tmp_path) -> None:
    core, artifact = _mock_core(tmp_path, ToollessMockAdapter())
    core.load(artifact, adapter="mock", load_options=LoadOptions(), consumer_id="rsvp")

    with pytest.raises(UnsupportedGenerationSettingError) as caught:
        core.generate_v3(_tool_request(artifact.artifact_id))

    assert caught.value.details == {"setting": "tools", "engine": "mock", "generation_started": False}
    assert ToollessMockAdapter().discover_capability_v3().to_dict()["features"]["chat_tools"]["status"] == "unsupported"


def test_tools_and_scheduling_reach_the_service_over_http(tmp_path) -> None:
    core, artifact = _mock_core(tmp_path, MockAdapter())
    client = LocalRuntimeClient(http_client=TestClient(create_app(core)))
    client.load(artifact, engine="mock", consumer_id="rsvp", load_options=LoadOptions())
    payload = _tool_request(artifact.artifact_id, request_id="http-tools").to_dict()
    payload["scheduling"] = {"priority": "background", "queue_timeout_ms": 1000}

    result = client.generate(payload)

    assert result["contract_version"] == "runtime-foundation.contract.v3"
    assert client.health_v3()["generation_queue"]["priorities"] == ["interactive", "background"]
