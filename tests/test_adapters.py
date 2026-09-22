from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from runtime_foundation.adapters.mlx import MLXAdapter
from runtime_foundation.adapters.mock import MockAdapter
from runtime_foundation.contracts import GenerationRequest, ModelArtifactBinding, RuntimeOptions
from runtime_foundation.errors import ContextLengthExceededError
from runtime_foundation.errors import UnsupportedRuntimeOptionError


def test_mlx_capability_marks_context_as_foundation_preflight(monkeypatch) -> None:
    def stream_generate(model, tokenizer, prompt, max_tokens=1, **kwargs):
        del model, tokenizer, prompt, max_tokens, kwargs
        yield {"text": "ok"}

    fake_mlx = SimpleNamespace(default_device=lambda: "gpu")
    fake_mlx_lm = SimpleNamespace(stream_generate=stream_generate)
    adapter = MLXAdapter()
    monkeypatch.setattr(adapter, "_modules", lambda: (fake_mlx, fake_mlx_lm, None))

    capability = adapter.discover_capability()
    assert capability.runtime_options["context.context_length"]["mode"] == "foundation-preflight-budget"
    assert capability.runtime_options["engine_options"]["status"] == "unsupported"

    with pytest.raises(UnsupportedRuntimeOptionError):
        adapter.resolve_runtime_options(RuntimeOptions.from_payload({"engine_options": {"mlx.foo": True}}))


def test_mock_capability_and_resolution_agree_on_option_status() -> None:
    adapter = MockAdapter()
    capability = adapter.discover_capability()
    resolution = adapter.resolve_runtime_options(RuntimeOptions.from_payload({}))

    for path, info in capability.runtime_options.items():
        assert resolution.option_status[path] == info["status"]
    assert resolution.option_status["context.sliding_window"] == "unsupported"
    assert resolution.option_status["prompt_cache.enabled"] == "unsupported"
    assert resolution.option_status["acceleration.threads"] == "unsupported"


def test_mlx_resolution_reuses_capability_option_status(monkeypatch) -> None:
    def stream_generate(model, tokenizer, prompt, max_tokens=1, **kwargs):
        del model, tokenizer, prompt, max_tokens, kwargs
        yield {"text": "ok"}

    fake_mlx = SimpleNamespace(default_device=lambda: "gpu")
    fake_mlx_lm = SimpleNamespace(stream_generate=stream_generate)
    adapter = MLXAdapter()
    monkeypatch.setattr(adapter, "_modules", lambda: (fake_mlx, fake_mlx_lm, None))

    capability = adapter.discover_capability()
    resolution = adapter.resolve_runtime_options(RuntimeOptions.from_payload({}))

    for path, info in capability.runtime_options.items():
        assert resolution.option_status[path] == info["status"]


def test_mlx_preflight_error_clears_active_request(monkeypatch, tmp_path) -> None:
    def stream_generate(model, tokenizer, prompt, max_tokens=1, **kwargs):
        del model, tokenizer, prompt, max_tokens, kwargs
        yield {"text": "never reached"}

    fake_mlx = SimpleNamespace(default_device=lambda: "gpu")
    fake_mlx_lm = SimpleNamespace(stream_generate=stream_generate)
    adapter = MLXAdapter()
    monkeypatch.setattr(adapter, "_modules", lambda: (fake_mlx, fake_mlx_lm, None))
    artifact = ModelArtifactBinding("fixture", str(tmp_path), "mlx")
    adapter._loaded = artifact
    adapter._model = object()
    adapter._tokenizer = SimpleNamespace(
        apply_chat_template=lambda messages, **kwargs: "tokenized prompt",
        encode=lambda prompt: [1, 2, 3],
    )
    adapter._stream_generate = stream_generate
    request = GenerationRequest(
        model_artifact_id=artifact.artifact_id,
        request_id="preflight-context-failure",
        messages=[{"role": "user", "content": "hello"}],
        max_tokens=4,
        runtime_options=RuntimeOptions.from_payload({"context": {"context_length": 4}}),
    )

    with pytest.raises(ContextLengthExceededError):
        list(adapter.stream(request, threading.Event()))

    assert adapter.runtime_metrics()["active_request_ids"] == []
