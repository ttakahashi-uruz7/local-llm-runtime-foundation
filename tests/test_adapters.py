from __future__ import annotations

from types import SimpleNamespace

import pytest

from runtime_foundation.adapters.mlx import MLXAdapter
from runtime_foundation.contracts import RuntimeOptions
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
