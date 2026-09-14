"""llama.cpp / GGUF compatibility skeleton.

The interface is intentionally present in v1 so consumers do not become MLX
coupled.  GGUF production execution is outside this milestone.
"""

from __future__ import annotations

import threading
from typing import Any, Iterator

from ..contracts import (
    EngineCapability,
    EngineIdentity,
    GenerationRequest,
    GenerationResult,
    ModelArtifactBinding,
    RUNTIME_OPTION_PATHS,
    RuntimeOptions,
    RuntimeSettingsResolution,
    StreamEvent,
)
from ..errors import EngineUnavailableError
from .base import EngineAdapter


class LlamaCppAdapter(EngineAdapter):
    """Future GGUF adapter boundary; no production implementation in v1."""

    name = "llama.cpp"

    def identity(self) -> EngineIdentity:
        return EngineIdentity(engine=self.name)

    def discover_capability(self) -> EngineCapability:
        reason = "llama.cpp/GGUF execution is a v1 interface skeleton"
        return EngineCapability(
            identity=self.identity(),
            available=False,
            streaming=False,
            cancellation=False,
            load_unload=False,
            chat_template="unavailable",
            thinking_flag="unavailable",
            artifact_formats=["gguf"],
            runtime_options={path: {"status": "unavailable", "reason": reason} for path in RUNTIME_OPTION_PATHS},
            generation_options={
                name: {"status": "unavailable", "reason": reason}
                for name in ("max_tokens", "temperature", "top_p", "thinking_enabled")
            },
            reason=reason,
        )

    def _unavailable(self) -> None:
        raise EngineUnavailableError("llama.cpp adapter is not implemented in Foundation v1", details={"engine": self.name})

    def load(self, artifact: ModelArtifactBinding) -> dict[str, Any]:
        del artifact
        self._unavailable()

    def unload(self, artifact_id: str) -> dict[str, Any]:
        del artifact_id
        self._unavailable()

    def resolve_runtime_options(self, options: RuntimeOptions) -> RuntimeSettingsResolution:
        del options
        self._unavailable()

    def generate(self, request: GenerationRequest, cancel_event: threading.Event) -> GenerationResult:
        del request, cancel_event
        self._unavailable()

    def stream(self, request: GenerationRequest, cancel_event: threading.Event) -> Iterator[StreamEvent]:
        del request, cancel_event
        self._unavailable()
        yield  # pragma: no cover - keeps the function a generator to satisfy the interface

    def cancel(self, request_id: str) -> bool:
        del request_id
        return False

    def health(self) -> dict[str, Any]:
        return {"status": "unavailable", "engine": self.name, "reason": self.discover_capability().reason}

    def runtime_metrics(self) -> dict[str, Any]:
        return {"engine": self.name, "active_request_ids": [], "last_metrics": {}}
