"""Engine adapter boundary."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any, Iterator

from ..contracts import (
    EngineCapability,
    EngineIdentity,
    GenerationRequest,
    GenerationResult,
    ModelArtifactBinding,
    RuntimeOptions,
    RuntimeSettingsResolution,
    StreamEvent,
)


class EngineAdapter(ABC):
    """The only execution interface Core may use for a model engine."""

    name: str

    def identity(self) -> EngineIdentity:
        capability = self.discover_capability()
        return capability.identity

    @abstractmethod
    def discover_capability(self) -> EngineCapability:
        raise NotImplementedError

    @abstractmethod
    def load(self, artifact: ModelArtifactBinding) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def unload(self, artifact_id: str) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def resolve_runtime_options(self, options: RuntimeOptions) -> RuntimeSettingsResolution:
        """Validate and resolve settings without silently dropping a field."""

        raise NotImplementedError

    @abstractmethod
    def generate(self, request: GenerationRequest, cancel_event: threading.Event) -> GenerationResult:
        raise NotImplementedError

    @abstractmethod
    def stream(self, request: GenerationRequest, cancel_event: threading.Event) -> Iterator[StreamEvent]:
        raise NotImplementedError

    @abstractmethod
    def cancel(self, request_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def health(self) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def runtime_metrics(self) -> dict[str, Any]:
        raise NotImplementedError
