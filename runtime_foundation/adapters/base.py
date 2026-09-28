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
from ..contracts_v2 import ExecutionInputV1
from ..errors import UnsupportedExecutionInputError


class EngineAdapter(ABC):
    """The only execution interface Core may use for a model engine."""

    name: str

    def build_identity(self) -> Any | None:
        """Return structured observed build identity, or unknown."""

        return None

    def identity(self) -> EngineIdentity:
        capability = self.discover_capability()
        return capability.identity

    @abstractmethod
    def discover_capability(self) -> EngineCapability:
        raise NotImplementedError

    @abstractmethod
    def load(self, artifact: ModelArtifactBinding) -> dict[str, Any]:
        raise NotImplementedError

    def validate_execution_input(self, execution_input: ExecutionInputV1) -> None:
        """Preflight a composition before any engine model load is attempted."""

        if execution_input.adapters:
            raise UnsupportedExecutionInputError(
                "this engine adapter does not support direct LoRA execution",
                details={"engine": self.name, "adapter_count": len(execution_input.adapters)},
            )

    def load_execution_input(self, execution_input: ExecutionInputV1) -> dict[str, Any]:
        """Load a base-only composition through the existing single-artifact path."""

        self.validate_execution_input(execution_input)
        return self.load(execution_input.base.to_legacy())

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
