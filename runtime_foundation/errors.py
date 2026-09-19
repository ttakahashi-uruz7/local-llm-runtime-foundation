"""Stable error semantics shared by Core, the service, and the client."""

from __future__ import annotations

from typing import Any


class RuntimeFoundationError(Exception):
    """Base exception with a version-independent machine-readable payload."""

    code = "runtime_error"
    http_status = 500
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        details: dict[str, Any] | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = dict(details or {})
        if retryable is not None:
            self.retryable = retryable

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": dict(self.details),
        }


class InvalidRequestError(RuntimeFoundationError):
    code = "invalid_request"
    http_status = 400


class ArtifactNotFoundError(RuntimeFoundationError):
    code = "artifact_not_found"
    http_status = 404


class UnsupportedArtifactLocatorError(RuntimeFoundationError):
    code = "unsupported_artifact_locator"
    http_status = 400


class ModelNotLoadedError(RuntimeFoundationError):
    code = "model_not_loaded"
    http_status = 409


class RuntimeBusyError(RuntimeFoundationError):
    code = "runtime_busy"
    http_status = 409
    retryable = True


class LoadConflictError(RuntimeFoundationError):
    code = "load_conflict"
    http_status = 409
    retryable = True


class UnloadConflictError(RuntimeFoundationError):
    code = "unload_conflict"
    http_status = 409
    retryable = True


class EngineNotFoundError(RuntimeFoundationError):
    code = "engine_not_found"
    http_status = 404


class EngineUnavailableError(RuntimeFoundationError):
    code = "engine_unavailable"
    http_status = 503


class EngineRuntimeError(RuntimeFoundationError):
    code = "engine_runtime_error"
    http_status = 500


class RequestCancelledError(RuntimeFoundationError):
    code = "cancelled"
    http_status = 409


class RuntimeTimeoutError(RuntimeFoundationError):
    code = "runtime_timeout"
    http_status = 504
    retryable = True


class ContextLengthExceededError(RuntimeFoundationError):
    code = "context_length_exceeded"
    http_status = 400


class UnsupportedRuntimeOptionError(RuntimeFoundationError):
    code = "unsupported_runtime_option"
    http_status = 400


class InvalidRuntimeOptionError(RuntimeFoundationError):
    code = "invalid_runtime_option"
    http_status = 400


class RuntimeOptionUnavailableError(RuntimeFoundationError):
    code = "runtime_option_unavailable"
    http_status = 503


class UnsupportedGenerationSettingError(RuntimeFoundationError):
    code = "unsupported_generation_setting"
    http_status = 400


class ThinkingResolutionError(RuntimeFoundationError):
    code = "thinking_resolution_error"
    http_status = 400


class ExecutionGuardMismatchError(RuntimeFoundationError):
    code = "execution_guard_mismatch"
    http_status = 409


class ExecutionBindingUnresolvableError(RuntimeFoundationError):
    code = "execution_binding_unresolvable"
    http_status = 409


class RuntimeOptionsSchemaMismatchError(RuntimeFoundationError):
    code = "runtime_options_schema_mismatch"
    http_status = 400


class RuntimeOptionsAliasConflictError(RuntimeFoundationError):
    code = "runtime_options_alias_conflict"
    http_status = 400


class LifecycleTransitionError(RuntimeFoundationError):
    code = "invalid_lifecycle_transition"
    http_status = 409


def error_payload(error: RuntimeFoundationError, *, execution_id: str | None = None) -> dict[str, Any]:
    """Return the wire error shape without exposing exception internals."""

    payload = error.as_dict()
    if execution_id is not None:
        payload.setdefault("details", {})["execution_id"] = execution_id
    return payload
