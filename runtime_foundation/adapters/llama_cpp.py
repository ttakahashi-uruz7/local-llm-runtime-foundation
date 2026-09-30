"""Formal llama.cpp adapter for consumer-supplied GGUF artifacts.

The optional native binding is imported lazily. Native model state stays behind
this adapter and in the Foundation process; see the current runtime design for
the cooperative cancellation and native-crash boundary.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib
import importlib.metadata
import inspect
import os
import platform
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

from ..capabilities_v3 import (
    CapabilityApplicabilityV3,
    CapabilityDependency,
    CapabilityScope,
    CapabilityStatus,
    EngineCapabilityV3,
    OptionCapabilityV3,
)
from ..contracts import (
    EngineCapability,
    EngineIdentity,
    GenerationRequest,
    GenerationResult,
    ModelArtifactBinding,
    RUNTIME_OPTION_PATHS,
    RuntimeMetrics,
    RuntimeOptions,
    RuntimeSettingsResolution,
    StreamEvent,
    TokenUsage,
    new_id,
    utc_now,
)
from ..contracts_v2 import (
    ArtifactBindingV2,
    BuildIdentityV1,
    EngineBindingV2,
    ExecutionInputV1,
    GenerationRequestV2,
    ThinkingIntent,
    ThinkingMode,
)
from ..contracts_v3 import (
    AccelerationConfiguration,
    GenerationOptions,
    GenerationRequestV3,
    KVCacheConfiguration,
    LoadOptions,
    LoadOptionsResolutionV1,
    OptionResolutionV3,
    SettingsEvidenceV3,
)
from ..errors import (
    ArtifactCompatibilityError,
    ContextLengthExceededError,
    EngineRuntimeError,
    EngineUnavailableError,
    RequestCancelledError,
    RuntimeBusyError,
    RuntimeFoundationError,
    RuntimeTimeoutError,
    ThinkingResolutionError,
    UnsupportedExecutionInputError,
    UnsupportedGenerationSettingError,
    UnsupportedRuntimeOptionError,
)
from ..gguf import GGUFObservedMetadata, VerifiedGGUFArtifact
from ..host import HostProfile, host_metal_capability, process_memory_bytes
from .base import EngineAdapter, PreparedGenerationV3


_GGML_KV_TYPES = ("f16", "q8_0")
_TEMPLATE_INTENT_NAMES = {
    "thinking": ("enable_thinking", "thinking"),
    "effort": ("thinking_effort", "reasoning_effort"),
    "budget": ("thinking_budget_tokens", "thinking_budget"),
}
_NATIVE_LOG_CALLBACK_LOCK = threading.RLock()


@dataclass(frozen=True)
class NativeBinding:
    """The small native binding surface used by the adapter and contract tests."""

    llama_class: Any
    api: Any
    package_version: str | None
    system_info: str | None
    library_path: str | None
    library_sha256: str | None
    gpu_offload: bool | None
    metal_build: bool | None
    chat_format_module: Any | None = None
    native_version: str | None = None
    native_commit: str | None = None


def _text(value: Any) -> str | None:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip() or None
    if isinstance(value, str):
        return value.strip() or None
    return None


def _sha256_file(path: str | None) -> str | None:
    if not path:
        return None
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _native_api_string(api: Any, symbol: str) -> str | None:
    """Read a native no-argument string API without assuming it is bound in Python."""

    library = getattr(api, "_lib", None)
    function = getattr(library, symbol, None)
    if not callable(function):
        return None
    try:
        function.argtypes = []
        function.restype = ctypes.c_char_p
        return _text(function())
    except Exception:  # noqa: BLE001 - native build metadata is optional evidence
        return None


def _metal_marker(system_info: str | None, api: Any | None = None) -> bool | None:
    """Observe a Metal build marker or a registered native Metal backend.

    Newer llama.cpp builds report ``MTL`` (Metal) through ggml's public backend
    registry API rather than including a ``Metal=1`` marker in system info.
    Missing registry APIs or an uninitialized registry remain unknown.
    """

    if system_info:
        match = re.search(r"\bMetal\s*[=:]\s*([01])\b", system_info, flags=re.IGNORECASE)
        if match:
            return bool(int(match.group(1)))

    library = getattr(api, "_lib", None)
    count_fn = getattr(library, "ggml_backend_reg_count", None)
    get_fn = getattr(library, "ggml_backend_reg_get", None)
    name_fn = getattr(library, "ggml_backend_reg_name", None)
    if not all(callable(fn) for fn in (count_fn, get_fn, name_fn)):
        return None
    try:
        count_fn.argtypes = []
        count_fn.restype = ctypes.c_size_t
        get_fn.argtypes = [ctypes.c_size_t]
        get_fn.restype = ctypes.c_void_p
        name_fn.argtypes = [ctypes.c_void_p]
        name_fn.restype = ctypes.c_char_p
        count = int(count_fn())
        if count <= 0:
            return None
        for index in range(count):
            name = _text(name_fn(get_fn(index)))
            if name and name.casefold() in {"mtl", "metal"}:
                return True
        return False
    except Exception:  # noqa: BLE001 - preserve unknown instead of guessing
        return None


def _construct_native_model(native: NativeBinding, kwargs: dict[str, Any]) -> tuple[Any, tuple[str, ...]]:
    """Construct a model while briefly observing llama.cpp's native load log.

    llama.cpp exposes a process-global logger. Serialize Foundation model
    construction while wrapping that callback, and always forward the original
    callback and restore it before returning.
    """

    api = native.api
    log_get = getattr(api, "llama_log_get", None)
    log_set = getattr(api, "llama_log_set", None)
    callback_type = getattr(api, "llama_log_callback", None)
    if not callable(log_get) or not callable(log_set) or not callable(callback_type):
        return native.llama_class(**kwargs), ()

    captured: list[str] = []
    with _NATIVE_LOG_CALLBACK_LOCK:
        original_callback = callback_type()
        original_user_data = ctypes.c_void_p()
        try:
            log_get(ctypes.byref(original_callback), ctypes.byref(original_user_data))
        except Exception:  # noqa: BLE001 - log observation is optional; model load remains authoritative
            return native.llama_class(**kwargs), ()

        @callback_type
        def capture(level: int, text: bytes, _user_data: ctypes.c_void_p) -> None:
            message = _text(text)
            if message and (
                ("offloaded " in message.lower() and " layers to gpu" in message.lower())
                or re.search(r"\bMTL\d+\s+compute buffer size\b", message, flags=re.IGNORECASE)
                or "ggml_metal_init: found device:" in message.lower()
            ):
                captured.append(message)
            if original_callback:
                original_callback(level, text, original_user_data)

        log_set(capture, ctypes.c_void_p())
        try:
            model = native.llama_class(**kwargs)
        finally:
            log_set(original_callback, original_user_data)
    return model, tuple(captured)


def _native_metal_load_observation(lines: tuple[str, ...]) -> dict[str, Any]:
    """Summarize native evidence that model layers and compute buffers use MTL."""

    offload: tuple[int, int] | None = None
    devices: list[str] = []
    metal_compute_buffers: list[str] = []
    for line in lines:
        match = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", line, flags=re.IGNORECASE)
        if match:
            offload = (int(match.group(1)), int(match.group(2)))
        if "ggml_metal_init: found device:" in line.lower():
            devices.append(line.split(":", 2)[-1].strip())
        if re.search(r"\bMTL\d+\s+compute buffer size\b", line, flags=re.IGNORECASE):
            metal_compute_buffers.append(line)
    all_layers_offloaded = bool(offload and offload[0] > 0 and offload[0] == offload[1])
    confirmed = bool(offload and offload[0] > 0) and bool(metal_compute_buffers) and bool(devices)
    return {
        "status": "confirmed" if confirmed else "unobserved",
        "source": "llama.cpp native load log",
        "offloaded_layers": offload[0] if offload else None,
        "total_layers": offload[1] if offload else None,
        "all_model_layers_offloaded": all_layers_offloaded,
        "metal_device_observations": devices,
        "metal_compute_buffer_observations": metal_compute_buffers,
        "log_lines": list(lines),
    }


def _read_native_binding() -> tuple[NativeBinding | None, str | None]:
    try:
        package = importlib.import_module("llama_cpp")
        api = importlib.import_module("llama_cpp.llama_cpp")
        chat_format_module = importlib.import_module("llama_cpp.llama_chat_format")
    except (ImportError, OSError) as exc:
        return None, f"llama-cpp-python native binding is unavailable: {exc}"
    llama_class = getattr(package, "Llama", None)
    if not callable(llama_class):
        return None, "llama-cpp-python does not expose Llama"
    system_info = None
    system_info_fn = getattr(api, "llama_print_system_info", None)
    if callable(system_info_fn):
        try:
            system_info = _text(system_info_fn())
        except Exception:  # noqa: BLE001 - native identity is optional but never guessed
            system_info = None
    native_library = getattr(api, "_lib", None)
    library_path = getattr(native_library, "_name", None)
    if isinstance(library_path, str) and library_path:
        library_path = os.path.realpath(library_path)
    else:
        library_path = None
    try:
        version = importlib.metadata.version("llama-cpp-python")
    except importlib.metadata.PackageNotFoundError:
        version = _text(getattr(package, "__version__", None))
    gpu_offload = None
    gpu_fn = getattr(api, "llama_supports_gpu_offload", None)
    if callable(gpu_fn):
        try:
            gpu_offload = bool(gpu_fn())
        except Exception:  # noqa: BLE001 - capability evidence remains unknown
            gpu_offload = None
    return (
        NativeBinding(
            llama_class=llama_class,
            api=api,
            package_version=version,
            system_info=system_info,
            library_path=library_path,
            library_sha256=_sha256_file(library_path),
            gpu_offload=gpu_offload,
            metal_build=_metal_marker(system_info, api),
            chat_format_module=chat_format_module,
            native_version=_native_api_string(api, "ggml_version"),
            native_commit=_native_api_string(api, "ggml_commit"),
        ),
        None,
    )


class LlamaCppAdapter(EngineAdapter):
    """Execute a local GGUF through the optional llama-cpp-python binding."""

    name = "llama.cpp"

    def __init__(self, *, native: NativeBinding | None = None, host_profile: HostProfile | None = None) -> None:
        self._native = native
        self._native_checked = native is not None
        self._unavailable_reason: str | None = None
        self._host_profile = host_profile
        self._host_metal: tuple[bool | None, str] | None = None
        self._lock = threading.RLock()
        self._loaded: ModelArtifactBinding | None = None
        self._loaded_v2: ArtifactBindingV2 | None = None
        self._load_options: LoadOptions | None = None
        self._verified_gguf: VerifiedGGUFArtifact | None = None
        self._observed_gguf: GGUFObservedMetadata | None = None
        self._chat_formatter: Callable[..., Any] | None = None
        self._chat_template_text: str | None = None
        self._template_variables: frozenset[str] = frozenset()
        self._active: dict[str, threading.Event] = {}
        self._last_metrics: dict[str, Any] = {}
        self._selected_execution_acceleration: dict[str, Any] | None = None
        self._build_identity: BuildIdentityV1 | None = None

    def _binding(self) -> NativeBinding | None:
        with self._lock:
            if not self._native_checked:
                self._native, self._unavailable_reason = _read_native_binding()
                self._native_checked = True
            return self._native

    def _host_metal_observation(self) -> tuple[bool | None, str]:
        with self._lock:
            if self._host_metal is not None:
                return self._host_metal
            profile = self._host_profile
        system = profile.platform if profile is not None else platform.system().lower()
        observed = host_metal_capability(system)
        with self._lock:
            self._host_metal = observed
        return observed

    def _host_capability(self, host_observation: dict[str, Any] | None = None) -> tuple[bool | None, str]:
        if host_observation is not None and isinstance(host_observation.get("host_metal_capability"), bool):
            return (
                host_observation["host_metal_capability"],
                str(host_observation.get("host_metal_reason", "host v3 Metal observation")),
            )
        return self._host_metal_observation()

    def build_identity(self) -> BuildIdentityV1 | None:
        native = self._binding()
        if native is None:
            return None
        with self._lock:
            if self._build_identity is None:
                self._build_identity = BuildIdentityV1.from_components(
                    kind="llama-cpp-native-build-v1",
                    components={
                        "python_binding": {
                            "distribution": "llama-cpp-python",
                            "version": native.package_version,
                        },
                        "native_library": {
                            "implementation": "llama.cpp",
                            "version": native.native_version,
                            "commit": native.native_commit,
                            "system_info": native.system_info,
                            "path": native.library_path,
                            "sha256": native.library_sha256,
                            "gpu_offload": native.gpu_offload,
                            "metal": native.metal_build,
                        },
                    },
                )
            return self._build_identity

    def identity(self) -> EngineIdentity:
        native = self._binding()
        build_identity = self.build_identity()
        return EngineIdentity(
            engine=self.name,
            version=native.package_version if native else None,
            build=build_identity.fingerprint if build_identity else None,
        )

    def _loaded_template_status(self) -> str:
        with self._lock:
            if self._loaded is None:
                return "unknown"
            return "supported" if self._chat_formatter is not None else "unsupported"

    def discover_capability(self) -> EngineCapability:
        native = self._binding()
        available = native is not None
        reason = self._unavailable_reason if not available else None
        unsupported_reason = "llama.cpp does not implement this legacy per-request setting; use v3 LoadOptions"
        runtime_options = {
            path: {"status": "unavailable" if not available else "unsupported", "reason": reason or unsupported_reason}
            for path in RUNTIME_OPTION_PATHS
        }
        if available:
            runtime_options["context.context_length"] = {
                "status": "supported",
                "scope": "CONSTRAINT",
                "minimum": 1,
                "maximum": 1_048_576,
                "unit": "tokens",
                "reason": "legacy v1/v2 value remains a Foundation preflight budget",
            }
            runtime_options["acceleration.backend"] = {
                "status": "supported",
                "scope": "GENERATION",
                "supported_values": ["auto", "cpu"],
                "reason": "legacy contract loads llama.cpp with CPU settings; use v3 LoadOptions for Metal",
            }
        generation_options = {
            name: {
                "status": "supported" if available else "unavailable",
                **({} if available else {"reason": reason}),
            }
            for name in (
                "max_tokens",
                "temperature",
                "top_p",
                "top_k",
                "repetition_penalty",
                "repetition_window",
                "stop",
                "seed",
            )
        }
        template_status = self._loaded_template_status() if available else "unavailable"
        thinking_status = template_status
        with self._lock:
            variables = self._template_variables
        if template_status == "supported" and not set(_TEMPLATE_INTENT_NAMES["thinking"]) & variables:
            thinking_status = "unsupported"
        return EngineCapability(
            identity=self.identity(),
            available=available,
            streaming=available,
            cancellation=available,
            load_unload=available,
            chat_template=template_status,
            thinking_flag=thinking_status,
            artifact_formats=["gguf"],
            runtime_options=runtime_options,
            generation_options=generation_options,
            reason=reason,
            build_identity=self.build_identity(),
        )

    @staticmethod
    def _applicability(
        *,
        engine: EngineBindingV2,
        host_observation: dict[str, Any] | None,
        artifact: ArtifactBindingV2 | None,
        load_identity_fingerprint: str | None,
    ) -> CapabilityApplicabilityV3:
        from ..contracts_v2 import canonical_fingerprint

        return CapabilityApplicabilityV3(
            depends_on=(
                CapabilityDependency.ARTIFACT,
                CapabilityDependency.ENGINE,
                CapabilityDependency.ENGINE_BUILD,
                CapabilityDependency.HOST,
                CapabilityDependency.LOAD,
            ),
            artifact_identity=artifact.canonical_payload() if artifact is not None else None,
            engine_identity={
                "family": engine.family,
                "implementation": engine.implementation.to_dict(),
                "adapter_id": engine.adapter_id,
            },
            engine_build_identity=engine.build_identity.to_dict() if engine.build_identity else None,
            host_fingerprint=canonical_fingerprint(host_observation) if host_observation is not None else None,
            load_identity_fingerprint=load_identity_fingerprint,
        )

    def discover_capability_v3(
        self,
        *,
        artifact: ArtifactBindingV2 | None = None,
        host_observation: dict[str, Any] | None = None,
        load_identity_fingerprint: str | None = None,
    ) -> EngineCapabilityV3:
        native = self._binding()
        legacy = self.discover_capability()
        with self._lock:
            current_artifact = self._loaded_v2
            observed = self._observed_gguf
            variables = self._template_variables
            loaded = self._loaded is not None
        effective_artifact = artifact or current_artifact
        engine_binding = EngineBindingV2.from_legacy(
            self.identity(), adapter_id=self.name, build_identity=self.build_identity()
        )
        applicability = self._applicability(
            engine=engine_binding,
            host_observation=host_observation,
            artifact=effective_artifact,
            load_identity_fingerprint=load_identity_fingerprint,
        )
        available = native is not None

        def option(
            status: CapabilityStatus,
            scope: CapabilityScope,
            *,
            reason: str,
            allowed: tuple[Any, ...] = (),
            minimum: int | float | None = None,
            maximum: int | float | None = None,
            unit: str | None = None,
            requires_reload: bool | None = None,
            evidence: dict[str, Any] | None = None,
        ) -> OptionCapabilityV3:
            return OptionCapabilityV3(
                status=status,
                scope=scope,
                requires_reload=(scope == CapabilityScope.LOAD) if requires_reload is None else requires_reload,
                allowed_values=allowed,
                minimum=minimum,
                maximum=maximum,
                unit=unit,
                evidence=evidence or {},
                reason=reason,
                applicability=applicability,
            )

        unavailable = CapabilityStatus.UNAVAILABLE
        supported = CapabilityStatus.SUPPORTED
        unknown = CapabilityStatus.UNKNOWN
        unsupported = CapabilityStatus.UNSUPPORTED
        missing_status = unavailable if not available else unknown
        context_max = observed.context_length if observed is not None else None
        template_status = supported if self._loaded_template_status() == "supported" else (
            unsupported if loaded else missing_status
        )
        host_metal, host_reason = self._host_capability(host_observation)
        metal_build = native.metal_build if native is not None else None
        if not available:
            metal_status = unavailable
            metal_reason = self._unavailable_reason or "llama-cpp-python is not installed"
        elif metal_build is False or host_metal is False or (native is not None and native.gpu_offload is False):
            metal_status = unsupported
            metal_reason = "Metal requires a Metal-enabled native build, GPU offload support, and host Metal capability"
        elif metal_build is True and host_metal is True and native is not None and native.gpu_offload is True:
            metal_status = supported
            metal_reason = "native build reports Metal and GPU offload; host observation reports Metal"
        else:
            metal_status = unknown
            metal_reason = "native build Metal or host Metal evidence is unknown"
        backend_values: tuple[Any, ...] = ("auto", "cpu") + (("metal",) if metal_status == supported else ())
        native_type_values = self._kv_cache_types(native) if native is not None else ()
        layer_count = None
        if observed is not None:
            value = observed.metadata.get(f"{observed.architecture}.block_count")
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                layer_count = value

        options: dict[str, OptionCapabilityV3] = {
            "model_context_size": option(
                supported if available else unavailable,
                CapabilityScope.LOAD,
                reason="context size is passed to llama_context_params and read back from the native context",
                minimum=1,
                maximum=context_max or 1_048_576,
                unit="tokens",
                evidence={"native_observation": "llama_n_ctx", "artifact_context_limit": context_max},
            ),
            "batch": option(
                supported if available else unavailable,
                CapabilityScope.LOAD,
                reason="logical batch is passed to and read back from the native context",
                minimum=1,
                maximum=context_max or 1_048_576,
                unit="tokens",
                evidence={"native_observation": "llama_n_batch"},
            ),
            "ubatch": option(
                supported if available else unavailable,
                CapabilityScope.LOAD,
                reason="physical batch is passed to and read back from the native context",
                minimum=1,
                maximum=context_max or 1_048_576,
                unit="tokens",
                evidence={"native_observation": "llama_n_ubatch"},
            ),
            "threads": option(
                supported if available else unavailable,
                CapabilityScope.LOAD,
                reason="thread count is passed through the binding's context parameters",
                minimum=1,
                maximum=4096,
                unit="threads",
                evidence={"observation": "llama-cpp-python context configuration"},
            ),
            "kv_cache.key_type": option(
                supported if available and "f16" in native_type_values else unavailable if not available else unsupported,
                CapabilityScope.LOAD,
                reason="KV key type is selected through native context parameters",
                allowed=tuple(native_type_values),
                evidence={"native_constants": list(native_type_values)},
            ),
            "kv_cache.value_type": option(
                supported if available and "f16" in native_type_values else unavailable if not available else unsupported,
                CapabilityScope.LOAD,
                reason="KV value type is selected through native context parameters",
                allowed=tuple(native_type_values),
                evidence={"native_constants": list(native_type_values)},
            ),
            "acceleration.backend": option(
                missing_status if not available else supported,
                CapabilityScope.LOAD,
                reason=metal_reason if metal_status != supported else "CPU and explicitly requested Metal are available",
                allowed=backend_values,
                evidence={
                    "host_metal": host_metal,
                    "host_reason": host_reason,
                    "native_build_metal": metal_build,
                    "native_gpu_offload": native.gpu_offload if native else None,
                    "metal_capability_status": metal_status.value,
                },
            ),
            "acceleration.device": option(
                unavailable if not available else unsupported,
                CapabilityScope.LOAD,
                reason="the binding does not expose a stable Foundation device identifier",
            ),
            "acceleration.gpu_offload_layers": option(
                supported if available else unavailable,
                CapabilityScope.LOAD,
                reason=(
                    metal_reason
                    if metal_status == supported
                    else "CPU zero-layer execution is supported; positive GPU offload requires proven Metal capability"
                ),
                allowed=("all",) + tuple(range(0, layer_count + 1))
                if metal_status == supported and layer_count is not None
                else ("all", 0) if metal_status == supported else (0,),
                minimum=0,
                maximum=layer_count if metal_status == supported else 0,
                unit="layers",
                evidence={
                    "artifact_layer_count": layer_count,
                    "native_build_metal": metal_build,
                    "native_gpu_offload": native.gpu_offload if native else None,
                    "metal_status": metal_status.value,
                },
            ),
            "max_context_tokens": option(
                template_status,
                CapabilityScope.CONSTRAINT,
                reason="exact token counting uses the selected GGUF chat formatter and native tokenizer",
                minimum=1,
                maximum=context_max or 1_048_576,
                unit="tokens",
                requires_reload=False,
                evidence={"chat_template_available": template_status == supported},
            ),
        }
        generation_supported = supported if available else unavailable
        for name, unit, minimum, maximum in (
            ("max_tokens", "tokens", 1, 131_072),
            ("temperature", None, 0, 2),
            ("top_p", None, 0, 1),
            ("top_k", "tokens", 1, 1_000_000),
            ("repetition_penalty", None, 0, None),
            ("repetition_window", "tokens", 0, 1_048_576),
            ("stop", None, None, None),
            ("seed", None, 0, 2**63 - 1),
        ):
            if name == "repetition_window":
                options[name] = option(
                    unavailable if not available else unsupported,
                    CapabilityScope.GENERATION,
                    reason="llama-cpp-python exposes repetition history as load-time last_n_tokens_size, not a per-request setting",
                    minimum=minimum,
                    maximum=maximum,
                    unit=unit,
                    requires_reload=False,
                    evidence={"binding_parameter": "last_n_tokens_size", "scope": "LOAD"},
                )
                continue
            options[name] = option(
                generation_supported,
                CapabilityScope.GENERATION,
                reason="mapped to llama-cpp-python create_completion",
                minimum=minimum,
                maximum=maximum,
                unit=unit,
                requires_reload=False,
                evidence={
                    "binding_parameter": {
                        "repetition_penalty": "repeat_penalty",
                        "repetition_window": "repeat_last_n",
                    }.get(name, name),
                },
            )

        def feature_status(key: str) -> CapabilityStatus:
            if not available:
                return unavailable
            if not loaded:
                return unknown
            return supported if any(name in variables for name in _TEMPLATE_INTENT_NAMES[key]) else unsupported

        def feature(status: CapabilityStatus, reason: str, values: tuple[Any, ...] = ()) -> OptionCapabilityV3:
            return option(
                status,
                CapabilityScope.GENERATION,
                reason=reason,
                allowed=values,
                requires_reload=False,
                evidence={"chat_template_variables": sorted(variables)},
            )

        selected_layer_values = tuple(range(0, layer_count + 1)) if layer_count is not None else ()
        options["acceleration.gpu_offload_layers"] = option(
            supported if available else unavailable,
            CapabilityScope.LOAD,
            reason=(
                metal_reason
                if metal_status == supported
                else "CPU zero-layer execution is supported; positive GPU offload requires proven Metal capability"
            ),
            allowed=("all",) + selected_layer_values
            if metal_status == supported and layer_count is not None
            else ("all", 0) if metal_status == supported else (0,),
            minimum=0,
            maximum=layer_count if metal_status == supported else 0,
            unit="layers",
            evidence={
                "artifact_layer_count": layer_count,
                "native_build_metal": metal_build,
                "native_gpu_offload": native.gpu_offload if native else None,
                "metal_status": metal_status.value,
            },
        )
        return EngineCapabilityV3(
            engine=engine_binding,
            status=supported if available else unavailable,
            options=options,
            streaming=supported if available else unavailable,
            cancellation=supported if available else unavailable,
            load_unload=supported if available else unavailable,
            chat_template=feature(
                template_status,
                "selected metadata chat template is available" if template_status == supported else "a usable GGUF chat template is required",
                ("tokenizer.chat_template",),
            ),
            thinking_supported=feature(
                feature_status("thinking"),
                "template has a thinking intent variable" if feature_status("thinking") == supported else "template cannot prove thinking intent control",
                ("OFF", "ON"),
            ),
            thinking_effort_supported=feature(
                feature_status("effort"),
                "template has a thinking effort variable" if feature_status("effort") == supported else "template cannot prove thinking effort control",
                ("LOW", "MEDIUM", "HIGH"),
            ),
            thinking_budget_supported=feature(
                feature_status("budget"),
                "template has a thinking budget variable" if feature_status("budget") == supported else "template cannot prove thinking budget control",
                (),
            ),
            reason=legacy.reason,
        )

    @staticmethod
    def _kv_cache_types(native: NativeBinding | None) -> tuple[str, ...]:
        if native is None:
            return ()
        return tuple(
            name
            for name in _GGML_KV_TYPES
            if getattr(native.api, f"GGML_TYPE_{name.upper()}", None) is not None
        )

    @staticmethod
    def _kv_type_id(native: NativeBinding, value: str) -> int:
        normalized = value.strip().lower()
        if normalized not in LlamaCppAdapter._kv_cache_types(native):
            raise UnsupportedRuntimeOptionError(
                "the selected llama.cpp build does not expose this KV cache type",
                details={"path": "load_options.kv_cache", "value": value},
            )
        return int(getattr(native.api, f"GGML_TYPE_{normalized.upper()}"))

    def _artifact_metadata(self, artifact: ModelArtifactBinding) -> GGUFObservedMetadata:
        if (artifact.format or "").lower() != "gguf":
            raise ArtifactCompatibilityError(
                "llama.cpp execution requires artifact format gguf",
                details={"artifact_id": artifact.artifact_id, "format": artifact.format},
            )
        artifact_v2 = ArtifactBindingV2.from_legacy(artifact)
        with VerifiedGGUFArtifact.open(artifact_v2) as verified:
            return verified.observed

    def resolve_load_options(self, artifact: ModelArtifactBinding, options: LoadOptions) -> LoadOptionsResolutionV1:
        observed = self._artifact_metadata(artifact)
        return self.resolve_load_options_with_observation(artifact, options, observed)

    def resolve_load_options_with_observation(
        self,
        artifact: ModelArtifactBinding,
        options: LoadOptions,
        artifact_observation: Any,
    ) -> LoadOptionsResolutionV1:
        native = self._binding()
        if native is None:
            raise EngineUnavailableError(
                "llama-cpp-python native binding is unavailable",
                details={"engine": self.name, "reason": self._unavailable_reason},
            )
        requested = LoadOptions.from_payload(options.to_dict())
        if not isinstance(artifact_observation, GGUFObservedMetadata):
            raise ArtifactCompatibilityError(
                "llama.cpp LoadOptions resolution requires validated GGUF metadata",
                details={"artifact_id": artifact.artifact_id},
            )
        observed = artifact_observation
        model_context = requested.model_context_size or observed.context_length
        if model_context is None:
            raise UnsupportedRuntimeOptionError(
                "GGUF metadata did not declare a model context size; provide LoadOptions.model_context_size",
                details={"artifact_id": artifact.artifact_id, "path": "model_context_size", "generation_started": False},
            )
        if observed.context_length is not None and model_context > observed.context_length:
            raise UnsupportedRuntimeOptionError(
                "requested model_context_size exceeds the GGUF trained context and no extension option is declared",
                details={
                    "path": "model_context_size",
                    "requested": model_context,
                    "maximum": observed.context_length,
                },
            )
        batch = requested.batch if requested.batch is not None else min(model_context, 512)
        if batch > model_context:
            raise UnsupportedRuntimeOptionError(
                "load_options.batch cannot exceed model_context_size",
                details={"path": "batch", "requested": batch, "model_context_size": model_context},
            )
        ubatch = requested.ubatch if requested.ubatch is not None else min(batch, 512)
        threads = requested.threads if requested.threads is not None else max((os.cpu_count() or 1) // 2, 1)
        if requested.acceleration.device is not None:
            raise UnsupportedRuntimeOptionError(
                "llama.cpp adapter does not expose a stable device identifier",
                details={"path": "acceleration.device", "requested": requested.acceleration.device},
            )
        backend = requested.acceleration.backend
        if backend == "auto":
            backend = "cpu"
        layer_request = requested.acceleration.gpu_offload_layers
        if backend == "metal":
            host_metal, host_reason = self._host_metal_observation()
            if native.metal_build is not True or host_metal is not True:
                raise UnsupportedRuntimeOptionError(
                    "Metal execution requires both a Metal-enabled llama.cpp build and host Metal capability",
                    details={
                        "path": "acceleration.backend",
                        "native_build_metal": native.metal_build,
                        "host_metal": host_metal,
                        "host_reason": host_reason,
                    },
                )
            if native.gpu_offload is not True:
                raise UnsupportedRuntimeOptionError(
                    "loaded llama.cpp build does not report GPU offload support",
                    details={"path": "acceleration.backend", "native_gpu_offload": native.gpu_offload},
                )
            if layer_request is None:
                layer_request = "all"
            if layer_request == 0:
                raise UnsupportedRuntimeOptionError(
                    "Metal backend requires a positive or all-layer GPU offload request",
                    details={"path": "acceleration.gpu_offload_layers", "requested": layer_request},
                )
            layer_count = observed.metadata.get(f"{observed.architecture}.block_count")
            if isinstance(layer_request, int) and isinstance(layer_count, int) and layer_request > layer_count:
                raise UnsupportedRuntimeOptionError(
                    "requested GPU offload layers exceed the GGUF architecture layer count",
                    details={"path": "acceleration.gpu_offload_layers", "requested": layer_request, "maximum": layer_count},
                )
        else:
            if layer_request not in {None, 0}:
                raise UnsupportedRuntimeOptionError(
                    "CPU backend cannot offload layers to a GPU",
                    details={"path": "acceleration.gpu_offload_layers", "requested": layer_request},
                )
            layer_request = 0

        kv_types = self._kv_cache_types(native)
        key_type = requested.kv_cache.key_type or "f16"
        value_type = requested.kv_cache.value_type or "f16"
        if key_type not in kv_types:
            raise UnsupportedRuntimeOptionError(
                "llama.cpp does not expose the requested KV key type",
                details={"path": "kv_cache.key_type", "requested": key_type, "supported_values": list(kv_types)},
            )
        if value_type not in kv_types:
            raise UnsupportedRuntimeOptionError(
                "llama.cpp does not expose the requested KV value type",
                details={"path": "kv_cache.value_type", "requested": value_type, "supported_values": list(kv_types)},
            )
        resolved = LoadOptions(
            model_context_size=model_context,
            batch=batch,
            ubatch=ubatch,
            threads=threads,
            kv_cache=KVCacheConfiguration(key_type=key_type, value_type=value_type),
            acceleration=AccelerationConfiguration(
                backend=backend,
                device=None,
                gpu_offload_layers=layer_request,
            ),
        )
        requested_values = {
            "model_context_size": requested.model_context_size,
            "batch": requested.batch,
            "ubatch": requested.ubatch,
            "threads": requested.threads,
            "kv_cache.key_type": requested.kv_cache.key_type,
            "kv_cache.value_type": requested.kv_cache.value_type,
            "acceleration.backend": requested.acceleration.backend,
            "acceleration.gpu_offload_layers": requested.acceleration.gpu_offload_layers,
        }
        resolved_values = {
            "model_context_size": resolved.model_context_size,
            "batch": resolved.batch,
            "ubatch": resolved.ubatch,
            "threads": resolved.threads,
            "kv_cache.key_type": resolved.kv_cache.key_type,
            "kv_cache.value_type": resolved.kv_cache.value_type,
            "acceleration.backend": resolved.acceleration.backend,
            "acceleration.gpu_offload_layers": resolved.acceleration.gpu_offload_layers,
        }
        reasons = {
            "model_context_size": "null resolves to the validated GGUF architecture context length",
            "batch": "null resolves to min(model_context_size, 512) and is passed explicitly",
            "ubatch": "null resolves to min(batch, 512) and is passed explicitly",
            "threads": "null resolves to the binding's default CPU thread policy and is passed explicitly",
            "kv_cache.key_type": "null resolves to native f16 and is passed explicitly",
            "kv_cache.value_type": "null resolves to native f16 and is passed explicitly",
            "acceleration.backend": "auto resolves to CPU; Metal requires an explicit request",
            "acceleration.gpu_offload_layers": "null resolves to zero for CPU or all layers for explicit Metal",
        }
        resolutions = tuple(
            OptionResolutionV3(
                path=path,
                requested=requested_values[path],
                resolved=resolved_values[path],
                effective=resolved_values[path],
                status="resolved",
                reason=reasons[path],
            )
            for path in requested_values
            if requested_values[path] != resolved_values[path]
        )
        return LoadOptionsResolutionV1(
            requested=requested,
            resolved=resolved,
            effective=resolved,
            resolutions=resolutions,
        )

    def _effective_native_state(self, native: NativeBinding, llama: Any, options: LoadOptions) -> dict[str, Any]:
        context_size = self._native_int(llama, "n_ctx", native, "llama_n_ctx")
        batch = self._native_int(llama, "n_batch", native, "llama_n_batch")
        ubatch = self._native_int(llama, "n_ubatch", native, "llama_n_ubatch")
        if (context_size, batch, ubatch) != (options.model_context_size, options.batch, options.ubatch):
            raise UnsupportedRuntimeOptionError(
                "llama.cpp effective context/batch values differ from the resolved LoadOptions",
                details={
                    "requested_effective": {
                        "model_context_size": options.model_context_size,
                        "batch": options.batch,
                        "ubatch": options.ubatch,
                    },
                    "native_effective": {"model_context_size": context_size, "batch": batch, "ubatch": ubatch},
                },
            )
        wrapper_threads = getattr(llama, "n_threads", None)
        if not isinstance(wrapper_threads, int) or wrapper_threads != options.threads:
            raise EngineUnavailableError(
                "llama-cpp-python did not expose its effective configured thread count",
                details={"requested_threads": options.threads, "observed_threads": wrapper_threads},
            )
        context_params = getattr(llama, "context_params", None)
        model_params = getattr(llama, "model_params", None)
        if context_params is None or model_params is None:
            raise EngineUnavailableError("llama-cpp-python did not expose effective load parameter evidence")
        key_id = self._native_int_value(getattr(context_params, "type_k", None))
        value_id = self._native_int_value(getattr(context_params, "type_v", None))
        expected_key_id = self._kv_type_id(native, options.kv_cache.key_type or "f16")
        expected_value_id = self._kv_type_id(native, options.kv_cache.value_type or "f16")
        if key_id != expected_key_id or value_id != expected_value_id:
            raise UnsupportedRuntimeOptionError(
                "native KV cache types differ from resolved LoadOptions",
                details={
                    "expected": {"key_type": expected_key_id, "value_type": expected_value_id},
                    "observed": {"key_type": key_id, "value_type": value_id},
                },
            )
        expected_layers = 0x7FFFFFFF if options.acceleration.gpu_offload_layers == "all" else int(
            options.acceleration.gpu_offload_layers or 0
        )
        observed_layers = self._native_int_value(getattr(model_params, "n_gpu_layers", None))
        if observed_layers != expected_layers:
            raise UnsupportedRuntimeOptionError(
                "native GPU offload layer setting differs from resolved LoadOptions",
                details={"expected": expected_layers, "observed": observed_layers},
            )
        return {
            "model_context_size": {"value": context_size, "source": "llama_n_ctx"},
            "batch": {"value": batch, "source": "llama_n_batch"},
            "ubatch": {"value": ubatch, "source": "llama_n_ubatch"},
            "threads": {"value": wrapper_threads, "source": "llama-cpp-python context configuration"},
            "kv_cache": {
                "key_type": {"value": options.kv_cache.key_type, "native_type_id": key_id, "source": "context_params"},
                "value_type": {"value": options.kv_cache.value_type, "native_type_id": value_id, "source": "context_params"},
            },
            "acceleration": {
                "backend": options.acceleration.backend,
                "gpu_offload_layers": options.acceleration.gpu_offload_layers,
                "native_n_gpu_layers": observed_layers,
                "actual_kernel_execution": (
                    "not_applicable_cpu_selected"
                    if options.acceleration.backend == "cpu"
                    else "not_observed"
                ),
            },
        }

    @staticmethod
    def _native_int_value(value: Any) -> int | None:
        if isinstance(value, bool):
            return int(value)
        raw = getattr(value, "value", value)
        return raw if isinstance(raw, int) else None

    def _native_int(self, llama: Any, method_name: str, native: NativeBinding, c_function: str) -> int:
        method = getattr(llama, method_name, None)
        if callable(method):
            value = self._native_int_value(method())
            if value is not None:
                return value
        native_function = getattr(native.api, c_function, None)
        context = getattr(llama, "ctx", None)
        if callable(native_function) and context is not None:
            value = self._native_int_value(native_function(context))
            if value is not None:
                return value
        raise EngineUnavailableError(
            "loaded llama.cpp binding cannot observe an effective context setting",
            details={"setting": method_name, "native_function": c_function},
        )

    def _load_native(
        self,
        artifact: ModelArtifactBinding,
        options: LoadOptions,
        *,
        load_resolutions: tuple[OptionResolutionV3, ...],
        verified_artifact: VerifiedGGUFArtifact | None = None,
    ) -> dict[str, Any]:
        native = self._binding()
        if native is None:
            raise EngineUnavailableError(
                "llama-cpp-python native binding is unavailable",
                details={"engine": self.name, "reason": self._unavailable_reason},
            )
        artifact_v2 = ArtifactBindingV2.from_legacy(artifact)
        verified = verified_artifact.duplicate() if verified_artifact is not None else VerifiedGGUFArtifact.open(artifact_v2)
        llama = None
        started = time.perf_counter()
        try:
            verified.revalidate()
            kwargs: dict[str, Any] = {
                "model_path": verified.pinned_path,
                "n_ctx": options.model_context_size,
                "n_batch": options.batch,
                "n_ubatch": options.ubatch,
                "n_threads": options.threads,
                "n_gpu_layers": -1 if options.acceleration.gpu_offload_layers == "all" else options.acceleration.gpu_offload_layers,
                "type_k": self._kv_type_id(native, options.kv_cache.key_type or "f16"),
                "type_v": self._kv_type_id(native, options.kv_cache.value_type or "f16"),
                "chat_format": "chat_template.default",
                "verbose": False,
                "use_mmap": True,
            }
            llama, native_load_log = _construct_native_model(native, kwargs)
            verified.revalidate()
            effective_state = self._effective_native_state(native, llama, options)
            metal_observation = _native_metal_load_observation(native_load_log)
            acceleration_state = effective_state["acceleration"]
            if options.acceleration.backend == "metal":
                acceleration_state["native_metal_load_observation"] = metal_observation
                acceleration_state["actual_kernel_execution"] = (
                    "pending_generation" if metal_observation["status"] == "confirmed" else "not_observed"
                )
            self._validate_native_metadata(verified.observed, getattr(llama, "metadata", {}))
            formatter, template_text, variables = self._select_chat_formatter(native, llama)
            with self._lock:
                old_verified = self._verified_gguf
                self._llama = llama
                self._loaded = artifact
                self._loaded_v2 = replace_artifact_identity(artifact_v2, verified.observed)
                self._load_options = options
                self._verified_gguf = verified
                self._observed_gguf = verified.observed
                self._chat_formatter = formatter
                self._chat_template_text = template_text
                self._template_variables = variables
                self._last_metrics = {}
                self._selected_execution_acceleration = {
                    "engine": self.name,
                    "status": "configured",
                    "backend": options.acceleration.backend,
                    "gpu_offload_layers": options.acceleration.gpu_offload_layers,
                    "native_build_metal": native.metal_build,
                    "actual_kernel_execution": (
                        "pending_generation"
                        if options.acceleration.backend == "metal" and metal_observation["status"] == "confirmed"
                        else "not_observed"
                        if options.acceleration.backend == "metal"
                        else "not_applicable_cpu_selected"
                    ),
                    "native_metal_load_observation": (
                        metal_observation if options.acceleration.backend == "metal" else None
                    ),
                }
            if old_verified is not None:
                old_verified.close()
            return {
                "loaded": True,
                "artifact_id": artifact.artifact_id,
                "artifact_identity": verified.observed.content_identity.to_dict(),
                "gguf_validation": verified.observed.to_dict(),
                "load_options_resolution": {
                    "requested": options.to_dict(),
                    "resolved": options.to_dict(),
                    "effective": options.to_dict(),
                    "resolutions": [item.to_dict() for item in load_resolutions],
                },
                "effective_load_options": options.to_dict(),
                "observed_effective_load_state": effective_state,
                "engine_build_identity": self.build_identity().to_dict() if self.build_identity() else None,
                "selected_execution_acceleration": dict(self._selected_execution_acceleration or {}),
                "load_duration_ms": (time.perf_counter() - started) * 1000,
                "measurement_provenance": "observed",
            }
        except RuntimeFoundationError:
            if llama is not None:
                self._close_native(llama)
            verified.close()
            raise
        except Exception as exc:  # noqa: BLE001 - normalize Python/native errors at the adapter boundary
            if llama is not None:
                self._close_native(llama)
            verified.close()
            raise EngineRuntimeError(
                "llama.cpp GGUF load failed",
                details={
                    "engine": self.name,
                    "artifact_id": artifact.artifact_id,
                    "failure_kind": self._failure_kind(exc, phase="load"),
                    "native_exception": type(exc).__name__,
                },
            ) from exc

    @staticmethod
    def _close_native(llama: Any) -> None:
        close = getattr(llama, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001 - cleanup outcome is returned separately where possible
                pass

    @staticmethod
    def _validate_native_metadata(observed: GGUFObservedMetadata, native_metadata: Any) -> None:
        if not isinstance(native_metadata, dict):
            raise EngineUnavailableError("llama.cpp binding did not expose loaded GGUF metadata")
        claims = {
            "general.architecture": observed.architecture,
            "general.file_type": observed.file_type,
        }
        if observed.context_length is not None:
            claims[f"{observed.architecture}.context_length"] = observed.context_length
        for key, expected in claims.items():
            actual = native_metadata.get(key)
            matches = actual == expected
            if isinstance(expected, int) and not isinstance(expected, bool) and isinstance(actual, str):
                try:
                    matches = int(actual, 10) == expected
                except ValueError:
                    matches = False
            if actual is not None and not matches:
                raise ArtifactCompatibilityError(
                    "native llama.cpp metadata differs from the validated GGUF metadata",
                    details={"metadata_key": key, "observed": expected, "native": actual},
                )
        if observed.chat_template is not None:
            native_template = native_metadata.get("tokenizer.chat_template")
            if native_template is None:
                native_template = native_metadata.get("tokenizer.chat_template.default")
            if isinstance(observed.chat_template, str):
                matches = native_template == observed.chat_template
            else:
                matches = native_template in observed.chat_template
            if not matches:
                raise ArtifactCompatibilityError(
                    "native llama.cpp chat-template metadata differs from the validated GGUF metadata",
                    details={
                        "metadata_key": "tokenizer.chat_template",
                        "observed_chat_template_count": (
                            len(observed.chat_template)
                            if isinstance(observed.chat_template, tuple)
                            else 1
                        ),
                        "native_template_available": isinstance(native_template, str),
                    },
                )

    @staticmethod
    def _special_token_text(llama: Any, token_id: int) -> str:
        if token_id < 0:
            return ""
        model = getattr(llama, "_model", None)
        token_get_text = getattr(model, "token_get_text", None)
        if not callable(token_get_text):
            raise EngineUnavailableError(
                "llama-cpp-python cannot read special token text required by the GGUF chat template",
                details={"token_id": token_id},
            )
        value = token_get_text(token_id)
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, str):
            return value
        raise EngineUnavailableError(
            "llama-cpp-python returned an invalid special token value",
            details={"token_id": token_id, "observed_type": type(value).__name__},
        )

    @staticmethod
    def _select_chat_formatter(
        native: NativeBinding, llama: Any
    ) -> tuple[Callable[..., Any] | None, str | None, frozenset[str]]:
        metadata = getattr(llama, "metadata", {})
        template = None
        if isinstance(metadata, dict):
            candidate = metadata.get("tokenizer.chat_template")
            if isinstance(candidate, str):
                template = candidate
            if template is None:
                candidate = metadata.get("tokenizer.chat_template.default")
                if isinstance(candidate, str):
                    template = candidate
        if isinstance(template, str):
            formatter_module = native.chat_format_module
            formatter_class = getattr(formatter_module, "Jinja2ChatFormatter", None)
            if not callable(formatter_class):
                raise EngineUnavailableError(
                    "llama-cpp-python does not expose its GGUF Jinja chat formatter"
                )
            eos_method = getattr(llama, "token_eos", None)
            bos_method = getattr(llama, "token_bos", None)
            if not callable(eos_method) or not callable(bos_method):
                raise EngineUnavailableError(
                    "llama-cpp-python does not expose EOS and BOS token identities for GGUF formatting"
                )
            eos_id = eos_method()
            bos_id = bos_method()
            if not isinstance(eos_id, int) or isinstance(eos_id, bool):
                raise EngineUnavailableError("llama-cpp-python returned an invalid EOS token identity")
            if not isinstance(bos_id, int) or isinstance(bos_id, bool):
                raise EngineUnavailableError("llama-cpp-python returned an invalid BOS token identity")
            eos_token = LlamaCppAdapter._special_token_text(llama, eos_id)
            bos_token = LlamaCppAdapter._special_token_text(llama, bos_id)
            try:
                formatter = formatter_class(
                    template=template,
                    eos_token=eos_token,
                    bos_token=bos_token,
                    stop_token_ids=[eos_id] if eos_id >= 0 else None,
                )
                compiled_template = getattr(formatter, "_environment", None)
                jinja_environment = getattr(compiled_template, "environment", None)
                if jinja_environment is None:
                    raise TypeError("native chat formatter does not expose its compiled Jinja environment")
                from jinja2 import meta

                syntax_tree = jinja_environment.parse(template)
                variables = frozenset(meta.find_undeclared_variables(syntax_tree))
            except (ImportError, ValueError, TypeError) as exc:
                raise EngineUnavailableError(
                    "llama-cpp-python could not inspect the selected GGUF chat template",
                    details={"failure_kind": "template_inspection_failure", "reason": str(exc)},
                ) from exc
            if not callable(formatter):
                raise EngineUnavailableError("llama-cpp-python returned an invalid GGUF chat formatter")
            return formatter, template, variables
        return None, template, frozenset()

    def load(self, artifact: ModelArtifactBinding) -> dict[str, Any]:
        artifact_v2 = ArtifactBindingV2.from_legacy(artifact)
        verified = VerifiedGGUFArtifact.open(artifact_v2)
        try:
            resolution = self.resolve_load_options_with_observation(artifact, LoadOptions(), verified.observed)
            return self.load_verified_gguf(artifact, resolution, verified)
        finally:
            verified.close()

    def load_with_options(
        self, artifact: ModelArtifactBinding, resolution: LoadOptionsResolutionV1
    ) -> dict[str, Any]:
        if not isinstance(resolution, LoadOptionsResolutionV1):
            raise UnsupportedRuntimeOptionError("llama.cpp requires a validated v3 LoadOptions resolution")
        artifact_v2 = ArtifactBindingV2.from_legacy(artifact)
        verified = VerifiedGGUFArtifact.open(artifact_v2)
        try:
            return self.load_verified_gguf(artifact, resolution, verified)
        finally:
            verified.close()

    def load_verified_gguf(
        self,
        artifact: ModelArtifactBinding,
        resolution: LoadOptionsResolutionV1,
        verified_artifact: VerifiedGGUFArtifact,
    ) -> dict[str, Any]:
        """Load from the already validated artifact's exact file identity."""

        if not isinstance(resolution, LoadOptionsResolutionV1):
            raise UnsupportedRuntimeOptionError("llama.cpp requires a validated v3 LoadOptions resolution")
        if verified_artifact.observed.format != "gguf":
            raise ArtifactCompatibilityError("llama.cpp can load only validated GGUF artifacts")
        expected_resolution = self.resolve_load_options_with_observation(
            artifact,
            resolution.requested,
            verified_artifact.observed,
        )
        if (
            expected_resolution.resolved != resolution.resolved
            or expected_resolution.effective != resolution.effective
            or expected_resolution.resolutions != resolution.resolutions
        ):
            raise UnsupportedRuntimeOptionError(
                "llama.cpp received LoadOptions evidence that does not match its validated resolution",
                details={"artifact_id": artifact.artifact_id, "generation_started": False},
            )
        return self._load_native(
            artifact,
            resolution.effective,
            load_resolutions=resolution.resolutions,
            verified_artifact=verified_artifact,
        )

    def validate_execution_input(self, execution_input: ExecutionInputV1) -> None:
        if execution_input.adapters:
            raise UnsupportedExecutionInputError(
                "llama.cpp direct Base plus Adapter execution is not part of this phase",
                details={"engine": self.name, "adapter_count": len(execution_input.adapters)},
            )

    def load_execution_input_with_options(
        self, execution_input: ExecutionInputV1, resolution: LoadOptionsResolutionV1
    ) -> dict[str, Any]:
        self.validate_execution_input(execution_input)
        artifact = execution_input.base.to_legacy()
        result = self.load_with_options(artifact, resolution)
        return {
            **result,
            "execution_input_kind": execution_input.kind,
            "execution_input_fingerprint": execution_input.fingerprint,
            "adapter_count": 0,
        }

    def unload(self, artifact_id: str) -> dict[str, Any]:
        with self._lock:
            loaded = self._loaded
            llama = getattr(self, "_llama", None)
            verified = self._verified_gguf
            if loaded is None or loaded.artifact_id != artifact_id:
                return {"unloaded": False, "noop": True, "artifact_id": artifact_id}
            if self._active:
                raise RuntimeBusyError(
                    "cannot unload llama.cpp while a generation request is active",
                    details={"active_request_ids": sorted(self._active)},
                )
            self._loaded = None
            self._loaded_v2 = None
            self._load_options = None
            self._llama = None
            self._verified_gguf = None
            self._observed_gguf = None
            self._chat_formatter = None
            self._chat_template_text = None
            self._template_variables = frozenset()
        cleanup_error = None
        started = time.perf_counter()
        if llama is not None:
            close = getattr(llama, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001 - expose native cleanup status
                    cleanup_error = type(exc).__name__
        if verified is not None:
            try:
                verified.close()
            except OSError as exc:
                cleanup_error = cleanup_error or type(exc).__name__
        return {
            "unloaded": True,
            "artifact_id": artifact_id,
            "cleanup_status": "cleanup_error" if cleanup_error else "clean",
            "cleanup_error_type": cleanup_error,
            "unload_duration_ms": (time.perf_counter() - started) * 1000,
        }

    def resolve_runtime_options(self, options: RuntimeOptions) -> RuntimeSettingsResolution:
        if self._binding() is None:
            raise EngineUnavailableError(
                "llama-cpp-python native binding is unavailable",
                details={"engine": self.name, "reason": self._unavailable_reason},
            )
        requested = RuntimeOptions.from_payload(options.to_dict())
        defaults = RuntimeOptions().to_dict()
        values = requested.to_dict()
        unsupported_paths = [
            f"{section}.{name}"
            for section in ("kv_cache", "prefill", "prompt_cache")
            for name, value in values[section].items()
            if value != defaults[section][name]
        ]
        unsupported_paths.extend(
            [
                "context.sliding_window"
                if values["context"]["sliding_window"] is not None
                else "",
                "acceleration.device" if values["acceleration"]["device"] is not None else "",
                "acceleration.threads" if values["acceleration"]["threads"] is not None else "",
                "engine_options" if values["engine_options"] else "",
            ]
        )
        unsupported_paths = [path for path in unsupported_paths if path]
        if requested.acceleration.backend not in {"auto", "cpu"}:
            unsupported_paths.append("acceleration.backend")
        if unsupported_paths:
            raise UnsupportedRuntimeOptionError(
                "legacy runtime options cannot change llama.cpp load state; use v3 LoadOptions",
                details={"unsupported_paths": sorted(set(unsupported_paths)), "generation_started": False},
            )
        effective_values = requested.to_dict()
        effective_values["acceleration"]["backend"] = "cpu"
        effective = RuntimeOptions.from_payload(effective_values)
        status = {
            "context.context_length": "supported",
            "acceleration.backend": "resolved",
            "other": "default_or_unsupported",
        }
        warnings = []
        if requested.acceleration.backend == "auto":
            warnings.append("legacy acceleration.backend auto resolved to CPU; use v3 LoadOptions for explicit Metal")
        return RuntimeSettingsResolution(requested=requested, effective=effective, option_status=status, warnings=warnings)

    def _loaded_state(self, request: GenerationRequest) -> tuple[Any, ModelArtifactBinding, LoadOptions]:
        with self._lock:
            llama = getattr(self, "_llama", None)
            loaded = self._loaded
            options = self._load_options
        if llama is None or loaded is None or options is None:
            raise EngineUnavailableError("llama.cpp has no loaded GGUF artifact")
        if request.model_artifact_id != loaded.artifact_id:
            raise ArtifactCompatibilityError(
                "generation request artifact does not match the loaded GGUF",
                details={"requested_artifact_id": request.model_artifact_id, "loaded_artifact_id": loaded.artifact_id},
            )
        return llama, loaded, options

    def _thinking_parameters(
        self,
        intent: ThinkingIntent,
        *,
        studio_resolved: ThinkingIntent | None = None,
    ) -> tuple[ThinkingIntent, dict[str, Any]]:
        resolved = intent
        if intent.mode == ThinkingMode.AUTO:
            if studio_resolved is None or studio_resolved.mode not in {ThinkingMode.ON, ThinkingMode.OFF}:
                raise ThinkingResolutionError(
                    "AUTO thinking requires an explicit Studio ON/OFF resolution",
                    details={"requested": intent.to_dict(), "studio_resolved": studio_resolved.to_dict() if studio_resolved else None},
                )
            resolved = ThinkingIntent(
                mode=studio_resolved.mode,
                effort=intent.effort or studio_resolved.effort,
                budget_tokens=intent.budget_tokens if intent.budget_tokens is not None else studio_resolved.budget_tokens,
            )
        with self._lock:
            variables = self._template_variables
        if not variables:
            raise UnsupportedGenerationSettingError(
                "the loaded GGUF chat template does not expose a verifiable thinking control",
                details={"engine": self.name, "requested": resolved.to_dict()},
            )
        kwargs: dict[str, Any] = {}
        thinking_name = next((name for name in _TEMPLATE_INTENT_NAMES["thinking"] if name in variables), None)
        if thinking_name is None:
            raise UnsupportedGenerationSettingError(
                "the selected GGUF chat template cannot represent the requested Thinking Intent",
                details={"requested": resolved.to_dict(), "required_template_variable": list(_TEMPLATE_INTENT_NAMES["thinking"])},
            )
        kwargs[thinking_name] = resolved.mode == ThinkingMode.ON
        if resolved.effort is not None:
            effort_name = next((name for name in _TEMPLATE_INTENT_NAMES["effort"] if name in variables), None)
            if effort_name is None:
                raise UnsupportedGenerationSettingError(
                    "the selected GGUF chat template cannot represent Thinking effort",
                    details={"requested": resolved.to_dict(), "required_template_variable": list(_TEMPLATE_INTENT_NAMES["effort"])},
                )
            kwargs[effort_name] = resolved.effort.value.lower()
        if resolved.budget_tokens is not None:
            budget_name = next((name for name in _TEMPLATE_INTENT_NAMES["budget"] if name in variables), None)
            if budget_name is None:
                raise UnsupportedGenerationSettingError(
                    "the selected GGUF chat template cannot represent Thinking budget",
                    details={"requested": resolved.to_dict(), "required_template_variable": list(_TEMPLATE_INTENT_NAMES["budget"])},
                )
            kwargs[budget_name] = resolved.budget_tokens
        return resolved, kwargs

    def _generation_options(self, request: GenerationRequest) -> tuple[GenerationOptions, ThinkingIntent | None]:
        if isinstance(request, GenerationRequestV2):
            options = GenerationOptions.from_legacy_v2(request)
            return options, request.studio_resolved_thinking
        options = GenerationOptions.from_legacy_v1(request)
        return options, None

    def _native_generation_defaults(self) -> dict[str, Any]:
        with self._lock:
            llama = getattr(self, "_llama", None)
        completion = getattr(llama, "create_completion", None)
        if not callable(completion):
            raise EngineUnavailableError("llama-cpp-python has no loaded completion implementation")
        try:
            parameters = inspect.signature(completion).parameters
        except (TypeError, ValueError) as exc:
            raise EngineUnavailableError(
                "llama-cpp-python completion defaults cannot be inspected for v3 evidence"
            ) from exc
        defaults: dict[str, Any] = {}
        for option_name, parameter_name in (
            ("top_p", "top_p"),
            ("top_k", "top_k"),
            ("repetition_penalty", "repeat_penalty"),
        ):
            parameter = parameters.get(parameter_name)
            if parameter is None or parameter.default is inspect.Parameter.empty:
                raise EngineUnavailableError(
                    "llama-cpp-python does not expose a resolvable default for a v3 generation option",
                    details={"option": option_name, "binding_parameter": parameter_name},
                )
            defaults[option_name] = parameter.default
        if (
            not isinstance(defaults["top_p"], (int, float))
            or isinstance(defaults["top_p"], bool)
            or not 0 <= defaults["top_p"] <= 1
            or not isinstance(defaults["top_k"], int)
            or isinstance(defaults["top_k"], bool)
            or defaults["top_k"] < 1
            or not isinstance(defaults["repetition_penalty"], (int, float))
            or isinstance(defaults["repetition_penalty"], bool)
            or defaults["repetition_penalty"] <= 0
        ):
            raise EngineUnavailableError(
                "llama-cpp-python exposed invalid defaults for v3 generation settings",
                details={"defaults": defaults},
            )
        return defaults

    def resolve_generation_options_v3(self, request: GenerationRequestV3) -> SettingsEvidenceV3:
        options = request.generation_options
        if options.top_k == 0:
            raise UnsupportedGenerationSettingError(
                "llama.cpp top_k must be at least one for the supported sampler contract",
                details={"engine": self.name, "option": "top_k", "requested": 0, "minimum": 1},
            )
        if options.repetition_window is not None:
            raise UnsupportedGenerationSettingError(
                "llama.cpp does not support request-scoped repetition_window",
                details={
                    "engine": self.name,
                    "option": "repetition_window",
                    "requested": options.repetition_window,
                    "native_scope": "LOAD",
                },
            )
        resolved_intent, _ = self._thinking_parameters(
            options.thinking_intent,
            studio_resolved=request.studio_resolved_thinking,
        )
        defaults = self._native_generation_defaults()
        seed = options.seed if options.seed is not None else secrets.randbits(32)
        effective = GenerationOptions(
            max_tokens=options.max_tokens,
            temperature=options.temperature,
            top_p=options.top_p if options.top_p is not None else defaults["top_p"],
            top_k=options.top_k if options.top_k is not None else defaults["top_k"],
            repetition_penalty=(
                options.repetition_penalty
                if options.repetition_penalty is not None
                else defaults["repetition_penalty"]
            ),
            repetition_window=None,
            stop=options.stop,
            seed=seed,
            thinking_intent=resolved_intent,
        )
        requested_intent = options.thinking_intent.to_dict()
        resolved_intent_values = resolved_intent.to_dict()
        resolutions = [
            OptionResolutionV3(
                path=f"thinking_intent.{field}",
                requested=requested_intent[field],
                resolved=resolved_intent_values[field],
                effective=resolved_intent_values[field],
                status="resolved",
                reason=(
                    "Foundation applied the explicit Studio ON/OFF resolution for AUTO intent"
                    if field == "mode"
                    else "Foundation carried the explicit Studio-resolved Thinking value"
                ),
            )
            for field in ("mode", "effort", "budget_tokens")
            if requested_intent[field] != resolved_intent_values[field]
        ]
        requested_values = options.to_dict()
        effective_values = effective.to_dict()
        reasons = {
            "top_p": "null resolves to the default exposed by the loaded llama-cpp-python create_completion method",
            "top_k": "null resolves to the default exposed by the loaded llama-cpp-python create_completion method",
            "repetition_penalty": "null resolves to the default exposed by the loaded llama-cpp-python create_completion method",
            "seed": "null resolves to a Foundation-generated per-request seed so the effective sampler input is recorded",
        }
        for path in reasons:
            if requested_values[path] != effective_values[path]:
                resolutions.append(
                    OptionResolutionV3(
                        path=path,
                        requested=requested_values[path],
                        resolved=effective_values[path],
                        effective=effective_values[path],
                        status="resolved",
                        reason=reasons[path],
                    )
                )
        return SettingsEvidenceV3.from_options(
            scope="GENERATION",
            requested=options,
            resolved=effective,
            effective=effective,
            resolutions=tuple(resolutions),
        )

    def _template_prompt(
        self,
        llama: Any,
        messages: list[dict[str, str]],
        options: GenerationOptions,
        *,
        studio_resolved: ThinkingIntent | None = None,
        allow_unset_v1_thinking: bool = False,
    ) -> tuple[list[int], int, Any, dict[str, Any]]:
        with self._lock:
            formatter = self._chat_formatter
            template_text = self._chat_template_text
        if not callable(formatter):
            raise UnsupportedGenerationSettingError(
                "the loaded GGUF has no selected default chat template supported by the binding",
                details={"engine": self.name, "artifact_id": self._loaded.artifact_id if self._loaded else None},
            )
        intent = options.thinking_intent
        if allow_unset_v1_thinking and intent.mode == ThinkingMode.AUTO:
            template_kwargs: dict[str, Any] = {}
            resolved_intent = intent
        else:
            resolved_intent, template_kwargs = self._thinking_parameters(intent, studio_resolved=studio_resolved)
        try:
            formatted = formatter(messages=messages, **template_kwargs)
        except Exception as exc:  # noqa: BLE001 - template failures are runtime evidence
            raise EngineRuntimeError(
                "GGUF chat template rendering failed",
                details={
                    "engine": self.name,
                    "failure_kind": "chat_template_failure",
                    "native_exception": type(exc).__name__,
                },
            ) from exc
        prompt = getattr(formatted, "prompt", None)
        if not isinstance(prompt, str):
            raise EngineRuntimeError(
                "llama.cpp chat handler did not return a rendered prompt",
                details={"engine": self.name, "failure_kind": "chat_template_failure"},
            )
        tokenizer = getattr(llama, "tokenize", None)
        if not callable(tokenizer):
            raise EngineUnavailableError("llama-cpp-python did not expose its native tokenizer")
        added_special = bool(getattr(formatted, "added_special", True))
        try:
            prompt_token_ids = tokenizer(prompt.encode("utf-8"), add_bos=not added_special, special=True)
        except Exception as exc:  # noqa: BLE001 - tokenizer failures block exact budgets
            raise EngineRuntimeError(
                "llama.cpp could not count tokens for the selected chat template",
                details={"engine": self.name, "failure_kind": "tokenization_failure"},
            ) from exc
        if not isinstance(prompt_token_ids, (list, tuple)) or any(
            not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in prompt_token_ids
        ):
            raise EngineRuntimeError(
                "llama.cpp tokenizer returned an invalid token sequence for the selected chat template",
                details={"engine": self.name, "failure_kind": "tokenization_failure"},
            )
        prompt_tokens = len(prompt_token_ids)
        if prompt_tokens < 1:
            raise EngineRuntimeError(
                "llama.cpp chat template produced an empty tokenized prompt",
                details={"engine": self.name, "failure_kind": "chat_template_failure"},
            )
        return list(prompt_token_ids), prompt_tokens, formatted, {
            "resolved_thinking_intent": resolved_intent.to_dict(),
            "template_variables_passed": dict(template_kwargs),
            "chat_template_sha256": hashlib.sha256((template_text or "").encode("utf-8")).hexdigest(),
            "chat_format": "tokenizer.chat_template.default",
        }

    def count_prompt_tokens_v3(self, request: GenerationRequestV3) -> int | None:
        options = GenerationOptions.from_payload(self.resolve_generation_options_v3(request).effective)
        return self.prepare_generation_v3(request, options).prompt_tokens

    def prepare_generation_v3(
        self,
        request: GenerationRequestV3,
        effective_options: GenerationOptions,
    ) -> PreparedGenerationV3:
        llama, _, _ = self._loaded_state(self._legacy_v3_request(request))
        prompt_token_ids, prompt_tokens, formatted, template_evidence = self._template_prompt(
            llama,
            [dict(message) for message in request.messages],
            effective_options,
            studio_resolved=request.studio_resolved_thinking,
        )
        with self._lock:
            load_options = self._load_options
        return PreparedGenerationV3(
            prompt_tokens=prompt_tokens,
            adapter_state={
                "prompt_token_ids": prompt_token_ids,
                "formatted": formatted,
                "template_evidence": template_evidence,
                "model_context_size": load_options.model_context_size if load_options else None,
            },
        )

    @staticmethod
    def _legacy_v3_request(request: GenerationRequestV3) -> GenerationRequest:
        options = request.generation_options
        thinking_enabled = options.thinking_intent.to_legacy_enabled()
        return GenerationRequest(
            model_artifact_id=request.model_artifact_id,
            messages=[dict(message) for message in request.messages],
            request_id=request.request_id or "",
            consumer_id=request.consumer_id,
            lease_id=request.lease_id,
            max_tokens=options.max_tokens,
            temperature=options.temperature,
            top_p=options.top_p,
            thinking_enabled=thinking_enabled,
            timeout_ms=request.execution_constraints.timeout_ms,
            metadata=dict(request.metadata),
        )

    def _legacy_budget(self, request: GenerationRequest, prompt_tokens: int, options: LoadOptions) -> None:
        budget = request.runtime_options.context.context_length
        context_size = options.model_context_size
        requested_max = request.max_tokens
        if budget is not None and prompt_tokens + requested_max > budget:
            raise ContextLengthExceededError(
                "prompt tokens plus requested generation exceed context.context_length",
                details={
                    "context_length": budget,
                    "prompt_tokens": prompt_tokens,
                    "max_tokens": requested_max,
                    "enforcement": "preflight",
                },
            )
        if context_size is not None and prompt_tokens + requested_max > context_size:
            raise ContextLengthExceededError(
                "prompt tokens plus requested generation exceed the loaded llama.cpp context",
                details={
                    "model_context_size": context_size,
                    "prompt_tokens": prompt_tokens,
                    "max_tokens": requested_max,
                    "enforcement": "preflight",
                },
            )

    @staticmethod
    def _completion_arguments(options: GenerationOptions, formatted: Any) -> dict[str, Any]:
        if options.repetition_window is not None:
            raise UnsupportedGenerationSettingError(
                "llama.cpp does not support request-scoped repetition_window",
                details={"engine": "llama.cpp", "option": "repetition_window"},
            )
        stop = list(getattr(formatted, "stop", None) or [])
        stop.extend(item for item in options.stop if item not in stop)
        arguments: dict[str, Any] = {
            "max_tokens": options.max_tokens,
            "temperature": options.temperature,
            "stream": True,
        }
        if stop:
            arguments["stop"] = stop
        stopping_criteria = getattr(formatted, "stopping_criteria", None)
        if stopping_criteria is not None:
            arguments["stopping_criteria"] = stopping_criteria
        for field, name in (
            ("top_p", "top_p"),
            ("top_k", "top_k"),
            ("repetition_penalty", "repeat_penalty"),
            ("seed", "seed"),
        ):
            value = getattr(options, field)
            if value is not None:
                arguments[name] = value
        return arguments

    @staticmethod
    def _failure_kind(exc: Exception, *, phase: str) -> str:
        message = str(exc).lower()
        if any(value in message for value in ("out of memory", "oom", "allocation failed", "failed to allocate")):
            return "out_of_memory"
        if any(value in message for value in ("context", "n_ctx", "sequence length", "too many tokens")):
            return "context_overflow"
        if "template" in message or "jinja" in message:
            return "chat_template_failure"
        return "load_failure" if phase == "load" else "inference_failure"

    def _raise_runtime_error(self, exc: Exception, request: GenerationRequest, *, phase: str) -> RuntimeFoundationError:
        if isinstance(exc, RuntimeFoundationError):
            return exc
        failure_kind = self._failure_kind(exc, phase=phase)
        if failure_kind == "context_overflow":
            return ContextLengthExceededError(
                "llama.cpp rejected the request because the context was exceeded",
                details={"engine": self.name, "failure_kind": failure_kind, "request_id": request.request_id},
            )
        return EngineRuntimeError(
            "llama.cpp generation failed",
            details={
                "engine": self.name,
                "failure_kind": failure_kind,
                "request_id": request.request_id,
                "native_exception": type(exc).__name__,
            },
        )

    @staticmethod
    def _chunk_fields(chunk: Any) -> tuple[str, str | None]:
        if not isinstance(chunk, dict):
            return "", None
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return "", None
        text = choices[0].get("text", "")
        return (text if isinstance(text, str) else str(text or "")), choices[0].get("finish_reason")

    def _record_native_metal_generation(self) -> None:
        with self._lock:
            selected = self._selected_execution_acceleration
            if not isinstance(selected, dict) or selected.get("backend") != "metal":
                return
            observation = selected.get("native_metal_load_observation")
            if not isinstance(observation, dict) or observation.get("status") != "confirmed":
                return
            selected["actual_kernel_execution"] = "confirmed_by_native_metal_offload_and_generated_token"
            selected["generation_output_observed"] = True
            selected["generation_confirmation_source"] = (
                "llama.cpp native generation yielded output after MTL device, compute buffer, and GPU layer-offload observations"
            )

    def stream(self, request: GenerationRequest, cancel_event: threading.Event) -> Iterator[StreamEvent]:
        yield from self._stream_common(request, None, cancel_event)

    def stream_v3(
        self,
        request: GenerationRequestV3,
        effective_options: GenerationOptions,
        cancel_event: threading.Event,
        prepared: PreparedGenerationV3 | None = None,
    ) -> Iterator[StreamEvent]:
        yield from self._stream_common(request, effective_options, cancel_event, prepared=prepared)

    def _stream_common(
        self,
        request: GenerationRequest | GenerationRequestV3,
        supplied_options: GenerationOptions | None,
        cancel_event: threading.Event,
        *,
        prepared: PreparedGenerationV3 | None = None,
    ) -> Iterator[StreamEvent]:
        is_v3 = isinstance(request, GenerationRequestV3)
        request_id = request.request_id or new_id("req")
        timeout_ms = request.execution_constraints.timeout_ms if is_v3 else request.timeout_ms
        if is_v3:
            resolution = self.resolve_runtime_options(RuntimeOptions())
            generation_options = supplied_options or GenerationOptions.from_payload(
                self.resolve_generation_options_v3(request).effective
            )
            studio_resolved = request.studio_resolved_thinking
            is_v1_unset_thinking = False
            effective_request = self._legacy_v3_request(request)
            requested_runtime_settings = RuntimeOptions().to_dict()
        else:
            resolution = self.resolve_runtime_options(request.runtime_options)
            generation_options, studio_resolved = self._generation_options(request)
            is_v1_unset_thinking = (
                not isinstance(request, GenerationRequestV2) and request.thinking_enabled is None
            )
            effective_request = (
                GenerationRequest.from_payload(request.to_dict())
                if not isinstance(request, GenerationRequestV2)
                else request
            )
            requested_runtime_settings = request.runtime_options.to_dict()
        llama, _, load_options = self._loaded_state(request)
        prepared_state = prepared.adapter_state if is_v3 and prepared is not None else None
        if is_v3 and isinstance(prepared_state, dict):
            prompt_token_ids = prepared_state.get("prompt_token_ids")
            formatted = prepared_state.get("formatted")
            template_evidence = prepared_state.get("template_evidence")
            if not isinstance(prompt_token_ids, list) or not isinstance(template_evidence, dict):
                raise EngineRuntimeError(
                    "llama.cpp v3 execution requires its prepared prompt token IDs",
                    details={"engine": self.name, "failure_kind": "prepared_prompt_missing"},
                )
            prompt_tokens = len(prompt_token_ids)
            model_context_size = prepared_state.get("model_context_size")
        else:
            prompt_token_ids, prompt_tokens, formatted, template_evidence = self._template_prompt(
                llama,
                [dict(message) for message in request.messages],
                generation_options,
                studio_resolved=studio_resolved,
                allow_unset_v1_thinking=is_v1_unset_thinking,
            )
            model_context_size = load_options.model_context_size
        if is_v3:
            required_tokens = prompt_tokens + generation_options.max_tokens
            max_context_tokens = request.execution_constraints.max_context_tokens
            if max_context_tokens is not None and required_tokens > max_context_tokens:
                raise ContextLengthExceededError(
                    "prompt plus requested generation exceed max_context_tokens",
                    details={
                        "prompt_tokens": prompt_tokens,
                        "max_tokens": generation_options.max_tokens,
                        "required_tokens": required_tokens,
                        "max_context_tokens": max_context_tokens,
                        "generation_started": False,
                    },
                )
            if model_context_size is not None and required_tokens > model_context_size:
                raise ContextLengthExceededError(
                    "prompt plus requested generation exceed the loaded llama.cpp context",
                    details={
                        "prompt_tokens": prompt_tokens,
                        "max_tokens": generation_options.max_tokens,
                        "required_tokens": required_tokens,
                        "model_context_size": model_context_size,
                        "generation_started": False,
                    },
                )
        if not is_v3:
            self._legacy_budget(request, prompt_tokens, load_options)
        if not is_v3 and resolution.effective.context.context_length is not None:
            effective_request = GenerationRequest(
                model_artifact_id=effective_request.model_artifact_id,
                messages=effective_request.messages,
                request_id=effective_request.request_id,
                consumer_id=effective_request.consumer_id,
                lease_id=effective_request.lease_id,
                max_tokens=generation_options.max_tokens,
                temperature=generation_options.temperature,
                top_p=generation_options.top_p,
                thinking_enabled=generation_options.thinking_intent.to_legacy_enabled(),
                timeout_ms=effective_request.timeout_ms,
                runtime_options=resolution.effective,
                metadata=effective_request.metadata,
            )
        with self._lock:
            self._active[request_id] = cancel_event
        started = time.perf_counter()
        deadline = started + timeout_ms / 1000 if timeout_ms else None
        metrics = RuntimeMetrics(
            started_at=utc_now(),
            measurement_kind="llama.cpp",
            measurement_provenance="observed",
            context_length=load_options.model_context_size,
            prefill_tokens=prompt_tokens,
        )
        text_parts: list[str] = []
        generated_chunks = 0
        finish_reason = "stop"
        completion_iterator: Any = None
        try:
            yield StreamEvent(type="started", request_id=request_id, sequence=0)
            completion_call = getattr(llama, "create_completion", None)
            if not callable(completion_call):
                raise EngineUnavailableError("llama-cpp-python did not expose create_completion")
            arguments = self._completion_arguments(generation_options, formatted)
            completion_iterator = iter(completion_call(prompt=prompt_token_ids, **arguments))
            for sequence, chunk in enumerate(completion_iterator, start=1):
                if cancel_event.is_set():
                    metrics.cancellation = True
                    metrics.finish_reason = "cancelled"
                    metrics.finished_at = utc_now()
                    self._last_metrics = metrics.to_dict()
                    yield StreamEvent(
                        type="error",
                        request_id=request_id,
                        sequence=sequence,
                        done=True,
                        error=RequestCancelledError("generation was cancelled").as_dict(),
                    )
                    return
                if deadline is not None and time.monotonic() >= deadline:
                    metrics.timeout = True
                    metrics.finish_reason = "timeout"
                    metrics.finished_at = utc_now()
                    self._last_metrics = metrics.to_dict()
                    yield StreamEvent(
                        type="error",
                        request_id=request_id,
                        sequence=sequence,
                        done=True,
                        error=RuntimeTimeoutError(
                            "generation exceeded timeout_ms",
                            details={"timeout_semantics": "cooperative", "engine": self.name},
                        ).as_dict(),
                    )
                    return
                delta, chunk_finish_reason = self._chunk_fields(chunk)
                if chunk_finish_reason:
                    finish_reason = str(chunk_finish_reason)
                if delta:
                    generated_chunks += 1
                    self._record_native_metal_generation()
                    if not text_parts:
                        metrics.cold_ttft_ms = (time.perf_counter() - started) * 1000
                    text_parts.append(delta)
                    yield StreamEvent(type="delta", request_id=request_id, sequence=sequence, delta=delta)
            output = "".join(text_parts)
            try:
                completion_tokens = len(llama.tokenize(output.encode("utf-8"), add_bos=False, special=True))
            except Exception:  # noqa: BLE001 - output token count remains unresolved if native tokenizer errors
                completion_tokens = None
            metrics.finished_at = utc_now()
            metrics.completion_tokens = completion_tokens
            metrics.generation_duration_ms = (time.perf_counter() - started) * 1000
            if completion_tokens is not None and metrics.generation_duration_ms > 0:
                metrics.generation_tokens_per_second = completion_tokens / (metrics.generation_duration_ms / 1000)
            metrics.process_memory_bytes = process_memory_bytes()
            metrics.finish_reason = finish_reason
            metrics.extra = {
                "prompt_token_count_source": "selected GGUF chat template plus native tokenizer",
                "completion_token_count_source": "visible output re-tokenized by native tokenizer",
                "generated_stream_chunks": generated_chunks,
                "chat_template": template_evidence,
                "effective_generation_options": generation_options.to_dict(),
                "selected_execution_acceleration": dict(self._selected_execution_acceleration or {}),
                "engine_build_identity": self.build_identity().to_dict() if self.build_identity() else None,
            }
            self._last_metrics = metrics.to_dict()
            result = GenerationResult(
                request_id=request_id,
                model_artifact_id=request.model_artifact_id,
                engine=self.name,
                text=output,
                finish_reason=finish_reason,
                usage=TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens if completion_tokens is not None else None,
                ),
                metrics=metrics,
                requested_runtime_settings=requested_runtime_settings,
                effective_runtime_settings=resolution.effective.to_dict(),
            )
            yield StreamEvent(
                type="completed",
                request_id=request_id,
                sequence=generated_chunks + 1,
                done=True,
                result=result.to_dict(),
            )
        except RuntimeFoundationError:
            raise
        except Exception as exc:  # noqa: BLE001 - catch only errors returned by the binding
            raise self._raise_runtime_error(exc, effective_request, phase="generation") from exc
        finally:
            close = getattr(completion_iterator, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - generator close is best effort
                    pass
            with self._lock:
                self._active.pop(request_id, None)

    def generate(self, request: GenerationRequest, cancel_event: threading.Event) -> GenerationResult:
        completed: dict[str, Any] | None = None
        for event in self.stream(request, cancel_event):
            if event.type == "completed" and event.result is not None:
                completed = event.result
            elif event.type == "error":
                error = event.error or {}
                details = dict(error.get("details") or {})
                if error.get("code") == "cancelled":
                    raise RequestCancelledError(str(error.get("message") or "generation was cancelled"), details=details)
                if error.get("code") == "runtime_timeout":
                    raise RuntimeTimeoutError(str(error.get("message") or "generation timed out"), details=details)
                if error.get("code") == "context_length_exceeded":
                    raise ContextLengthExceededError(str(error.get("message") or "context budget exceeded"), details=details)
                raise EngineRuntimeError(str(error.get("message") or "llama.cpp generation failed"), details=details)
        if completed is None:
            raise EngineRuntimeError("llama.cpp generation ended without a result", details={"engine": self.name})
        metrics_payload = dict(completed.get("metrics") or {})
        metrics_payload.pop("contract_version", None)
        metrics = RuntimeMetrics(
            **{key: value for key, value in metrics_payload.items() if key in RuntimeMetrics.__dataclass_fields__}
        )
        usage_payload = dict(completed.get("usage") or {})
        usage_payload.pop("contract_version", None)
        usage = TokenUsage(**{key: value for key, value in usage_payload.items() if key in TokenUsage.__dataclass_fields__})
        return GenerationResult(
            request_id=completed["request_id"],
            model_artifact_id=completed["model_artifact_id"],
            engine=completed["engine"],
            text=completed["text"],
            finish_reason=completed["finish_reason"],
            usage=usage,
            metrics=metrics,
            execution_id=completed.get("execution_id"),
            requested_runtime_settings=completed.get("requested_runtime_settings"),
            effective_runtime_settings=completed.get("effective_runtime_settings"),
        )

    def generate_v3(
        self,
        request: GenerationRequestV3,
        effective_options: GenerationOptions,
        cancel_event: threading.Event,
        prepared: PreparedGenerationV3 | None = None,
    ) -> GenerationResult:
        completed: dict[str, Any] | None = None
        for event in self.stream_v3(request, effective_options, cancel_event, prepared=prepared):
            if event.type == "completed" and event.result is not None:
                completed = event.result
            elif event.type == "error":
                error = event.error or {}
                details = dict(error.get("details") or {})
                if error.get("code") == "cancelled":
                    raise RequestCancelledError(str(error.get("message") or "generation was cancelled"), details=details)
                if error.get("code") == "runtime_timeout":
                    raise RuntimeTimeoutError(str(error.get("message") or "generation timed out"), details=details)
                if error.get("code") == "context_length_exceeded":
                    raise ContextLengthExceededError(str(error.get("message") or "context budget exceeded"), details=details)
                raise EngineRuntimeError(str(error.get("message") or "llama.cpp generation failed"), details=details)
        if completed is None:
            raise EngineRuntimeError("llama.cpp generation ended without a result", details={"engine": self.name})
        metrics_payload = dict(completed.get("metrics") or {})
        metrics_payload.pop("contract_version", None)
        metrics = RuntimeMetrics(
            **{key: value for key, value in metrics_payload.items() if key in RuntimeMetrics.__dataclass_fields__}
        )
        usage_payload = dict(completed.get("usage") or {})
        usage_payload.pop("contract_version", None)
        usage = TokenUsage(**{key: value for key, value in usage_payload.items() if key in TokenUsage.__dataclass_fields__})
        return GenerationResult(
            request_id=completed["request_id"],
            model_artifact_id=completed["model_artifact_id"],
            engine=completed["engine"],
            text=completed["text"],
            finish_reason=completed["finish_reason"],
            usage=usage,
            metrics=metrics,
            execution_id=completed.get("execution_id"),
            requested_runtime_settings=completed.get("requested_runtime_settings"),
            effective_runtime_settings=completed.get("effective_runtime_settings"),
        )

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            event = self._active.get(request_id)
        if event is None:
            return False
        event.set()
        return True

    def health(self) -> dict[str, Any]:
        capability = self.discover_capability()
        with self._lock:
            loaded = self._loaded
            active = sorted(self._active)
            acceleration = dict(self._selected_execution_acceleration or {}) if loaded else None
        return {
            "status": "loaded" if loaded else "ready" if capability.available else "unavailable",
            "engine": self.name,
            "loaded_artifact_id": loaded.artifact_id if loaded else None,
            "active_request_ids": active,
            "process_model": "same-process",
            "selected_execution_acceleration": acceleration,
            "capability": capability.to_dict(),
        }

    def runtime_metrics(self) -> dict[str, Any]:
        with self._lock:
            return {
                "engine": self.name,
                "active_request_ids": sorted(self._active),
                "last_metrics": dict(self._last_metrics),
            }


def replace_artifact_identity(artifact: ArtifactBindingV2, observed: GGUFObservedMetadata) -> ArtifactBindingV2:
    """Bind the adapter's second validation to the existing Artifact v2 identity."""

    from dataclasses import replace

    return replace(
        artifact,
        content_identity=observed.content_identity,
        format="gguf",
        quantization=observed.quantization,
    )
