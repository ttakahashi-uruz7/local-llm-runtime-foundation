"""Lazy MLX / mlx-lm adapter.

This module is import-safe on Windows.  The actual MLX path is intentionally
runtime-gated and remains subject to real Apple Silicon validation.
"""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import inspect
import json
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Mapping

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
    utc_now,
)
from ..errors import (
    ArtifactCompatibilityError,
    ArtifactNotFoundError,
    ContextLengthExceededError,
    EngineRuntimeError,
    EngineUnavailableError,
    InvalidRuntimeOptionError,
    RequestCancelledError,
    RuntimeTimeoutError,
    UnsupportedGenerationSettingError,
    UnsupportedRuntimeOptionError,
    UnsupportedExecutionInputError,
)
from ..contracts_v2 import ADAPTER_LINEAGE_METADATA_KEY, AdapterLineageV1, BuildIdentityV1, ExecutionInputV1
from ..host import runtime_snapshot
from .base import EngineAdapter


class MLXAdapter(EngineAdapter):
    name = "mlx"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._mlx_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="foundation-mlx")
        self._mlx_thread_id: int | None = None
        self._loaded: ModelArtifactBinding | None = None
        self._loaded_execution_input: ExecutionInputV1 | None = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._stream_generate: Any = None
        self._active: dict[str, threading.Event] = {}
        self._last_metrics: dict[str, Any] = {}

    @staticmethod
    def _version(distribution: str) -> str | None:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return None

    def _modules(self) -> tuple[Any | None, Any | None, str | None]:
        system = platform.system().lower()
        architecture = platform.machine().lower()
        if system != "darwin" or architecture not in {"arm64", "aarch64"}:
            return None, None, "MLX requires Apple Silicon macOS"
        try:
            mlx = importlib.import_module("mlx.core")
            mlx_lm = importlib.import_module("mlx_lm")
            stream_generate = getattr(mlx_lm, "stream_generate", None)
            if not callable(stream_generate):
                return None, None, "installed mlx-lm does not expose stream_generate"
            device = str(mlx.default_device()).lower()
            if "gpu" not in device and "metal" not in device:
                return None, None, f"MLX default device is not Metal-backed: {device}"
            return mlx, mlx_lm, None
        except (ImportError, ModuleNotFoundError, AttributeError, RuntimeError, ValueError) as exc:
            return None, None, f"MLX import failed: {exc}"

    def _available(self) -> tuple[Any, Any]:
        mlx, mlx_lm, reason = self._modules()
        if mlx is None or mlx_lm is None:
            raise EngineUnavailableError("MLX/Metal adapter is unavailable on this host", details={"engine": self.name, "reason": reason})
        return mlx, mlx_lm

    def _run_on_mlx_thread(self, function: Any) -> Any:
        """Run MLX stateful operations on the one thread that owns the loaded model."""

        if threading.get_ident() == self._mlx_thread_id:
            return function()

        def invoke() -> Any:
            self._mlx_thread_id = threading.get_ident()
            return function()

        return self._mlx_executor.submit(invoke).result()

    def _iter_on_mlx_thread(self, iterator_factory: Any) -> Iterator[Any]:
        """Create, advance, and close an MLX generator on its owning thread."""

        sentinel = object()
        iterator = self._run_on_mlx_thread(lambda: iter(iterator_factory()))
        try:
            while True:
                item = self._run_on_mlx_thread(lambda: next(iterator, sentinel))
                if item is sentinel:
                    return
                yield item
        finally:
            close = getattr(iterator, "close", None)
            if callable(close):
                self._run_on_mlx_thread(close)

    @staticmethod
    def _signature(function: Any) -> Mapping[str, inspect.Parameter]:
        try:
            return inspect.signature(function).parameters
        except (TypeError, ValueError):
            return {}

    @staticmethod
    def _supports(parameters: Mapping[str, inspect.Parameter], name: str) -> bool:
        return name in parameters or any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())

    def identity(self) -> EngineIdentity:
        mlx, mlx_lm, _ = self._modules()
        if mlx is None or mlx_lm is None:
            return EngineIdentity(engine=self.name)
        return EngineIdentity(
            engine=self.name,
            version=self._version("mlx-lm") or getattr(mlx_lm, "__version__", None),
            build=self._version("mlx") or getattr(mlx, "__version__", None),
        )

    def build_identity(self) -> BuildIdentityV1 | None:
        """Build identity from observed mlx-lm and mlx distribution versions."""

        mlx, mlx_lm, _ = self._modules()
        if mlx is None or mlx_lm is None:
            return None
        mlx_lm_version = self._version("mlx-lm") or getattr(mlx_lm, "__version__", None)
        mlx_version = self._version("mlx") or getattr(mlx, "__version__", None)
        if not isinstance(mlx_lm_version, str) or not mlx_lm_version.strip():
            return None
        if not isinstance(mlx_version, str) or not mlx_version.strip():
            return None
        return BuildIdentityV1.from_components(
            kind="python-distribution-set-v1",
            components={
                "mlx-lm": {"version": mlx_lm_version.strip()},
                "mlx": {"version": mlx_version.strip()},
            },
        )

    def discover_capability(self) -> EngineCapability:
        mlx, mlx_lm, reason = self._modules()
        if mlx is None or mlx_lm is None:
            return EngineCapability(
                identity=self.identity(),
                available=False,
                streaming=False,
                cancellation=False,
                load_unload=False,
                chat_template="unavailable",
                thinking_flag="unavailable",
                artifact_formats=["mlx", "safetensors"],
                runtime_options={path: {"status": "unavailable", "reason": reason} for path in RUNTIME_OPTION_PATHS},
                generation_options={
                    name: {"status": "unavailable", "reason": reason}
                    for name in ("max_tokens", "temperature", "top_p", "thinking_enabled")
                },
                reason=reason,
                build_identity=self.build_identity(),
            )
        stream_generate = getattr(mlx_lm, "stream_generate")
        parameters = self._signature(stream_generate)

        def supported() -> dict[str, Any]:
            return {"status": "supported", "mode": "observed", "signature_checked": True}

        def unavailable(message: str) -> dict[str, Any]:
            return {"status": "unavailable", "reason": message}

        runtime_options = {
            "context.context_length": {
                "status": "supported",
                "mode": "foundation-preflight-budget",
                "enforcement": "prompt_tokens_plus_max_tokens",
            },
            "context.sliding_window": unavailable("mlx-lm does not expose a sliding-window option"),
            "kv_cache.mode": {"status": "supported", "supported_values": ["auto", "full_precision", "quantized"]},
            "kv_cache.precision": supported(),
            "kv_cache.bits": supported() if self._supports(parameters, "kv_bits") else unavailable("no kv_bits parameter"),
            "kv_cache.group_size": supported() if self._supports(parameters, "kv_group_size") else unavailable("no kv_group_size parameter"),
            "kv_cache.quantization_start": (
                supported() if self._supports(parameters, "quantized_kv_start") else unavailable("no quantized_kv_start parameter")
            ),
            "kv_cache.max_size_tokens": supported() if self._supports(parameters, "max_kv_size") else unavailable("no max_kv_size parameter"),
            "kv_cache.cache_limit_bytes": unavailable("mlx-lm exposes no byte cache limit"),
            "prefill.chunk_size": supported() if self._supports(parameters, "prefill_step_size") else unavailable("no prefill_step_size parameter"),
            "prefill.batch_size": {"status": "supported", "supported_values": [1]},
            "prompt_cache.enabled": unavailable("prompt-cache object lineage is not implemented in Foundation v1"),
            "prompt_cache.max_entries": unavailable("mlx-lm exposes no prompt-cache entry limit"),
            "prompt_cache.max_size_tokens": unavailable("mlx-lm exposes no prompt-cache token limit"),
            "acceleration.backend": {"status": "supported", "supported_values": ["auto", "metal"]},
            "acceleration.device": unavailable("mlx-lm selects the Metal device internally"),
            "acceleration.threads": unavailable("MLX uses Metal device scheduling"),
            "engine_options": {"status": "unsupported", "reason": "no engine-specific options are registered"},
        }
        generation_options = {
            "max_tokens": {"status": "supported"},
            "temperature": {"status": "supported", "mode": "sampler"},
            "top_p": {"status": "supported", "mode": "sampler"},
            "thinking_enabled": {"status": "supported", "mode": "tokenizer-chat-template"},
        }
        return EngineCapability(
            identity=self.identity(),
            available=True,
            streaming=True,
            cancellation=True,
            load_unload=True,
            chat_template="tokenizer.apply_chat_template",
            thinking_flag="tokenizer chat-template keyword pass-through",
            artifact_formats=["mlx", "safetensors"],
            runtime_options=runtime_options,
            generation_options=generation_options,
            reason="MLX/Metal capability observed at runtime; hardware validation remains pending",
            build_identity=self.build_identity(),
        )

    def health(self) -> dict[str, Any]:
        capability = self.discover_capability()
        with self._lock:
            loaded = self._loaded
            active = list(self._active)
        return {
            "status": "loaded" if loaded and capability.available else "ready" if capability.available else "unavailable",
            "engine": self.name,
            "loaded_artifact_id": loaded.artifact_id if loaded else None,
            "active_request_ids": active,
            "capability": capability.to_dict(),
        }

    def load(self, artifact: ModelArtifactBinding) -> dict[str, Any]:
        _, mlx_lm = self._available()
        path = Path(artifact.local_path)
        if not path.exists():
            raise ArtifactNotFoundError(
                "model artifact path does not exist",
                details={"artifact_id": artifact.artifact_id, "local_path": artifact.local_path},
            )
        loader = getattr(mlx_lm, "load", None)
        if not callable(loader):
            raise EngineUnavailableError("installed mlx-lm does not expose load", details={"engine": self.name})
        parameters = self._signature(loader)
        started = time.perf_counter()
        try:
            if "path_or_hf_repo" in parameters:
                model, tokenizer = self._run_on_mlx_thread(lambda: loader(path_or_hf_repo=str(path)))
            else:
                model, tokenizer = self._run_on_mlx_thread(lambda: loader(str(path)))
        except Exception as exc:
            raise EngineRuntimeError(
                "MLX model load failed",
                details={"engine": self.name, "artifact_id": artifact.artifact_id, "failure_kind": "load_failure"},
            ) from exc
        with self._lock:
            self._model = model
            self._tokenizer = tokenizer
            self._stream_generate = getattr(mlx_lm, "stream_generate")
            self._loaded = artifact
            self._loaded_execution_input = None
        return {
            "loaded": True,
            "artifact_id": artifact.artifact_id,
            "load_duration_ms": (time.perf_counter() - started) * 1000,
            "measurement_provenance": "observed",
        }

    @staticmethod
    def _read_json(path: Path, *, role: str) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactCompatibilityError(
                f"{role} metadata could not be read as JSON",
                details={"path": str(path), "reason": str(exc)},
            ) from exc
        if not isinstance(value, dict):
            raise ArtifactCompatibilityError(f"{role} metadata must be a JSON object", details={"path": str(path)})
        return value

    @staticmethod
    def _safetensors_shapes(files: list[Path], *, role: str) -> dict[str, list[int]]:
        try:
            from safetensors import safe_open
        except (ImportError, ModuleNotFoundError) as exc:
            raise EngineUnavailableError(
                "safetensors metadata reader is unavailable; compatibility cannot be certified",
                details={"engine": "mlx", "role": role},
            ) from exc
        shapes: dict[str, list[int]] = {}
        for path in files:
            try:
                with safe_open(str(path), framework="np", device="cpu") as tensors:
                    for key in tensors.keys():
                        shapes[key] = list(tensors.get_slice(key).get_shape())
            except Exception as exc:  # noqa: BLE001 - normalize parser errors at the adapter boundary
                raise ArtifactCompatibilityError(
                    f"{role} safetensors header could not be inspected",
                    details={"path": str(path), "reason": str(exc)},
                ) from exc
        return shapes

    @staticmethod
    def _target_key(key: str, module: str, suffix: str) -> bool:
        return key.endswith(f".{module}.{suffix}") or key == f"{module}.{suffix}"

    @classmethod
    def _base_module_shapes(
        cls, base_path: Path, modules: tuple[str, ...], expected_quantization: str | None = None
    ) -> dict[str, dict[str, int]]:
        config_path = base_path / "config.json"
        index_path = base_path / "model.safetensors.index.json"
        if not config_path.is_file():
            raise ArtifactCompatibilityError("Base model is missing config.json", details={"path": str(config_path)})
        config = cls._read_json(config_path, role="Base config")
        if index_path.is_file():
            index = cls._read_json(index_path, role="Base safetensors index")
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ArtifactCompatibilityError("Base safetensors index has no weight_map")
            matched_keys = {
                key: value
                for key, value in weight_map.items()
                if isinstance(key, str)
                and any(cls._target_key(key, module, "scales") or cls._target_key(key, module, "weight") for module in modules)
                and isinstance(value, str)
            }
            files = sorted({base_path / filename for filename in matched_keys.values()})
            shapes = cls._safetensors_shapes(files, role="Base")
            matched_keys = {key: value for key, value in matched_keys.items() if key in shapes}
        else:
            files = sorted(base_path.glob("*.safetensors"))
            if not files:
                raise ArtifactCompatibilityError("Base has no indexed or single-file safetensors weights")
            shapes = cls._safetensors_shapes(files, role="Base")
            matched_keys = {
                key: "" for key in shapes
                if any(cls._target_key(key, module, "scales") or cls._target_key(key, module, "weight") for module in modules)
            }
        quantization = config.get("quantization_config") or config.get("quantization") or {}
        group_size = quantization.get("group_size") if isinstance(quantization, dict) else None
        if isinstance(quantization, dict) and quantization:
            bits = quantization.get("bits")
            mode = quantization.get("mode")
            if (
                isinstance(bits, bool)
                or not isinstance(bits, int)
                or bits <= 0
                or isinstance(group_size, bool)
                or not isinstance(group_size, int)
                or group_size <= 0
                or not isinstance(mode, str)
                or not mode.strip()
            ):
                raise ArtifactCompatibilityError(
                    "Base quantization_config must declare positive bits/group_size and a mode",
                    details={"quantization_config": quantization},
                )
            actual_quantization = f"mlx:bits={bits};group_size={group_size};mode={mode.strip().lower()}"
        else:
            actual_quantization = "mlx:unquantized"
        if expected_quantization is not None and expected_quantization != actual_quantization:
            raise ArtifactCompatibilityError(
                "Base config quantization does not match the declared direct-execution binding",
                details={"expected_quantization": expected_quantization, "actual_quantization": actual_quantization},
            )
        result: dict[str, dict[str, int]] = {}
        for module in modules:
            scale_keys = [key for key in matched_keys if cls._target_key(key, module, "scales")]
            weight_keys = [key for key in matched_keys if cls._target_key(key, module, "weight")]
            if scale_keys:
                if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0:
                    raise ArtifactCompatibilityError(
                        "quantized Base does not expose a valid quantization group_size",
                        details={"module": module, "group_size": group_size},
                    )
                dims = {tuple(shapes[key]) for key in scale_keys}
                if len(dims) != 1:
                    raise ArtifactCompatibilityError("Base target module has inconsistent quantized shapes", details={"module": module})
                output_features, input_groups = next(iter(dims))
                result[module] = {"in_features": input_groups * group_size, "out_features": output_features}
            elif weight_keys:
                dims = {tuple(shapes[key]) for key in weight_keys}
                if len(dims) != 1:
                    raise ArtifactCompatibilityError("Base target module has inconsistent weight shapes", details={"module": module})
                output_features, input_features = next(iter(dims))
                result[module] = {"in_features": input_features, "out_features": output_features}
            else:
                raise ArtifactCompatibilityError(
                    "Base weights do not contain an adapter target module",
                    details={"module": module},
                )
        return result

    def validate_execution_input(self, execution_input: ExecutionInputV1) -> None:
        self._available()
        if len(execution_input.adapters) > 1:
            raise UnsupportedExecutionInputError(
                "MLX direct execution supports at most one adapter",
                details={"adapter_count": len(execution_input.adapters), "maximum_supported": 1},
            )
        base_path = Path(execution_input.base.local_path or "")
        if not base_path.exists():
            raise ArtifactNotFoundError(
                "Base artifact path does not exist",
                details={"role": "base", "artifact_id": execution_input.base.artifact_id, "local_path": str(base_path)},
            )
        if not execution_input.adapters:
            return
        adapter = execution_input.adapters[0]
        if adapter.format not in {"mlx-lora", "mlx_lora", "lora"}:
            raise ArtifactCompatibilityError(
                "MLX direct execution requires an MLX LoRA adapter format",
                details={"artifact_id": adapter.artifact_id, "format": adapter.format},
            )
        adapter_path = Path(adapter.local_path or "")
        if not adapter_path.is_dir():
            raise ArtifactNotFoundError(
                "Adapter artifact path does not exist or is not a directory",
                details={"role": "adapter", "artifact_id": adapter.artifact_id, "local_path": str(adapter_path)},
            )
        config_path = adapter_path / "adapter_config.json"
        tensors_path = adapter_path / "adapters.safetensors"
        if not config_path.is_file() or not tensors_path.is_file():
            raise ArtifactNotFoundError(
                "MLX LoRA adapter requires adapter_config.json and adapters.safetensors",
                details={"artifact_id": adapter.artifact_id, "missing": [str(path) for path in (config_path, tensors_path) if not path.is_file()]},
            )
        config = self._read_json(config_path, role="Adapter config")
        if config.get("fine_tune_type") != "lora":
            raise ArtifactCompatibilityError(
                "Adapter config is not a LoRA adapter",
                details={"artifact_id": adapter.artifact_id, "fine_tune_type": config.get("fine_tune_type")},
            )
        lineage = AdapterLineageV1.from_payload(adapter.metadata[ADAPTER_LINEAGE_METADATA_KEY])
        parameters = config.get("lora_parameters")
        if not isinstance(parameters, dict):
            raise ArtifactCompatibilityError("Adapter config is missing lora_parameters")
        rank = parameters.get("rank")
        configured_modules = parameters.get("keys")
        if rank != lineage.rank:
            raise ArtifactCompatibilityError(
                "Adapter config rank does not match the supplied lineage",
                details={"config_rank": rank, "lineage_rank": lineage.rank},
            )
        if not isinstance(configured_modules, list) or set(configured_modules) != set(lineage.target_modules):
            raise ArtifactCompatibilityError(
                "Adapter config target modules do not match the supplied lineage",
                details={"config_target_modules": configured_modules, "lineage_target_modules": list(lineage.target_modules)},
            )
        adapter_shapes = self._safetensors_shapes([tensors_path], role="Adapter")
        for module in lineage.target_modules:
            for factor in ("lora_a", "lora_b"):
                actual_keys = [key for key in adapter_shapes if self._target_key(key, module, factor)]
                if not actual_keys:
                    raise ArtifactCompatibilityError(
                        "Adapter safetensors are missing a declared target factor",
                        details={"module": module, "factor": factor},
                    )
                actual_shapes = {tuple(adapter_shapes[key]) for key in actual_keys}
                expected_shape = tuple(lineage.tensor_shapes[module][factor])
                if actual_shapes != {expected_shape}:
                    raise ArtifactCompatibilityError(
                        "Adapter safetensors tensor shape does not match its declared lineage",
                        details={"module": module, "factor": factor, "expected_shape": list(expected_shape), "actual_shapes": [list(item) for item in sorted(actual_shapes)]},
                    )
        actual_base_shapes = self._base_module_shapes(
            base_path, lineage.target_modules, expected_quantization=execution_input.base.quantization
        )
        for module in lineage.target_modules:
            if actual_base_shapes[module] != execution_input.base.metadata["target_module_shapes"][module]:
                raise ArtifactCompatibilityError(
                    "Base safetensors shape does not match its declared compatibility manifest",
                    details={
                        "module": module,
                        "declared_shape": execution_input.base.metadata["target_module_shapes"][module],
                        "actual_shape": actual_base_shapes[module],
                    },
                )

    def load_execution_input(self, execution_input: ExecutionInputV1) -> dict[str, Any]:
        self.validate_execution_input(execution_input)
        if not execution_input.adapters:
            return self.load(execution_input.base.to_legacy())
        _, mlx_lm = self._available()
        loader = getattr(mlx_lm, "load", None)
        if not callable(loader):
            raise EngineUnavailableError("installed mlx-lm does not expose load", details={"engine": self.name})
        parameters = self._signature(loader)
        if not self._supports(parameters, "adapter_path"):
            raise EngineUnavailableError(
                "installed mlx-lm load does not expose adapter_path",
                details={"engine": self.name, "mlx_lm_version": self._version("mlx-lm")},
            )
        base_path = execution_input.base.local_path
        adapter_path = execution_input.adapters[0].local_path
        if base_path is None or adapter_path is None:
            raise ArtifactNotFoundError("Base and Adapter require filesystem locators for MLX direct load")
        kwargs: dict[str, Any] = {"adapter_path": adapter_path}
        if "path_or_hf_repo" in parameters:
            kwargs["path_or_hf_repo"] = base_path
            args: tuple[Any, ...] = ()
        else:
            args = (base_path,)
        started = time.perf_counter()
        try:
            model, tokenizer = self._run_on_mlx_thread(lambda: loader(*args, **kwargs))
        except Exception as exc:
            raise EngineRuntimeError(
                "MLX Base plus LoRA direct load failed",
                details={
                    "engine": self.name,
                    "base_artifact_id": execution_input.base.artifact_id,
                    "adapter_artifact_id": execution_input.adapters[0].artifact_id,
                    "failure_kind": "direct_load_failure",
                },
            ) from exc
        binding = execution_input.base.to_legacy()
        with self._lock:
            self._model = model
            self._tokenizer = tokenizer
            self._stream_generate = getattr(mlx_lm, "stream_generate")
            self._loaded = binding
            self._loaded_execution_input = execution_input
        return {
            "loaded": True,
            "artifact_id": binding.artifact_id,
            "execution_input_kind": execution_input.kind,
            "execution_input_fingerprint": execution_input.fingerprint,
            "adapter_count": len(execution_input.adapters),
            "load_mode": "direct_base_plus_adapter",
            "load_duration_ms": (time.perf_counter() - started) * 1000,
            "measurement_provenance": "observed",
        }

    def resolve_runtime_options(self, options: RuntimeOptions) -> RuntimeSettingsResolution:
        self._available()
        capability = self.discover_capability()
        requested = RuntimeOptions.from_payload(options.to_dict())
        effective = RuntimeOptions.from_payload(requested.to_dict())
        warnings: list[str] = []

        def explicit(path: str, value: Any) -> None:
            if value is None or value == {} or (path == "prompt_cache.enabled" and value is False):
                return
            info = capability.runtime_options.get(path, {})
            if info.get("status") != "supported":
                raise UnsupportedRuntimeOptionError(
                    f"MLX does not support {path}",
                    details={"path": path, "value": value, "reason": info.get("reason")},
                )
            supported_values = info.get("supported_values")
            if isinstance(supported_values, list) and value not in supported_values:
                raise UnsupportedRuntimeOptionError(
                    f"MLX does not support this value for {path}",
                    details={"path": path, "value": value, "supported_values": supported_values},
                )

        explicit("context.sliding_window", effective.context.sliding_window)
        explicit("kv_cache.cache_limit_bytes", effective.kv_cache.cache_limit_bytes)
        explicit("prefill.chunk_size", effective.prefill.chunk_size)
        explicit("prompt_cache.enabled", effective.prompt_cache.enabled)
        explicit("prompt_cache.max_entries", effective.prompt_cache.max_entries)
        explicit("prompt_cache.max_size_tokens", effective.prompt_cache.max_size_tokens)
        explicit("acceleration.device", effective.acceleration.device)
        explicit("acceleration.threads", effective.acceleration.threads)
        explicit("engine_options", effective.engine_options)

        if effective.acceleration.backend not in {"auto", "metal"}:
            raise UnsupportedRuntimeOptionError(
                "MLX supports only auto or metal acceleration",
                details={"path": "acceleration.backend", "value": effective.acceleration.backend},
            )
        if effective.acceleration.backend == "auto":
            effective = replace(effective, acceleration=replace(effective.acceleration, backend="metal"))
            warnings.append("MLX resolved acceleration.backend auto to metal")
        if effective.context.context_length is None:
            effective = replace(effective, context=replace(effective.context, context_length=4096))
            warnings.append("MLX defaulted context.context_length to 4096")
        if effective.prefill.batch_size not in {None, 1}:
            raise UnsupportedRuntimeOptionError(
                "MLX prefill batch size is fixed at 1",
                details={"path": "prefill.batch_size", "value": effective.prefill.batch_size},
            )
        if effective.prefill.batch_size is None:
            effective = replace(effective, prefill=replace(effective.prefill, batch_size=1))
        if effective.prefill.chunk_size is None:
            explicit("prefill.chunk_size", 2048)
            effective = replace(effective, prefill=replace(effective.prefill, chunk_size=2048))
            warnings.append("MLX defaulted prefill.chunk_size to 2048")

        kv = effective.kv_cache
        quant_fields = (kv.bits, kv.group_size, kv.quantization_start)
        if kv.mode == "auto" and any(value is not None for value in quant_fields):
            raise InvalidRuntimeOptionError(
                "KV quantization fields require an explicit kv_cache.mode", details={"path": "kv_cache.mode"}
            )
        if kv.mode == "full_precision" and any(value is not None for value in quant_fields):
            raise InvalidRuntimeOptionError(
                "KV quantization fields require kv_cache.mode=quantized", details={"path": "kv_cache"}
            )
        if kv.mode == "auto":
            effective = replace(effective, kv_cache=replace(kv, mode="full_precision", precision=kv.precision or "fp16"))
            warnings.append("MLX resolved kv_cache.mode auto to full_precision")
        elif kv.mode == "quantized":
            for path, value in (("kv_cache.bits", kv.bits), ("kv_cache.group_size", kv.group_size), ("kv_cache.quantization_start", kv.quantization_start)):
                explicit(path, value)
            effective = replace(
                effective,
                kv_cache=replace(
                    kv,
                    precision=kv.precision or "int8",
                    bits=kv.bits or 8,
                    group_size=kv.group_size or 64,
                    quantization_start=kv.quantization_start or 0,
                ),
            )
        if effective.kv_cache.max_size_tokens is not None:
            explicit("kv_cache.max_size_tokens", effective.kv_cache.max_size_tokens)
        return RuntimeSettingsResolution(
            requested=requested,
            effective=effective,
            option_status={
                path: str(capability.runtime_options.get(path, {}).get("status", "unsupported"))
                for path in RUNTIME_OPTION_PATHS
            },
            warnings=warnings,
        )

    def _ensure_loaded(self, request: GenerationRequest) -> tuple[Any, Any, Any]:
        with self._lock:
            loaded, model, tokenizer, stream_generate = self._loaded, self._model, self._tokenizer, self._stream_generate
        if not loaded or loaded.artifact_id != request.model_artifact_id or model is None or tokenizer is None or stream_generate is None:
            raise EngineRuntimeError(
                "requested artifact is not loaded in MLX",
                details={"artifact_id": request.model_artifact_id},
            )
        return model, tokenizer, stream_generate

    @staticmethod
    def _item_value(item: Any, name: str, default: Any = None) -> Any:
        if isinstance(item, dict):
            return item.get(name, default)
        return getattr(item, name, default)

    @staticmethod
    def _prompt(request: GenerationRequest, tokenizer: Any) -> str:
        messages = [{"role": str(message["role"]), "content": str(message["content"])} for message in request.messages]
        template = getattr(tokenizer, "apply_chat_template", None)
        if not callable(template):
            if request.thinking_enabled is not None:
                raise UnsupportedGenerationSettingError(
                    "the tokenizer does not expose a chat template for thinking_enabled",
                    details={"setting": "thinking_enabled"},
                )
            return "\n".join(f"{message['role']}: {message['content']}" for message in messages) + "\nassistant:"
        kwargs: dict[str, Any] = {"add_generation_prompt": True}
        if request.thinking_enabled is not None:
            kwargs["enable_thinking"] = request.thinking_enabled
        try:
            rendered = template(messages, **kwargs)
        except TypeError as exc:
            if request.thinking_enabled is not None:
                raise UnsupportedGenerationSettingError(
                    "the tokenizer chat template does not support thinking_enabled",
                    details={"setting": "thinking_enabled"},
                ) from exc
            rendered = template(messages, add_generation_prompt=True)
        if isinstance(rendered, str):
            return rendered
        if isinstance(rendered, list) and callable(getattr(tokenizer, "decode", None)):
            return tokenizer.decode(rendered)
        raise EngineRuntimeError("MLX tokenizer returned an unsupported chat-template value")

    @staticmethod
    def _enforce_context_budget(request: GenerationRequest, prompt_tokens: int) -> None:
        context_length = request.runtime_options.context.context_length
        if context_length is None:
            return
        requested_total = prompt_tokens + request.max_tokens
        if requested_total > context_length:
            raise ContextLengthExceededError(
                "prompt tokens plus generation budget exceed context.context_length",
                details={
                    "context_length": context_length,
                    "prompt_tokens": prompt_tokens,
                    "max_tokens": request.max_tokens,
                    "requested_total_tokens": requested_total,
                    "enforcement": "preflight",
                },
            )

    @staticmethod
    def _timeout_error(request: GenerationRequest) -> RuntimeTimeoutError:
        return RuntimeTimeoutError(
            "generation exceeded timeout_ms",
            details={"timeout_ms": request.timeout_ms, "timeout_semantics": "cooperative"},
        )

    def _stream_kwargs(self, request: GenerationRequest, options: RuntimeOptions, mlx_lm: Any) -> dict[str, Any]:
        stream_generate = getattr(mlx_lm, "stream_generate")
        parameters = self._signature(stream_generate)
        kwargs: dict[str, Any] = {"max_tokens": request.max_tokens}
        if request.temperature != 0.0 or request.top_p not in {None, 1.0}:
            try:
                sample_utils = importlib.import_module("mlx_lm.sample_utils")
                make_sampler = getattr(sample_utils, "make_sampler")
                sampler_parameters = self._signature(make_sampler)
                sampler_kwargs: dict[str, Any] = {}
                if "top_p" in sampler_parameters:
                    sampler_kwargs["top_p"] = request.top_p if request.top_p is not None else 1.0
                kwargs["sampler"] = make_sampler(request.temperature, **sampler_kwargs)
            except (ImportError, AttributeError, TypeError, ValueError) as exc:
                raise UnsupportedGenerationSettingError(
                    "installed mlx-lm could not construct the requested sampler",
                    details={"settings": {"temperature": request.temperature, "top_p": request.top_p}},
                ) from exc
        if options.prefill.chunk_size is not None and self._supports(parameters, "prefill_step_size"):
            kwargs["prefill_step_size"] = options.prefill.chunk_size
        if options.kv_cache.mode == "quantized":
            for key, value in (
                ("kv_bits", options.kv_cache.bits),
                ("kv_group_size", options.kv_cache.group_size),
                ("quantized_kv_start", options.kv_cache.quantization_start),
            ):
                if value is not None and self._supports(parameters, key):
                    kwargs[key] = value
        if options.kv_cache.max_size_tokens is not None and self._supports(parameters, "max_kv_size"):
            kwargs["max_kv_size"] = options.kv_cache.max_size_tokens
        return kwargs

    @staticmethod
    def _result_from_payload(payload: dict[str, Any]) -> GenerationResult:
        metrics_payload = dict(payload["metrics"])
        metrics_payload.pop("contract_version", None)
        metrics = RuntimeMetrics(**{key: value for key, value in metrics_payload.items() if key in RuntimeMetrics.__dataclass_fields__})
        usage_payload = dict(payload["usage"])
        usage_payload.pop("contract_version", None)
        usage = TokenUsage(**{key: value for key, value in usage_payload.items() if key in TokenUsage.__dataclass_fields__})
        return GenerationResult(
            request_id=payload["request_id"],
            model_artifact_id=payload["model_artifact_id"],
            engine=payload["engine"],
            text=payload["text"],
            finish_reason=payload["finish_reason"],
            usage=usage,
            metrics=metrics,
            requested_runtime_settings=payload.get("requested_runtime_settings"),
            effective_runtime_settings=payload.get("effective_runtime_settings"),
        )

    def stream(self, request: GenerationRequest, cancel_event: threading.Event) -> Iterator[StreamEvent]:
        model, tokenizer, stream_generate = self._ensure_loaded(request)
        _, mlx_lm = self._available()
        with self._lock:
            self._active[request.request_id] = cancel_event
        try:
            started_monotonic = time.perf_counter()
            metrics = RuntimeMetrics(
                started_at=utc_now(),
                measurement_kind="mlx",
                measurement_provenance="observed",
                context_length=request.runtime_options.context.context_length,
            )
            before = runtime_snapshot()
            prompt = self._prompt(request, tokenizer)
            prompt_tokens = max(1, len(getattr(tokenizer, "encode", lambda value: value.split())(prompt)))
            self._enforce_context_budget(request, prompt_tokens)
        except Exception:
            # Preflight failures happen before the generation try/finally below.
            # Do not leave a request behind that can make cancel/unload appear busy.
            with self._lock:
                self._active.pop(request.request_id, None)
            raise
        timeout_deadline = (
            time.monotonic() + request.timeout_ms / 1000 if request.timeout_ms is not None else None
        )
        text_parts: list[str] = []
        last_item: Any = None
        try:
            yield StreamEvent(type="started", request_id=request.request_id, sequence=0)
            iterator_factory = lambda: stream_generate(
                model, tokenizer, prompt, **self._stream_kwargs(request, request.runtime_options, mlx_lm)
            )
            for sequence, item in enumerate(self._iter_on_mlx_thread(iterator_factory), start=1):
                if timeout_deadline is not None and time.monotonic() >= timeout_deadline:
                    metrics.timeout = True
                    metrics.finished_at = utc_now()
                    metrics.finish_reason = "timeout"
                    self._last_metrics = metrics.to_dict()
                    yield StreamEvent(
                        type="error",
                        request_id=request.request_id,
                        sequence=sequence,
                        done=True,
                        error=self._timeout_error(request).as_dict(),
                    )
                    return
                if cancel_event.is_set():
                    metrics.cancellation = True
                    metrics.finished_at = utc_now()
                    metrics.finish_reason = "cancelled"
                    self._last_metrics = metrics.to_dict()
                    yield StreamEvent(
                        type="error",
                        request_id=request.request_id,
                        sequence=sequence,
                        done=True,
                        error=RequestCancelledError("generation was cancelled").as_dict(),
                    )
                    return
                last_item = item
                delta = self._item_value(item, "text", self._item_value(item, "response", ""))
                if not isinstance(delta, str):
                    delta = str(delta or "")
                if delta:
                    if not text_parts:
                        metrics.cold_ttft_ms = (time.perf_counter() - started_monotonic) * 1000
                    text_parts.append(delta)
                    yield StreamEvent(type="delta", request_id=request.request_id, sequence=sequence, delta=delta)
            if timeout_deadline is not None and time.monotonic() >= timeout_deadline:
                metrics.timeout = True
                metrics.finished_at = utc_now()
                metrics.finish_reason = "timeout"
                self._last_metrics = metrics.to_dict()
                yield StreamEvent(
                    type="error",
                    request_id=request.request_id,
                    sequence=len(text_parts) + 1,
                    done=True,
                    error=self._timeout_error(request).as_dict(),
                )
                return
            after = runtime_snapshot()
            metrics.finished_at = utc_now()
            metrics.prefill_tokens = self._item_value(last_item, "prompt_tokens", prompt_tokens) or prompt_tokens
            prompt_tps = self._item_value(last_item, "prompt_tps")
            generation_tokens = self._item_value(last_item, "generation_tokens")
            generation_tps = self._item_value(last_item, "generation_tps")
            peak_memory = self._item_value(last_item, "peak_memory")
            metrics.completion_tokens = int(generation_tokens) if isinstance(generation_tokens, int) else len(text_parts)
            if isinstance(prompt_tps, (int, float)) and prompt_tps > 0:
                metrics.prefill_tokens_per_second = float(prompt_tps)
                metrics.prefill_duration_ms = metrics.prefill_tokens / float(prompt_tps) * 1000
            if isinstance(generation_tps, (int, float)) and generation_tps > 0:
                metrics.generation_tokens_per_second = float(generation_tps)
                metrics.generation_duration_ms = metrics.completion_tokens / float(generation_tps) * 1000
            metrics.process_memory_bytes = after.get("process_memory_bytes")
            metrics.peak_memory_bytes = int(float(peak_memory) * 1_000_000_000) if isinstance(peak_memory, (int, float)) else None
            unified = after.get("unified_memory_bytes")
            if isinstance(unified, int) and unified > 0 and isinstance(metrics.peak_memory_bytes, int):
                metrics.extra["peak_memory_ratio"] = metrics.peak_memory_bytes / unified
            metrics.memory_pressure = after.get("memory_pressure")
            metrics.swap_before_bytes = before.get("swap_used_bytes")
            metrics.swap_after_bytes = after.get("swap_used_bytes")
            if isinstance(metrics.swap_before_bytes, int) and isinstance(metrics.swap_after_bytes, int):
                metrics.swap_delta_bytes = metrics.swap_after_bytes - metrics.swap_before_bytes
            metrics.finish_reason = self._item_value(last_item, "finish_reason", "stop") or "stop"
            result = GenerationResult(
                request_id=request.request_id,
                model_artifact_id=request.model_artifact_id,
                engine=self.name,
                text="".join(text_parts),
                finish_reason=metrics.finish_reason,
                usage=TokenUsage(
                    prompt_tokens=metrics.prefill_tokens,
                    completion_tokens=metrics.completion_tokens,
                    total_tokens=(metrics.prefill_tokens or 0) + (metrics.completion_tokens or 0),
                ),
                metrics=metrics,
            )
            self._last_metrics = metrics.to_dict()
            yield StreamEvent(
                type="completed",
                request_id=request.request_id,
                sequence=len(text_parts) + 1,
                done=True,
                result=result.to_dict(),
            )
        except (RequestCancelledError, UnsupportedGenerationSettingError):
            raise
        except Exception as exc:
            message = str(exc).lower()
            details: dict[str, Any] = {"failure_kind": "inference_failure", "measurement_provenance": "observed"}
            if "metal" in message and ("alloc" in message or "memory" in message or "out of" in message):
                details["failure_kind"] = "metal_allocation_failure"
            elif "context" in message or "sequence" in message:
                details["failure_kind"] = "context_failure"
            raise EngineRuntimeError("MLX generation failed", details=details) from exc
        finally:
            with self._lock:
                self._active.pop(request.request_id, None)

    def generate(self, request: GenerationRequest, cancel_event: threading.Event) -> GenerationResult:
        completed: dict[str, Any] | None = None
        for event in self.stream(request, cancel_event):
            if event.type == "completed":
                completed = event.result
            elif event.type == "error":
                if (event.error or {}).get("code") == "cancelled":
                    raise RequestCancelledError("generation was cancelled")
                if (event.error or {}).get("code") == "runtime_timeout":
                    timeout_details = dict((event.error or {}).get("details") or {})
                    timeout_details.setdefault("timeout_semantics", "cooperative")
                    raise RuntimeTimeoutError(
                        "generation exceeded timeout_ms",
                        details=timeout_details,
                    )
                raise EngineRuntimeError("MLX generation failed", details=event.error or {})
        if completed is None:
            raise EngineRuntimeError("MLX generation ended without a result")
        return self._result_from_payload(completed)

    def cancel(self, request_id: str) -> bool:
        with self._lock:
            event = self._active.get(request_id)
        if event is None:
            return False
        event.set()
        return True

    def unload(self, artifact_id: str) -> dict[str, Any]:
        with self._lock:
            loaded = self._loaded
            if loaded is None or loaded.artifact_id != artifact_id:
                return {"unloaded": False, "noop": True, "artifact_id": artifact_id}
            self._loaded = None
            self._loaded_execution_input = None
            self._model = None
            self._tokenizer = None
            self._stream_generate = None
        started = time.perf_counter()
        try:
            mlx, _, _ = self._modules()
            if mlx is not None:
                clear_cache = getattr(mlx, "clear_cache", None)
                if callable(clear_cache):
                    self._run_on_mlx_thread(clear_cache)
            gc.collect()
        except (ImportError, ModuleNotFoundError, RuntimeError, AttributeError):
            return {
                "unloaded": True,
                "artifact_id": artifact_id,
                "cleanup_status": "cleanup_error",
                "unload_duration_ms": (time.perf_counter() - started) * 1000,
                "measurement_provenance": "observed",
            }
        return {
            "unloaded": True,
            "artifact_id": artifact_id,
            "cleanup_status": "clean",
            "unload_duration_ms": (time.perf_counter() - started) * 1000,
            "measurement_provenance": "observed",
        }

    def runtime_metrics(self) -> dict[str, Any]:
        with self._lock:
            active = list(self._active)
        return {"engine": self.name, "active_request_ids": active, "last_metrics": dict(self._last_metrics)}
