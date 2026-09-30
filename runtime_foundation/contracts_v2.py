"""Foundation Contract v2 primitives.

This module is deliberately additive.  The v1 contracts remain in
``runtime_foundation.contracts`` and are still the wire contract used by the
existing service paths.  v2 adds explicit execution provenance and stable
canonical fingerprints without importing Studio policy into Foundation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .contracts import (
    CONTRACT_VERSION,
    GENERATION_REQUEST_VERSION,
    GenerationRequest,
    GenerationResult,
    ModelArtifactBinding,
    RuntimeOptions,
    _mapping,
    _non_negative_int,
)
from .errors import ArtifactCompatibilityError, UnsupportedArtifactLocatorError, UnsupportedExecutionInputError
from .engines import normalize_engine_identifier

CONTRACT_V2_VERSION = "runtime-foundation.contract.v2"
GENERATION_REQUEST_V2_VERSION = "runtime-foundation.generation-request.v2"
GENERATION_RESULT_V2_VERSION = "runtime-foundation.generation-result.v2"
EXECUTION_TRACE_V2_VERSION = "runtime-foundation.execution-trace.v2"
EXECUTION_GUARD_VERSION = "runtime-foundation.execution-guard.v1"
EXECUTION_BINDING_VERSION = "runtime-foundation.execution-binding.v1"
BUILD_IDENTITY_VERSION = "runtime-foundation.build-identity.v1"
EXECUTION_INPUT_VERSION = "runtime-foundation.execution-input.v1"
ADAPTER_LINEAGE_VERSION = "runtime-foundation.adapter-lineage.v1"
ADAPTER_LINEAGE_METADATA_KEY = "runtime_foundation_adapter_lineage"

SUPPORTED_CONTRACT_VERSIONS = (CONTRACT_VERSION, CONTRACT_V2_VERSION)

COMPLETE_FILE_HASH_SCHEME = "complete-file-v1"
COMPLETE_DIRECTORY_HASH_SCHEME = "complete-directory-manifest-v1"
FAST_FILE_HASH_SCHEME = "fast-file-v1"
FAST_DIRECTORY_HASH_SCHEME = "fast-directory-manifest-v1"


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible data deterministically.

    Object key order is normalized and insignificant whitespace is removed.
    ``allow_nan=False`` prevents non-JSON numeric values from acquiring a
    platform-specific representation.
    """

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_fingerprint(value: Any) -> str:
    """Return a self-describing SHA-256 fingerprint for canonical JSON data."""

    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _canonical_copy(value: Any) -> Any:
    """Return a detached JSON value suitable for stable contract snapshots."""

    return json.loads(canonical_json(value))


def is_valid_fingerprint(value: Any) -> bool:
    """Return whether *value* uses the current canonical fingerprint format."""

    return isinstance(value, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is not None


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")
    return value.strip()


def _optional_text(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _raw_sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    length = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            length += len(chunk)
            digest.update(chunk)
    return length, digest.hexdigest()


def _directory_files(path: Path) -> list[tuple[bytes, Path]]:
    entries = [(item.relative_to(path).as_posix().encode("utf-8"), item) for item in path.rglob("*") if item.is_file()]
    entries.sort(key=lambda item: item[0])
    return entries


def _complete_file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update((COMPLETE_FILE_HASH_SCHEME + "\0").encode("ascii"))
    length, _ = _raw_sha256_file(path)
    digest.update(length.to_bytes(8, "big"))
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _complete_directory_digest(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update((COMPLETE_DIRECTORY_HASH_SCHEME + "\0").encode("ascii"))
    entries = _directory_files(path)
    digest.update(len(entries).to_bytes(8, "big"))
    for relative, child in entries:
        length, file_digest = _raw_sha256_file(child)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(b"file\0")
        digest.update(length.to_bytes(8, "big"))
        digest.update(bytes.fromhex(file_digest))
    return digest.hexdigest()


def _complete_file(path: Path) -> tuple[str, str]:
    if path.is_dir():
        return _complete_directory_digest(path), COMPLETE_DIRECTORY_HASH_SCHEME
    return _complete_file_digest(path), COMPLETE_FILE_HASH_SCHEME


def _fast_sample_digest(path: Path) -> str:
    stat = path.stat()
    digest = hashlib.sha256()
    digest.update(stat.st_size.to_bytes(8, "big"))
    digest.update(stat.st_mtime_ns.to_bytes(8, "big", signed=True))
    if path.is_file():
        with path.open("rb") as handle:
            first = handle.read(64 * 1024)
            digest.update(first)
            if stat.st_size > len(first):
                handle.seek(max(0, stat.st_size - 64 * 1024))
                digest.update(handle.read(64 * 1024))
    return digest.hexdigest()


def _fast_file_digest(path: Path) -> str:
    """Create a development/change-detection digest without claiming full content."""

    digest = hashlib.sha256()
    if path.is_dir():
        digest.update((FAST_DIRECTORY_HASH_SCHEME + "\0").encode("ascii"))
        entries = _directory_files(path)
        digest.update(len(entries).to_bytes(8, "big"))
        for relative, child in entries:
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(bytes.fromhex(_fast_sample_digest(child)))
        return digest.hexdigest()
    digest.update((FAST_FILE_HASH_SCHEME + "\0").encode("ascii"))
    digest.update(bytes.fromhex(_fast_sample_digest(path)))
    return digest.hexdigest()


class ContentIdentityScheme(str, Enum):
    COMPLETE = "complete"
    FAST = "fast"
    LEGACY = "legacy"


@dataclass(frozen=True)
class ContentIdentity:
    """Content identity independent of any filesystem locator."""

    algorithm: str
    digest: str
    scope: str = "artifact"
    scheme: str = ContentIdentityScheme.COMPLETE.value
    canonicalization_scheme: str | None = None

    def __post_init__(self) -> None:
        algorithm = _required_text(self.algorithm, "content_identity.algorithm").lower()
        digest = _required_text(self.digest, "content_identity.digest")
        scope = _required_text(self.scope, "content_identity.scope")
        scheme = _required_text(self.scheme, "content_identity.scheme").lower()
        canonicalization = _optional_text(self.canonicalization_scheme, "content_identity.canonicalization_scheme")
        if ":" in digest and digest.lower().startswith(f"{algorithm}:"):
            digest = digest.split(":", 1)[1]
        allowed_canonicalization: dict[str, set[str | None]] = {
            ContentIdentityScheme.COMPLETE.value: {COMPLETE_FILE_HASH_SCHEME, COMPLETE_DIRECTORY_HASH_SCHEME},
            ContentIdentityScheme.FAST.value: {FAST_FILE_HASH_SCHEME, FAST_DIRECTORY_HASH_SCHEME},
            ContentIdentityScheme.LEGACY.value: {None},
        }
        allowed_values = allowed_canonicalization.get(scheme)
        if allowed_values is None:
            raise ValueError(f"unsupported content identity scheme: {scheme}")
        if canonicalization not in allowed_values:
            raise ValueError("content_identity.canonicalization_scheme is required and unsupported for this scheme")
        if algorithm == "sha256" and scheme == ContentIdentityScheme.COMPLETE.value:
            if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
                raise ValueError("complete sha256 content identity requires a 64-character hexadecimal digest")
            digest = digest.lower()
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "scheme", scheme)
        object.__setattr__(self, "canonicalization_scheme", canonicalization)

    @classmethod
    def from_payload(cls, payload: Any) -> ContentIdentity:
        value = _mapping(payload, "content_identity")
        allowed = {"algorithm", "digest", "scope", "scheme", "canonicalization_scheme"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unsupported fields in content_identity: {', '.join(unknown)}")
        return cls(
            algorithm=value.get("algorithm", "sha256"),
            digest=_required_text(value.get("digest"), "content_identity.digest"),
            scope=_required_text(value.get("scope", "artifact"), "content_identity.scope"),
            scheme=_required_text(value.get("scheme", ContentIdentityScheme.COMPLETE.value), "content_identity.scheme"),
            canonicalization_scheme=value.get("canonicalization_scheme"),
        )

    @classmethod
    def from_legacy_hash(cls, value: str) -> ContentIdentity:
        raw = _required_text(value, "artifact_hash")
        algorithm, separator, digest = raw.partition(":")
        if not separator:
            algorithm, digest = "sha256", raw
        return cls(algorithm=algorithm, digest=digest, scope="artifact", scheme=ContentIdentityScheme.LEGACY.value)

    @classmethod
    def from_file(cls, path: str | Path, *, scheme: str = "complete", scope: str = "artifact") -> ContentIdentity:
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(target)
        normalized_scheme = _required_text(scheme, "scheme").lower()
        canonicalization: str
        if normalized_scheme == ContentIdentityScheme.COMPLETE.value:
            digest, canonicalization = _complete_file(target)
        elif normalized_scheme == ContentIdentityScheme.FAST.value:
            digest = _fast_file_digest(target)
            canonicalization = FAST_DIRECTORY_HASH_SCHEME if target.is_dir() else FAST_FILE_HASH_SCHEME
        else:
            raise ValueError("content identity scheme must be complete or fast")
        return cls(
            algorithm="sha256",
            digest=digest,
            scope=scope,
            scheme=normalized_scheme,
            canonicalization_scheme=canonicalization,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "digest": self.digest,
            "scope": self.scope,
            "scheme": self.scheme,
            "canonicalization_scheme": self.canonicalization_scheme,
        }


@dataclass(frozen=True)
class ArtifactLocator:
    """A place where an artifact can be found; never a content identity."""

    type: str
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", _required_text(self.type, "locator.type").lower())
        object.__setattr__(self, "value", _required_text(self.value, "locator.value"))

    @classmethod
    def from_payload(cls, payload: Any) -> ArtifactLocator:
        value = _mapping(payload, "locator")
        unknown = sorted(set(value) - {"type", "value"})
        if unknown:
            raise ValueError(f"unsupported fields in locator: {', '.join(unknown)}")
        return cls(
            type=_required_text(value.get("type"), "locator.type"),
            value=_required_text(value.get("value"), "locator.value"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"type": self.type, "value": self.value}


@dataclass(frozen=True)
class ArtifactBindingV2:
    """Registry identity, content identity, and locator with separate semantics."""

    artifact_id: str
    locator: ArtifactLocator
    content_identity: ContentIdentity | None = None
    format: str | None = None
    quantization: str | None = None
    revision: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _required_text(self.artifact_id, "artifact_id"))
        if self.format is not None:
            object.__setattr__(self, "format", _required_text(self.format, "format").lower())
        if not isinstance(self.metadata, dict):
            raise TypeError("artifact metadata must be an object")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def local_path(self) -> str | None:
        return self.locator.value if self.locator.type == "filesystem" else None

    @classmethod
    def from_payload(cls, payload: Any) -> ArtifactBindingV2:
        value = _mapping(payload, "artifact")
        allowed = {
            "contract_version",
            "registry_identity",
            "artifact_id",
            "content_identity",
            "locator",
            "local_path",
            "format",
            "quantization",
            "revision",
            "metadata",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unsupported fields in artifact v2: {', '.join(unknown)}")
        version = value.get("contract_version")
        if version not in {None, CONTRACT_V2_VERSION}:
            raise ValueError("artifact contract_version is not runtime-foundation.contract.v2")
        registry = (
            _mapping(value.get("registry_identity"), "registry_identity")
            if value.get("registry_identity") is not None
            else {}
        )
        artifact_id = value.get("artifact_id", registry.get("artifact_id"))
        locator_payload = value.get("locator")
        if locator_payload is None and value.get("local_path") is not None:
            locator_payload = {"type": "filesystem", "value": value.get("local_path")}
        return cls(
            artifact_id=_required_text(artifact_id, "artifact_id"),
            locator=ArtifactLocator.from_payload(locator_payload),
            content_identity=(
                ContentIdentity.from_payload(value["content_identity"])
                if value.get("content_identity") is not None
                else None
            ),
            format=value.get("format"),
            quantization=value.get("quantization"),
            revision=value.get("revision"),
            metadata=value.get("metadata", {}),
        )

    @classmethod
    def from_legacy(cls, artifact: ModelArtifactBinding) -> ArtifactBindingV2:
        content = ContentIdentity.from_legacy_hash(artifact.artifact_hash) if artifact.artifact_hash else None
        return cls(
            artifact_id=artifact.artifact_id,
            locator=ArtifactLocator(type="filesystem", value=artifact.local_path),
            content_identity=content,
            format=artifact.format,
            quantization=artifact.quantization,
            revision=artifact.revision,
            metadata=artifact.metadata,
        )

    def to_legacy(self) -> ModelArtifactBinding:
        if self.locator.type != "filesystem":
            raise UnsupportedArtifactLocatorError(
                "the current Foundation execution boundary supports filesystem locators only",
                details={"locator_type": self.locator.type, "artifact_id": self.artifact_id},
            )
        artifact_hash = None
        if self.content_identity is not None:
            artifact_hash = f"{self.content_identity.algorithm}:{self.content_identity.digest}"
        return ModelArtifactBinding(
            artifact_id=self.artifact_id,
            local_path=self.locator.value,
            format=self.format or "unknown",
            quantization=self.quantization,
            artifact_hash=artifact_hash,
            revision=self.revision,
            metadata=dict(self.metadata),
        )

    def canonical_payload(self) -> dict[str, Any]:
        """Return the certified artifact identity without execution location."""

        return {
            "registry_identity": {"artifact_id": self.artifact_id},
            "content_identity": self.content_identity.to_dict() if self.content_identity else None,
            "format": self.format,
            "quantization": self.quantization,
            "revision": self.revision,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_V2_VERSION,
            "registry_identity": {"artifact_id": self.artifact_id},
            "artifact_id": self.artifact_id,
            "content_identity": self.content_identity.to_dict() if self.content_identity else None,
            "locator": self.locator.to_dict(),
            "format": self.format,
            "quantization": self.quantization,
            "revision": self.revision,
            "metadata": dict(self.metadata),
        }


def _complete_identity(binding: ArtifactBindingV2, role: str) -> ContentIdentity:
    identity = binding.content_identity
    if identity is None or identity.algorithm != "sha256" or identity.scheme != ContentIdentityScheme.COMPLETE.value:
        raise ArtifactCompatibilityError(
            f"{role} requires a complete SHA-256 content identity",
            details={"role": role, "artifact_id": binding.artifact_id},
        )
    return identity


def _positive_shape(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly two dimensions")
    result: list[int] = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError(f"{field_name} dimensions must be positive integers")
        result.append(dimension)
    return result


@dataclass(frozen=True)
class AdapterTargetBaseV1:
    """Consumer asserted Base lineage for an adapter, with locator provenance only."""

    artifact_id: str
    content_identity: ContentIdentity
    revision: str
    format: str
    quantization: str
    model_identity: str
    tokenizer_identity: str
    locator: ArtifactLocator | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _required_text(self.artifact_id, "target_base.artifact_id"))
        object.__setattr__(self, "revision", _required_text(self.revision, "target_base.revision"))
        object.__setattr__(self, "format", _required_text(self.format, "target_base.format").lower())
        object.__setattr__(self, "quantization", _required_text(self.quantization, "target_base.quantization"))
        object.__setattr__(self, "model_identity", _required_text(self.model_identity, "target_base.model_identity"))
        object.__setattr__(
            self, "tokenizer_identity", _required_text(self.tokenizer_identity, "target_base.tokenizer_identity")
        )
        if not isinstance(self.content_identity, ContentIdentity):
            raise ValueError("target_base.content_identity must be a ContentIdentity")
        if self.locator is not None and not isinstance(self.locator, ArtifactLocator):
            raise ValueError("target_base.locator must be an ArtifactLocator")

    @classmethod
    def from_payload(cls, payload: Any) -> AdapterTargetBaseV1:
        value = _mapping(payload, "adapter_lineage.target_base")
        allowed = {
            "artifact_id", "content_identity", "revision", "format", "quantization",
            "model_identity", "tokenizer_identity", "locator",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unsupported fields in adapter target_base: {', '.join(unknown)}")
        return cls(
            artifact_id=value.get("artifact_id"),
            content_identity=ContentIdentity.from_payload(value.get("content_identity")),
            revision=value.get("revision"),
            format=value.get("format"),
            quantization=value.get("quantization"),
            model_identity=value.get("model_identity"),
            tokenizer_identity=value.get("tokenizer_identity"),
            locator=ArtifactLocator.from_payload(value["locator"]) if value.get("locator") is not None else None,
        )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "content_identity": self.content_identity.to_dict(),
            "revision": self.revision,
            "format": self.format,
            "quantization": self.quantization,
            "model_identity": self.model_identity,
            "tokenizer_identity": self.tokenizer_identity,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.canonical_payload()
        payload["locator"] = self.locator.to_dict() if self.locator else None
        return payload


@dataclass(frozen=True)
class AdapterLineageV1:
    target_base: AdapterTargetBaseV1
    rank: int
    target_modules: tuple[str, ...]
    tensor_shapes: dict[str, dict[str, list[int]]]
    schema_version: str = ADAPTER_LINEAGE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ADAPTER_LINEAGE_VERSION:
            raise ValueError(f"adapter lineage schema_version must be {ADAPTER_LINEAGE_VERSION}")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError("adapter lineage rank must be a positive integer")
        if not isinstance(self.target_base, AdapterTargetBaseV1):
            raise ValueError("adapter lineage target_base must be validated")
        if not isinstance(self.target_modules, (tuple, list)) or not self.target_modules:
            raise ValueError("adapter lineage target_modules must be a non-empty array")
        modules = tuple(_required_text(item, "adapter_lineage.target_modules[]") for item in self.target_modules)
        if len(set(modules)) != len(modules):
            raise ValueError("adapter lineage target_modules must not contain duplicates")
        if not isinstance(self.tensor_shapes, dict):
            raise ValueError("adapter lineage tensor_shapes must be an object")
        normalized: dict[str, dict[str, list[int]]] = {}
        if set(self.tensor_shapes) != set(modules):
            raise ValueError("adapter lineage tensor_shapes must describe every target module exactly once")
        for module in modules:
            entry = _mapping(self.tensor_shapes[module], f"adapter_lineage.tensor_shapes.{module}")
            if set(entry) != {"lora_a", "lora_b"}:
                raise ValueError(f"adapter lineage tensor_shapes.{module} must contain lora_a and lora_b")
            normalized[module] = {
                "lora_a": _positive_shape(entry["lora_a"], f"{module}.lora_a"),
                "lora_b": _positive_shape(entry["lora_b"], f"{module}.lora_b"),
            }
        object.__setattr__(self, "target_modules", modules)
        object.__setattr__(self, "tensor_shapes", normalized)

    @classmethod
    def from_payload(cls, payload: Any) -> AdapterLineageV1:
        value = _mapping(payload, "adapter_lineage")
        allowed = {"schema_version", "target_base", "rank", "target_modules", "tensor_shapes"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unsupported fields in adapter_lineage: {', '.join(unknown)}")
        if value.get("schema_version") != ADAPTER_LINEAGE_VERSION:
            raise ValueError(f"adapter_lineage.schema_version must be {ADAPTER_LINEAGE_VERSION}")
        modules = value.get("target_modules")
        if not isinstance(modules, list):
            raise ValueError("adapter_lineage.target_modules must be an array")
        return cls(
            target_base=AdapterTargetBaseV1.from_payload(value.get("target_base")),
            rank=value.get("rank"),
            target_modules=tuple(modules),
            tensor_shapes=value.get("tensor_shapes"),
            schema_version=value["schema_version"],
        )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "target_base": self.target_base.canonical_payload(),
            "rank": self.rank,
            "target_modules": list(self.target_modules),
            "tensor_shapes": _canonical_copy(self.tensor_shapes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.canonical_payload(),
            "target_base": self.target_base.to_dict(),
        }


@dataclass(frozen=True)
class ExecutionInputV1:
    """Versioned Base plus zero-or-one adapter execution composition."""

    base: ArtifactBindingV2
    adapters: tuple[ArtifactBindingV2, ...] = ()
    kind: str = "base_plus_adapters"
    schema_version: str = EXECUTION_INPUT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTION_INPUT_VERSION:
            raise ValueError(f"execution_input.schema_version must be {EXECUTION_INPUT_VERSION}")
        if self.kind != "base_plus_adapters":
            raise ValueError("execution_input.kind must be base_plus_adapters")
        if not isinstance(self.base, ArtifactBindingV2):
            raise ValueError("execution_input.base must be an ArtifactBindingV2")
        if not isinstance(self.adapters, (tuple, list)):
            raise ValueError("execution_input.adapters must be an array")
        adapters = tuple(self.adapters)
        if len(adapters) > 1:
            raise UnsupportedExecutionInputError(
                "multiple adapters are not supported by this Foundation execution input version",
                details={"adapter_count": len(adapters), "maximum_supported": 1},
            )
        if any(not isinstance(item, ArtifactBindingV2) for item in adapters):
            raise ValueError("execution_input.adapters entries must be ArtifactBindingV2 values")
        if self.base.locator.type != "filesystem":
            raise UnsupportedArtifactLocatorError(
                "the current Foundation execution boundary supports filesystem locators only",
                details={"role": "base", "locator_type": self.base.locator.type},
            )
        if self.base.format is None or self.base.quantization is None or self.base.revision is None:
            raise ArtifactCompatibilityError(
                "Base binding requires format, quantization, and revision for direct execution",
                details={"artifact_id": self.base.artifact_id},
            )
        _complete_identity(self.base, "base")
        try:
            base_metadata = _canonical_copy(self.base.metadata)
        except (TypeError, ValueError) as exc:
            raise ValueError("Base metadata must be JSON-compatible") from exc
        for key in ("model_identity", "tokenizer_identity"):
            if not isinstance(base_metadata.get(key), str) or not base_metadata[key].strip():
                raise ArtifactCompatibilityError(
                    f"Base binding requires consumer-supplied {key}",
                    details={"artifact_id": self.base.artifact_id, "missing_field": f"metadata.{key}"},
                )
        shapes = base_metadata.get("target_module_shapes")
        if len(adapters) == 1 and (not isinstance(shapes, dict) or not shapes):
            raise ArtifactCompatibilityError(
                "Base binding requires a target_module_shapes manifest for compatibility validation",
                details={"artifact_id": self.base.artifact_id},
            )
        normalized_shapes: dict[str, dict[str, int]] = {}
        if shapes is not None and not isinstance(shapes, dict):
            raise ArtifactCompatibilityError(
                "Base target_module_shapes manifest must be an object",
                details={"artifact_id": self.base.artifact_id},
            )
        for module, raw_shape in (shapes or {}).items():
            if not isinstance(module, str) or not module.strip():
                raise ArtifactCompatibilityError("Base target module names must be non-empty strings")
            try:
                shape = _mapping(raw_shape, f"target_module_shapes.{module}")
                if set(shape) != {"in_features", "out_features"}:
                    raise ValueError("expected in_features and out_features")
                dimensions = [shape["in_features"], shape["out_features"]]
                if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in dimensions):
                    raise ValueError("dimensions must be positive integers")
                normalized_shapes[module] = {"in_features": dimensions[0], "out_features": dimensions[1]}
            except ValueError as exc:
                raise ArtifactCompatibilityError(
                    "Base target module shape manifest is malformed",
                    details={"module": module, "reason": str(exc)},
                ) from exc
        if len(adapters) == 1:
            adapter = adapters[0]
            if adapter.locator.type != "filesystem":
                raise UnsupportedArtifactLocatorError(
                    "the current Foundation execution boundary supports filesystem locators only",
                    details={"role": "adapter", "locator_type": adapter.locator.type},
                )
            if adapter.format is None or adapter.revision is None:
                raise ArtifactCompatibilityError(
                    "Adapter binding requires format and revision for direct execution",
                    details={"artifact_id": adapter.artifact_id},
                )
            _complete_identity(adapter, "adapter")
            raw_lineage = adapter.metadata.get(ADAPTER_LINEAGE_METADATA_KEY)
            if not isinstance(raw_lineage, dict):
                raise ArtifactCompatibilityError(
                    "Adapter binding requires consumer-supplied versioned Base lineage",
                    details={"artifact_id": adapter.artifact_id, "missing_field": f"metadata.{ADAPTER_LINEAGE_METADATA_KEY}"},
                )
            lineage = AdapterLineageV1.from_payload(raw_lineage)
            target = lineage.target_base
            expected_base = {
                "artifact_id": self.base.artifact_id,
                "content_identity": self.base.content_identity.to_dict(),
                "revision": self.base.revision,
                "format": self.base.format,
                "quantization": self.base.quantization,
                "model_identity": base_metadata["model_identity"],
                "tokenizer_identity": base_metadata["tokenizer_identity"],
            }
            actual_base = target.canonical_payload()
            if actual_base != expected_base:
                raise ArtifactCompatibilityError(
                    "Adapter target Base lineage does not match the requested Base",
                    details={"expected_target_base": expected_base, "adapter_target_base": actual_base},
                )
            for module in lineage.target_modules:
                if module not in normalized_shapes:
                    raise ArtifactCompatibilityError(
                        "Adapter targets a module absent from the Base compatibility manifest",
                        details={"module": module, "artifact_id": self.base.artifact_id},
                    )
                target_shape = normalized_shapes[module]
                factors = lineage.tensor_shapes[module]
                if factors["lora_a"] != [target_shape["in_features"], lineage.rank] or factors["lora_b"] != [
                    lineage.rank, target_shape["out_features"]
                ]:
                    raise ArtifactCompatibilityError(
                        "Adapter tensor shapes are incompatible with the target Base module",
                        details={"module": module, "rank": lineage.rank, "base_shape": target_shape, "adapter_shapes": factors},
                    )
        copied_adapters = tuple(
            ArtifactBindingV2(
                artifact_id=item.artifact_id,
                locator=item.locator,
                content_identity=item.content_identity,
                format=item.format,
                quantization=item.quantization,
                revision=item.revision,
                metadata=_canonical_copy(item.metadata),
            )
            for item in adapters
        )
        object.__setattr__(self, "adapters", copied_adapters)
        object.__setattr__(self, "base", ArtifactBindingV2(
            artifact_id=self.base.artifact_id,
            locator=self.base.locator,
            content_identity=self.base.content_identity,
            format=self.base.format,
            quantization=self.base.quantization,
            revision=self.base.revision,
            metadata={**base_metadata, "target_module_shapes": normalized_shapes},
        ))

    @classmethod
    def from_payload(cls, payload: Any) -> ExecutionInputV1:
        value = _mapping(payload, "execution_input")
        allowed = {
            "contract_version", "schema_version", "kind", "base", "adapters", "execution_input_fingerprint"
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unsupported fields in execution_input: {', '.join(unknown)}")
        if value.get("contract_version") != CONTRACT_V2_VERSION:
            raise ValueError(f"execution_input.contract_version must be {CONTRACT_V2_VERSION}")
        if value.get("schema_version") != EXECUTION_INPUT_VERSION:
            raise ValueError(f"execution_input.schema_version must be {EXECUTION_INPUT_VERSION}")
        adapters = value.get("adapters")
        if not isinstance(adapters, list):
            raise ValueError("execution_input.adapters must be an array")
        if len(adapters) > 1:
            raise UnsupportedExecutionInputError(
                "multiple adapters are not supported by this Foundation execution input version",
                details={"adapter_count": len(adapters), "maximum_supported": 1},
            )
        parsed = cls(
            base=ArtifactBindingV2.from_payload(value.get("base")),
            adapters=tuple(ArtifactBindingV2.from_payload(item) for item in adapters),
            kind=value.get("kind"),
        )
        expected = value.get("execution_input_fingerprint")
        if expected is not None and expected != parsed.fingerprint:
            raise ValueError("execution_input_fingerprint does not match execution input")
        return parsed

    def canonical_payload(self) -> dict[str, Any]:
        base_identity = {
            **self.base.canonical_payload(),
            "model_identity": self.base.metadata["model_identity"],
            "tokenizer_identity": self.base.metadata["tokenizer_identity"],
            "target_module_shapes": _canonical_copy(self.base.metadata.get("target_module_shapes", {})),
        }
        adapter_payloads = []
        for index, adapter in enumerate(self.adapters):
            lineage = AdapterLineageV1.from_payload(adapter.metadata[ADAPTER_LINEAGE_METADATA_KEY])
            adapter_payloads.append({
                "role": "lora_adapter",
                "order": index,
                "artifact_binding": adapter.canonical_payload(),
                "lineage": lineage.canonical_payload(),
            })
        return {
            "contract_version": CONTRACT_V2_VERSION,
            "schema_version": self.schema_version,
            "kind": self.kind,
            "base": base_identity,
            "adapters": adapter_payloads,
        }

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.canonical_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_V2_VERSION,
            "schema_version": self.schema_version,
            "kind": self.kind,
            "base": _canonical_copy(self.base.to_dict()),
            "adapters": [_canonical_copy(item.to_dict()) for item in self.adapters],
            "execution_input_fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class EngineImplementationBinding:
    id: str
    version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _required_text(self.id, "implementation.id"))
        object.__setattr__(self, "version", _optional_text(self.version, "implementation.version"))

    def to_dict(self) -> dict[str, str | None]:
        return {"id": self.id, "version": self.version}


@dataclass(frozen=True)
class BuildIdentityV1:
    """A deterministic identity for an observed runtime component set."""

    kind: str
    fingerprint: str
    components: dict[str, Any]
    schema_version: str = BUILD_IDENTITY_VERSION

    def __post_init__(self) -> None:
        kind = _required_text(self.kind, "build_identity.kind")
        fingerprint = _required_text(self.fingerprint, "build_identity.fingerprint")
        if not is_valid_fingerprint(fingerprint):
            raise ValueError("build_identity.fingerprint must use sha256:<64 lowercase hex characters>")
        if not isinstance(self.components, dict):
            raise ValueError("build_identity.components must be an object")
        schema_version = _required_text(self.schema_version, "build_identity.schema_version")
        if schema_version != BUILD_IDENTITY_VERSION:
            raise ValueError(f"build_identity.schema_version must be {BUILD_IDENTITY_VERSION}")
        try:
            normalized_components = json.loads(canonical_json(self.components))
            expected = canonical_fingerprint(normalized_components)
        except (TypeError, ValueError) as exc:
            raise ValueError("build_identity.components must be JSON-compatible") from exc
        if fingerprint != expected:
            raise ValueError("build_identity.fingerprint does not match canonical components")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "fingerprint", fingerprint)
        object.__setattr__(self, "components", normalized_components)
        object.__setattr__(self, "schema_version", schema_version)

    @classmethod
    def from_components(cls, *, kind: str, components: dict[str, Any]) -> BuildIdentityV1:
        return cls(kind=kind, fingerprint=canonical_fingerprint(components), components=components)

    @classmethod
    def from_payload(cls, payload: Any) -> BuildIdentityV1:
        value = _mapping(payload, "build_identity")
        allowed = {"schema_version", "kind", "fingerprint", "components"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unsupported fields in build_identity: {', '.join(unknown)}")
        kind = _required_text(value.get("kind"), "build_identity.kind")
        fingerprint = _required_text(value.get("fingerprint"), "build_identity.fingerprint")
        components = value.get("components")
        if not isinstance(components, dict):
            raise ValueError("build_identity.components must be an object")
        return cls(
            kind=kind,
            fingerprint=fingerprint,
            components=components,
            schema_version=value.get("schema_version", BUILD_IDENTITY_VERSION),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "fingerprint": self.fingerprint,
            "components": dict(self.components),
        }


def _optional_build_identity(value: Any) -> BuildIdentityV1 | None:
    """Read structured identity and downgrade old string wire values to unknown."""

    if value is None:
        return None
    if isinstance(value, BuildIdentityV1):
        return value
    if isinstance(value, str):
        # RAH-1's string field could contain a package version.  It is legacy
        # evidence only and must never be promoted to a certified identity.
        return None
    return BuildIdentityV1.from_payload(value)


@dataclass(frozen=True)
class EngineBindingV2:
    family: str
    implementation: EngineImplementationBinding
    build_identity: BuildIdentityV1 | None
    adapter_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "family", _required_text(self.family, "engine.family").lower())
        object.__setattr__(self, "build_identity", _optional_build_identity(self.build_identity))
        object.__setattr__(self, "adapter_id", _required_text(self.adapter_id, "engine.adapter_id"))

    @classmethod
    def from_legacy(
        cls,
        identity: Any,
        *,
        adapter_id: str,
        build_identity: BuildIdentityV1 | None = None,
    ) -> EngineBindingV2:
        engine = normalize_engine_identifier(_required_text(getattr(identity, "engine", None), "engine"))
        if engine == "mlx":
            family, implementation_id = "mlx", "mlx-lm"
        elif engine == "mock":
            family, implementation_id = "mock", "mock-runtime"
        elif engine == "llama.cpp":
            family, implementation_id = "llama.cpp", "llama.cpp"
        else:
            family, implementation_id = engine, engine
        return cls(
            family=family,
            implementation=EngineImplementationBinding(
                id=implementation_id,
                version=getattr(identity, "version", None),
            ),
            build_identity=build_identity,
            adapter_id=adapter_id,
        )

    @classmethod
    def from_payload(cls, payload: Any) -> EngineBindingV2:
        value = _mapping(payload, "engine_binding")
        implementation = EngineImplementationBinding(**_mapping(value.get("implementation"), "implementation"))
        return cls(
            family=_required_text(value.get("family"), "engine.family"),
            implementation=implementation,
            build_identity=_optional_build_identity(value.get("build_identity")),
            adapter_id=_required_text(value.get("adapter_id"), "engine.adapter_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "implementation": self.implementation.to_dict(),
            "build_identity": self.build_identity.to_dict() if self.build_identity else None,
            "adapter_id": self.adapter_id,
        }

    @property
    def implementation_id(self) -> str:
        return self.implementation.id

    @property
    def implementation_version(self) -> str | None:
        return self.implementation.version


@dataclass(frozen=True)
class FoundationBindingV2:
    contract_version: str
    foundation_version: str
    build_identity: BuildIdentityV1 | None
    adapter_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "contract_version", _required_text(self.contract_version, "foundation.contract_version")
        )
        object.__setattr__(
            self, "foundation_version", _required_text(self.foundation_version, "foundation.foundation_version")
        )
        object.__setattr__(self, "build_identity", _optional_build_identity(self.build_identity))
        object.__setattr__(self, "adapter_id", _required_text(self.adapter_id, "foundation.adapter_id"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "foundation_version": self.foundation_version,
            "build_identity": self.build_identity.to_dict() if self.build_identity else None,
            "adapter_id": self.adapter_id,
        }


@dataclass(frozen=True)
class RuntimeSettingsBindingV2:
    runtime_options_schema_version: str
    exact_settings: dict[str, Any] | None
    exact_settings_fingerprint: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "runtime_options_schema_version",
            _required_text(self.runtime_options_schema_version, "runtime_options_schema_version"),
        )
        if self.exact_settings is not None:
            if not isinstance(self.exact_settings, dict):
                raise ValueError("exact_settings must be an object")
            exact = dict(self.exact_settings)
            object.__setattr__(self, "exact_settings", exact)
            expected = canonical_fingerprint(exact)
            if self.exact_settings_fingerprint is not None and self.exact_settings_fingerprint != expected:
                raise ValueError("exact_settings_fingerprint does not match exact_settings")
            object.__setattr__(self, "exact_settings_fingerprint", expected)

    @classmethod
    def from_effective(cls, options: RuntimeOptions | None) -> RuntimeSettingsBindingV2 | None:
        if options is None:
            return None
        return cls(
            runtime_options_schema_version=options.schema_version,
            exact_settings=options.to_dict(),
        )

    @classmethod
    def from_payload(cls, payload: Any) -> RuntimeSettingsBindingV2:
        value = _mapping(payload, "runtime_settings_binding")
        return cls(
            runtime_options_schema_version=_required_text(
                value.get("runtime_options_schema_version"), "runtime_options_schema_version"
            ),
            exact_settings=value.get("exact_settings"),
            exact_settings_fingerprint=value.get("exact_settings_fingerprint"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_options_schema_version": self.runtime_options_schema_version,
            "exact_settings": dict(self.exact_settings) if self.exact_settings is not None else None,
            "exact_settings_fingerprint": self.exact_settings_fingerprint,
        }


@dataclass(frozen=True)
class ExecutionBindingV2:
    artifact_binding: ArtifactBindingV2
    engine_binding: EngineBindingV2
    foundation_binding: FoundationBindingV2
    runtime_settings_binding: RuntimeSettingsBindingV2 | None
    execution_input: ExecutionInputV1 | None = None

    def canonical_payload(self) -> dict[str, Any]:
        payload = {
            "contract_version": CONTRACT_V2_VERSION,
            "schema_version": EXECUTION_BINDING_VERSION,
            # Locator is recorded in the trace/artifact binding, but it is
            # deliberately excluded from certified execution identity.
            "artifact_binding": self.artifact_binding.canonical_payload(),
            "engine_binding": self.engine_binding.to_dict(),
            "foundation_binding": self.foundation_binding.to_dict(),
            "runtime_settings_binding": self.runtime_settings_binding.to_dict()
            if self.runtime_settings_binding
            else None,
        }
        if self.execution_input is not None:
            payload["execution_input"] = self.execution_input.canonical_payload()
        return payload

    @property
    def fingerprint(self) -> str:
        return canonical_fingerprint(self.canonical_payload())

    @classmethod
    def from_payload(cls, payload: Any) -> ExecutionBindingV2:
        value = _mapping(payload, "execution_binding")
        expected = value.get("execution_binding_fingerprint")
        binding = cls(
            artifact_binding=ArtifactBindingV2.from_payload(value.get("artifact_binding")),
            engine_binding=EngineBindingV2.from_payload(value.get("engine_binding")),
            foundation_binding=FoundationBindingV2(**_mapping(value.get("foundation_binding"), "foundation_binding")),
            runtime_settings_binding=(
                RuntimeSettingsBindingV2.from_payload(value["runtime_settings_binding"])
                if value.get("runtime_settings_binding") is not None
                else None
            ),
            execution_input=(
                ExecutionInputV1.from_payload(value["execution_input"])
                if value.get("execution_input") is not None
                else None
            ),
        )
        if expected is not None and expected != binding.fingerprint:
            raise ValueError("execution_binding_fingerprint does not match execution binding")
        return binding

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "contract_version": CONTRACT_V2_VERSION,
            "schema_version": EXECUTION_BINDING_VERSION,
            # The wire binding records the locator for provenance.  The
            # fingerprint is still computed from canonical_payload(), which
            # intentionally excludes it.
            "artifact_binding": self.artifact_binding.to_dict(),
            "engine_binding": self.engine_binding.to_dict(),
            "foundation_binding": self.foundation_binding.to_dict(),
            "runtime_settings_binding": self.runtime_settings_binding.to_dict()
            if self.runtime_settings_binding
            else None,
        }
        if self.execution_input is not None:
            payload["execution_input"] = self.execution_input.to_dict()
        return {**payload, "execution_binding_fingerprint": self.fingerprint}


class ThinkingMode(str, Enum):
    OFF = "OFF"
    AUTO = "AUTO"
    ON = "ON"


class ThinkingEffort(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass(frozen=True)
class ThinkingIntent:
    mode: ThinkingMode | str = ThinkingMode.AUTO
    effort: ThinkingEffort | str | None = None
    budget_tokens: int | None = None

    def __post_init__(self) -> None:
        try:
            mode = self.mode if isinstance(self.mode, ThinkingMode) else ThinkingMode(str(self.mode).upper())
        except ValueError as exc:
            raise ValueError("thinking_intent.mode must be OFF, AUTO, or ON") from exc
        effort: ThinkingEffort | None
        if self.effort is None:
            effort = None
        else:
            try:
                effort = (
                    self.effort if isinstance(self.effort, ThinkingEffort) else ThinkingEffort(str(self.effort).upper())
                )
            except ValueError as exc:
                raise ValueError("thinking_intent.effort must be LOW, MEDIUM, or HIGH") from exc
        budget = _non_negative_int(self.budget_tokens, "thinking_intent.budget_tokens", maximum=10_000_000)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "effort", effort)
        object.__setattr__(self, "budget_tokens", budget)

    @classmethod
    def from_payload(cls, payload: Any) -> ThinkingIntent:
        if payload is None:
            return cls()
        value = _mapping(payload, "thinking_intent")
        unknown = sorted(set(value) - {"mode", "effort", "budget_tokens"})
        if unknown:
            raise ValueError(f"unsupported fields in thinking_intent: {', '.join(unknown)}")
        return cls(mode=value.get("mode", "AUTO"), effort=value.get("effort"), budget_tokens=value.get("budget_tokens"))

    @classmethod
    def from_legacy(cls, thinking_enabled: bool | None) -> ThinkingIntent:
        if thinking_enabled is False:
            return cls(mode=ThinkingMode.OFF)
        if thinking_enabled is True:
            return cls(mode=ThinkingMode.ON)
        return cls(mode=ThinkingMode.AUTO)

    def to_legacy_enabled(self) -> bool | None:
        if self.mode == ThinkingMode.OFF:
            return False
        if self.mode == ThinkingMode.ON:
            return True
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value if isinstance(self.mode, ThinkingMode) else str(self.mode),
            "effort": (
                self.effort.value
                if isinstance(self.effort, ThinkingEffort)
                else str(self.effort)
                if self.effort is not None
                else None
            ),
            "budget_tokens": self.budget_tokens,
        }


@dataclass(frozen=True)
class ThinkingResolution:
    requested: ThinkingIntent
    studio_resolved: ThinkingIntent | None
    foundation_effective: ThinkingIntent | None
    status: str = "unresolved"
    reason: str | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested.to_dict(),
            "studio_resolved": self.studio_resolved.to_dict() if self.studio_resolved else None,
            "foundation_effective": self.foundation_effective.to_dict() if self.foundation_effective else None,
            "status": self.status,
            "reason": self.reason,
            "error": dict(self.error) if self.error else None,
        }


@dataclass(frozen=True)
class ExecutionGuardV1:
    expected_execution_binding_fingerprint: str | None = None
    expected_runtime_settings_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_V2_VERSION,
            "schema_version": EXECUTION_GUARD_VERSION,
            "expected_execution_binding_fingerprint": self.expected_execution_binding_fingerprint,
            "expected_runtime_settings_fingerprint": self.expected_runtime_settings_fingerprint,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> ExecutionGuardV1 | None:
        if payload is None:
            return None
        value = _mapping(payload, "execution_guard")
        allowed = {
            "contract_version",
            "schema_version",
            "expected_execution_binding_fingerprint",
            "expected_runtime_settings_fingerprint",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ValueError(f"unsupported fields in execution_guard: {', '.join(unknown)}")
        return cls(
            expected_execution_binding_fingerprint=_optional_text(
                value.get("expected_execution_binding_fingerprint"), "expected_execution_binding_fingerprint"
            ),
            expected_runtime_settings_fingerprint=_optional_text(
                value.get("expected_runtime_settings_fingerprint"), "expected_runtime_settings_fingerprint"
            ),
        )


@dataclass(frozen=True)
class ExecutionGuardVerification:
    """Additive evidence for one execution guard decision."""

    status: str
    mismatch_category: str | None = None
    expected_execution_binding_fingerprint: str | None = None
    actual_execution_binding_fingerprint: str | None = None
    expected_runtime_settings_fingerprint: str | None = None
    actual_runtime_settings_fingerprint: str | None = None
    generation_started: bool = False
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "mismatch_category": self.mismatch_category,
            "expected_execution_binding_fingerprint": self.expected_execution_binding_fingerprint,
            "actual_execution_binding_fingerprint": self.actual_execution_binding_fingerprint,
            "expected_runtime_settings_fingerprint": self.expected_runtime_settings_fingerprint,
            "actual_runtime_settings_fingerprint": self.actual_runtime_settings_fingerprint,
            "generation_started": self.generation_started,
            "error": dict(self.error) if self.error else None,
        }


# Public short spelling for callers that use the contract concept rather than
# its versioned wire name.
ExecutionGuard = ExecutionGuardV1


@dataclass(frozen=True)
class GenerationRequestV2(GenerationRequest):
    """Generation request v2 with explicit thinking intent and guard fields."""

    thinking_intent: ThinkingIntent = field(default_factory=ThinkingIntent)
    studio_resolved_thinking: ThinkingIntent | None = None
    execution_guard: ExecutionGuardV1 | None = None
    execution_input: ExecutionInputV1 | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> GenerationRequestV2:
        value = _mapping(payload, "generation_request")
        version = value.get("contract_version")
        schema = value.get("schema_version")
        if version not in {None, CONTRACT_VERSION, CONTRACT_V2_VERSION}:
            raise ValueError("generation_request contract_version is not supported")
        if schema not in {None, GENERATION_REQUEST_VERSION, GENERATION_REQUEST_V2_VERSION}:
            raise ValueError("generation_request schema_version is not supported")
        legacy_payload = dict(value)
        legacy_payload["contract_version"] = CONTRACT_VERSION
        legacy_payload["schema_version"] = GENERATION_REQUEST_VERSION
        thinking_intent_payload = legacy_payload.pop("thinking_intent", None)
        studio_payload = legacy_payload.pop("studio_resolved_thinking", None)
        guard_payload = legacy_payload.pop("execution_guard", None)
        execution_input_payload = legacy_payload.pop("execution_input", None)
        expected_execution = legacy_payload.pop("expected_execution_binding_fingerprint", None)
        expected_settings = legacy_payload.pop("expected_runtime_settings_fingerprint", None)
        base = GenerationRequest.from_payload(legacy_payload)
        thinking_intent = (
            ThinkingIntent.from_payload(thinking_intent_payload)
            if thinking_intent_payload is not None
            else ThinkingIntent.from_legacy(base.thinking_enabled)
        )
        guard = ExecutionGuardV1.from_payload(guard_payload)
        if expected_execution is not None or expected_settings is not None:
            top_level_guard = ExecutionGuardV1(
                expected_execution_binding_fingerprint=expected_execution,
                expected_runtime_settings_fingerprint=expected_settings,
            )
            if guard is not None and guard != top_level_guard:
                raise ValueError("execution_guard and top-level expected fingerprints disagree")
            guard = top_level_guard
        return cls(
            model_artifact_id=base.model_artifact_id,
            messages=base.messages,
            request_id=base.request_id,
            consumer_id=base.consumer_id,
            lease_id=base.lease_id,
            max_tokens=base.max_tokens,
            temperature=base.temperature,
            top_p=base.top_p,
            thinking_enabled=base.thinking_enabled,
            timeout_ms=base.timeout_ms,
            runtime_options=base.runtime_options,
            metadata=base.metadata,
            thinking_intent=thinking_intent,
            studio_resolved_thinking=ThinkingIntent.from_payload(studio_payload)
            if studio_payload is not None
            else None,
            execution_guard=guard,
            execution_input=ExecutionInputV1.from_payload(execution_input_payload)
            if execution_input_payload is not None
            else None,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = super().to_dict()
        payload.update(
            {
                "contract_version": CONTRACT_V2_VERSION,
                "schema_version": GENERATION_REQUEST_V2_VERSION,
                "thinking_intent": self.thinking_intent.to_dict(),
                "studio_resolved_thinking": self.studio_resolved_thinking.to_dict()
                if self.studio_resolved_thinking
                else None,
                "execution_guard": self.execution_guard.to_dict() if self.execution_guard else None,
            }
        )
        if self.execution_guard:
            payload["expected_execution_binding_fingerprint"] = (
                self.execution_guard.expected_execution_binding_fingerprint
            )
            payload["expected_runtime_settings_fingerprint"] = (
                self.execution_guard.expected_runtime_settings_fingerprint
            )
        if self.execution_input is not None:
            payload["execution_input"] = self.execution_input.to_dict()
        return payload


@dataclass
class ExecutionTraceV2:
    execution_id: str
    request_id: str
    request_contract_version: str
    execution_binding: ExecutionBindingV2
    thinking_resolution: ThinkingResolution
    started_at: str
    guard: ExecutionGuardV1 | None = None
    guard_verification: ExecutionGuardVerification | None = None
    generation_started: bool = False
    finished_at: str | None = None
    status: str = "running"
    metrics: dict[str, Any] | None = None
    finish_reason: str | None = None
    raw_execution_error: dict[str, Any] | None = None

    @property
    def execution_binding_fingerprint(self) -> str | None:
        return (
            self.execution_binding.fingerprint if self.execution_binding.runtime_settings_binding is not None else None
        )

    @property
    def artifact_binding(self) -> ArtifactBindingV2:
        return self.execution_binding.artifact_binding

    @property
    def engine_binding(self) -> EngineBindingV2:
        return self.execution_binding.engine_binding

    @property
    def foundation_binding(self) -> FoundationBindingV2:
        return self.execution_binding.foundation_binding

    @property
    def runtime_settings_binding(self) -> RuntimeSettingsBindingV2 | None:
        return self.execution_binding.runtime_settings_binding

    @property
    def execution_input(self) -> ExecutionInputV1 | None:
        return self.execution_binding.execution_input

    def with_runtime_settings(self, options: RuntimeOptions) -> None:
        self.execution_binding = ExecutionBindingV2(
            artifact_binding=self.execution_binding.artifact_binding,
            engine_binding=self.execution_binding.engine_binding,
            foundation_binding=self.execution_binding.foundation_binding,
            runtime_settings_binding=RuntimeSettingsBindingV2.from_effective(options),
            execution_input=self.execution_binding.execution_input,
        )

    def to_dict(self) -> dict[str, Any]:
        binding = self.execution_binding.to_dict()
        payload = {
            "contract_version": CONTRACT_V2_VERSION,
            "schema_version": EXECUTION_TRACE_V2_VERSION,
            "execution_id": self.execution_id,
            "request_id": self.request_id,
            "request_contract_version": self.request_contract_version,
            "execution_binding": binding,
            "execution_binding_fingerprint": self.execution_binding_fingerprint,
            "artifact_binding": self.artifact_binding.to_dict(),
            "engine_binding": self.engine_binding.to_dict(),
            "foundation_binding": self.foundation_binding.to_dict(),
            "effective_runtime_settings_identity": self.runtime_settings_binding.to_dict()
            if self.runtime_settings_binding
            else None,
            "thinking_requested": self.thinking_resolution.requested.to_dict(),
            "thinking_studio_resolved": self.thinking_resolution.studio_resolved.to_dict()
            if self.thinking_resolution.studio_resolved
            else None,
            "thinking_foundation_effective": self.thinking_resolution.foundation_effective.to_dict()
            if self.thinking_resolution.foundation_effective
            else None,
            "thinking_resolution": self.thinking_resolution.to_dict(),
            "execution_guard": self.guard.to_dict() if self.guard else None,
            "guard_verification": self.guard_verification.to_dict() if self.guard_verification else None,
            "generation_started": self.generation_started,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "metrics": self.metrics,
            "finish_reason": self.finish_reason,
            "raw_execution_error": self.raw_execution_error,
        }
        if self.execution_input is not None:
            payload["execution_input"] = self.execution_input.to_dict()
        return payload


@dataclass
class GenerationResultV2:
    """v2 result envelope derived from a v1 result without rewriting it."""

    result: GenerationResult
    trace: ExecutionTraceV2

    @classmethod
    def from_result(cls, result: GenerationResult, trace: ExecutionTraceV2) -> GenerationResultV2:
        return cls(result=result, trace=trace)

    def to_dict(self) -> dict[str, Any]:
        payload = self.result.to_dict(include_v2=False)
        payload.update(
            {
                "contract_version": CONTRACT_V2_VERSION,
                "schema_version": GENERATION_RESULT_V2_VERSION,
                "execution_binding": self.trace.execution_binding.to_dict(),
                "execution_binding_fingerprint": self.trace.execution_binding_fingerprint,
                "trace": self.trace.to_dict(),
                "trace_v1": self.result.trace.to_dict() if self.result.trace else None,
            }
        )
        return payload


def build_execution_binding(
    *,
    artifact: ModelArtifactBinding | ArtifactBindingV2,
    engine: Any,
    adapter_id: str,
    engine_build_identity: BuildIdentityV1 | None = None,
    foundation_version: str,
    foundation_build_identity: BuildIdentityV1 | None,
    effective_runtime_options: RuntimeOptions | None,
    execution_input: ExecutionInputV1 | None = None,
) -> ExecutionBindingV2:
    return ExecutionBindingV2(
        artifact_binding=artifact
        if isinstance(artifact, ArtifactBindingV2)
        else ArtifactBindingV2.from_legacy(artifact),
        engine_binding=EngineBindingV2.from_legacy(
            engine,
            adapter_id=adapter_id,
            build_identity=engine_build_identity,
        ),
        foundation_binding=FoundationBindingV2(
            contract_version=CONTRACT_V2_VERSION,
            foundation_version=foundation_version,
            build_identity=foundation_build_identity,
            adapter_id=adapter_id,
        ),
        runtime_settings_binding=RuntimeSettingsBindingV2.from_effective(effective_runtime_options),
        execution_input=execution_input,
    )


__all__ = [
    "BUILD_IDENTITY_VERSION",
    "ADAPTER_LINEAGE_METADATA_KEY",
    "ADAPTER_LINEAGE_VERSION",
    "CONTRACT_V2_VERSION",
    "EXECUTION_BINDING_VERSION",
    "EXECUTION_GUARD_VERSION",
    "EXECUTION_INPUT_VERSION",
    "EXECUTION_TRACE_V2_VERSION",
    "GENERATION_REQUEST_V2_VERSION",
    "GENERATION_RESULT_V2_VERSION",
    "SUPPORTED_CONTRACT_VERSIONS",
    "ArtifactBindingV2",
    "AdapterLineageV1",
    "AdapterTargetBaseV1",
    "ArtifactLocator",
    "BuildIdentityV1",
    "ContentIdentity",
    "ContentIdentityScheme",
    "EngineBindingV2",
    "EngineImplementationBinding",
    "ExecutionBindingV2",
    "ExecutionInputV1",
    "ExecutionGuard",
    "ExecutionGuardV1",
    "ExecutionGuardVerification",
    "ExecutionTraceV2",
    "FoundationBindingV2",
    "GenerationRequestV2",
    "GenerationResultV2",
    "RuntimeSettingsBindingV2",
    "ThinkingEffort",
    "ThinkingIntent",
    "ThinkingMode",
    "ThinkingResolution",
    "build_execution_binding",
    "canonical_fingerprint",
    "canonical_json",
    "is_valid_fingerprint",
]
