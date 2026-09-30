"""Strict, read-only GGUF header inspection and load-time identity pinning.

This module reads the GGUF container header without loading tensor payloads.
The native engine remains responsible for validating tensor support. The
validated file descriptor is kept open while the native loader runs so POSIX
engines can load through a descriptor path instead of re-opening a mutable
filesystem locator.
"""

from __future__ import annotations

import hashlib
import os
import platform
import stat
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from .contracts_v2 import ArtifactBindingV2, ContentIdentity
from .errors import ArtifactCompatibilityError, ArtifactNotFoundError

GGUF_MAGIC = b"GGUF"
SUPPORTED_GGUF_VERSIONS = {2, 3}
MAX_GGUF_HEADER_BYTES = 256 * 1024 * 1024
MAX_GGUF_STRING_BYTES = 16 * 1024 * 1024
MAX_GGUF_ARRAY_ITEMS = 10_000_000

_SCALARS: dict[int, tuple[str, int]] = {
    0: ("B", 1),
    1: ("b", 1),
    2: ("H", 2),
    3: ("h", 2),
    4: ("I", 4),
    5: ("i", 4),
    6: ("f", 4),
    7: ("?", 1),
    10: ("Q", 8),
    11: ("q", 8),
    12: ("d", 8),
}
_FILE_TYPES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    4: "Q4_1_SOME_F16",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
    38: "MXFP4_MOE",
    39: "NVFP4",
    40: "Q1_0",
    41: "Q2_0",
}


class _Reader:
    def __init__(self, handle: BinaryIO, file_size: int) -> None:
        self.handle = handle
        self.file_size = file_size
        self.start = handle.tell()

    @property
    def consumed(self) -> int:
        return self.handle.tell() - self.start

    def read(self, size: int) -> bytes:
        if size < 0 or self.consumed + size > MAX_GGUF_HEADER_BYTES:
            raise ValueError("GGUF header exceeds the supported inspection limit")
        data = self.handle.read(size)
        if len(data) != size:
            raise ValueError("GGUF header is truncated")
        return data

    def unpack(self, fmt: str) -> tuple[Any, ...]:
        size = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.read(size))

    def string(self) -> str:
        length = self.unpack("Q")[0]
        if length > MAX_GGUF_STRING_BYTES or length > self.file_size:
            raise ValueError("GGUF string length is invalid")
        try:
            return self.read(length).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("GGUF string is not valid UTF-8") from exc

    def value(self, value_type: int) -> Any:
        if value_type == 8:
            return self.string()
        if value_type == 9:
            element_type, count = self.unpack("IQ")
            if count > MAX_GGUF_ARRAY_ITEMS:
                raise ValueError("GGUF metadata array exceeds the supported inspection limit")
            if element_type == 9:
                raise ValueError("nested GGUF metadata arrays are invalid")
            return [self.value(element_type) for _ in range(count)]
        scalar = _SCALARS.get(value_type)
        if scalar is None:
            raise ValueError(f"GGUF metadata uses unknown value type {value_type}")
        if value_type == 7:
            raw = self.read(1)[0]
            if raw not in {0, 1}:
                raise ValueError("GGUF boolean metadata must be encoded as 0 or 1")
            return bool(raw)
        value = self.unpack(scalar[0])[0]
        return value


@dataclass(frozen=True)
class GGUFObservedMetadata:
    format: str
    version: int
    architecture: str
    file_type: int
    quantization: str
    context_length: int | None
    chat_template: str | tuple[str, ...] | None
    tensor_count: int
    metadata: dict[str, Any]
    content_identity: ContentIdentity

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "architecture": self.architecture,
            "file_type": self.file_type,
            "quantization": self.quantization,
            "context_length": self.context_length,
            "chat_template": self.chat_template,
            "tensor_count": self.tensor_count,
            "metadata": dict(self.metadata),
            "content_identity": self.content_identity.to_dict(),
        }


def _read_header(handle: BinaryIO, file_size: int, content_identity: ContentIdentity) -> GGUFObservedMetadata:
    reader = _Reader(handle, file_size)
    if reader.read(4) != GGUF_MAGIC:
        raise ValueError("artifact does not have GGUF magic")
    version = reader.unpack("I")[0]
    if version not in SUPPORTED_GGUF_VERSIONS:
        raise ValueError(f"GGUF version {version} is not supported")
    tensor_count, metadata_count = reader.unpack("QQ")
    if tensor_count < 1 or tensor_count > MAX_GGUF_ARRAY_ITEMS:
        raise ValueError("GGUF tensor count is invalid")
    if metadata_count > MAX_GGUF_ARRAY_ITEMS:
        raise ValueError("GGUF metadata count is invalid")
    metadata: dict[str, Any] = {}
    for _ in range(metadata_count):
        key = reader.string()
        if not key or key in metadata:
            raise ValueError("GGUF metadata keys must be non-empty and unique")
        value_type = reader.unpack("I")[0]
        metadata[key] = reader.value(value_type)

    architecture = metadata.get("general.architecture")
    file_type = metadata.get("general.file_type")
    if not isinstance(architecture, str) or not architecture.strip():
        raise ValueError("GGUF metadata is missing general.architecture")
    if isinstance(file_type, bool) or not isinstance(file_type, int) or file_type < 0:
        raise ValueError("GGUF metadata is missing a valid general.file_type")
    quantization = _FILE_TYPES.get(file_type)
    if quantization is None:
        quantization = f"GGUF_FILE_TYPE_{file_type}"

    context = metadata.get(f"{architecture}.context_length")
    if isinstance(context, bool) or (context is not None and (not isinstance(context, int) or context <= 0)):
        raise ValueError("GGUF context length metadata is invalid")
    template = metadata.get("tokenizer.chat_template")
    if template is not None and not isinstance(template, (str, list)):
        raise ValueError("GGUF chat template metadata has an unsupported value shape")
    if isinstance(template, list):
        if not all(isinstance(item, str) for item in template):
            raise ValueError("GGUF chat template array must contain strings")
        template = tuple(template)

    alignment = metadata.get("general.alignment", 32)
    if isinstance(alignment, bool) or not isinstance(alignment, int) or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("GGUF general.alignment must be a positive power of two")
    tensor_names: set[str] = set()
    tensor_records: list[tuple[int, tuple[int, ...], int]] = []
    for _ in range(tensor_count):
        name = reader.string()
        if not name or name in tensor_names:
            raise ValueError("GGUF tensor names must be non-empty and unique")
        tensor_names.add(name)
        dimensions = reader.unpack("I")[0]
        if dimensions < 1 or dimensions > 4:
            raise ValueError("GGUF tensor dimension count is invalid")
        shape = reader.unpack("Q" * dimensions)
        if any(dimension == 0 for dimension in shape):
            raise ValueError("GGUF tensor dimensions must be non-zero")
        tensor_type = reader.unpack("I")[0]
        offset = reader.unpack("Q")[0]
        if tensor_type > 4096:
            raise ValueError("GGUF tensor type is invalid")
        # The data section follows the descriptors, so only check the declared
        # offset here. The native engine validates the type-specific byte size.
        tensor_records.append((offset, tuple(shape), tensor_type))
    data_start = ((handle.tell() + alignment - 1) // alignment) * alignment
    if data_start > file_size:
        raise ValueError("GGUF tensor data section is truncated")
    tensor_size_by_type = {
        0: (1, 4),  # F32
        1: (1, 2),  # F16
        24: (1, 1),  # I8
        25: (1, 2),  # I16
        26: (1, 4),  # I32
        27: (1, 8),  # I64
        28: (1, 8),  # F64
        30: (1, 2),  # BF16
    }
    for offset, shape, tensor_type in tensor_records:
        tensor_start = data_start + offset
        if offset % alignment != 0 or tensor_start >= file_size:
            raise ValueError("GGUF tensor offset is unaligned or outside the file")
        block = tensor_size_by_type.get(tensor_type)
        if block is not None:
            block_elements, block_bytes = block
            if shape[0] % block_elements != 0:
                raise ValueError("GGUF tensor dimensions do not match the declared tensor type")
            element_count = 1
            for dimension in shape:
                element_count *= dimension
            tensor_bytes = (element_count // block_elements) * block_bytes
            if tensor_start + tensor_bytes > file_size:
                raise ValueError("GGUF tensor data is truncated")
    return GGUFObservedMetadata(
        format="gguf",
        version=version,
        architecture=architecture.strip(),
        file_type=file_type,
        quantization=quantization,
        context_length=context,
        chat_template=template,
        tensor_count=tensor_count,
        metadata=metadata,
        content_identity=content_identity,
    )


def _complete_identity_from_fd(fd: int, size: int) -> ContentIdentity:
    digest = hashlib.sha256()
    digest.update(b"complete-file-v1\0")
    digest.update(size.to_bytes(8, "big"))
    offset = 0
    while offset < size:
        read_size = min(1024 * 1024, size - offset)
        if hasattr(os, "pread"):
            chunk = os.pread(fd, read_size, offset)
        else:  # Windows does not expose pread; only one thread owns validation here.
            current = os.lseek(fd, 0, os.SEEK_CUR)
            os.lseek(fd, offset, os.SEEK_SET)
            chunk = os.read(fd, read_size)
            os.lseek(fd, current, os.SEEK_SET)
        if not chunk:
            raise ArtifactCompatibilityError("GGUF artifact changed while its content identity was being computed")
        digest.update(chunk)
        offset += len(chunk)
    return ContentIdentity(
        algorithm="sha256",
        digest=digest.hexdigest(),
        scope="artifact",
        scheme="complete",
        canonicalization_scheme="complete-file-v1",
    )


def _matches_declared_quantization(claim: str, observed: str) -> bool:
    normalized = "".join(character for character in claim.upper() if character.isalnum())
    actual = "".join(character for character in observed.upper() if character.isalnum())
    return normalized == actual


def _validate_claims(artifact: ArtifactBindingV2, observed: GGUFObservedMetadata) -> None:
    if (artifact.format or "").lower() != "gguf":
        raise ArtifactCompatibilityError(
            "llama.cpp GGUF execution requires artifact format gguf",
            details={"artifact_id": artifact.artifact_id, "declared_format": artifact.format, "observed_format": "gguf"},
        )
    if artifact.quantization and not _matches_declared_quantization(artifact.quantization, observed.quantization):
        raise ArtifactCompatibilityError(
            "declared GGUF quantization does not match observed file metadata",
            details={
                "artifact_id": artifact.artifact_id,
                "declared_quantization": artifact.quantization,
                "observed_quantization": observed.quantization,
                "observed_file_type": observed.file_type,
            },
        )
    declared_architecture = artifact.metadata.get("architecture", artifact.metadata.get("general.architecture"))
    if declared_architecture is not None and declared_architecture != observed.architecture:
        raise ArtifactCompatibilityError(
            "declared GGUF architecture does not match observed file metadata",
            details={
                "artifact_id": artifact.artifact_id,
                "declared_architecture": declared_architecture,
                "observed_architecture": observed.architecture,
            },
        )
    declared_context = artifact.metadata.get("context_length")
    if declared_context is not None and declared_context != observed.context_length:
        raise ArtifactCompatibilityError(
            "declared GGUF context length does not match observed file metadata",
            details={
                "artifact_id": artifact.artifact_id,
                "declared_context_length": declared_context,
                "observed_context_length": observed.context_length,
            },
        )
    expected = artifact.content_identity
    identities_match = expected is None or expected == observed.content_identity
    # v1 callers only provide an opaque sha256:... field. When Core has
    # upgraded a previously validated GGUF binding, that legacy-shaped value
    # carries the canonical complete digest and can be compared directly.
    if expected is not None and expected.scheme == "legacy":
        identities_match = (
            expected.algorithm == observed.content_identity.algorithm
            and expected.digest == observed.content_identity.digest
            and expected.scope == observed.content_identity.scope
        )
    if not identities_match:
        raise ArtifactCompatibilityError(
            "GGUF content identity does not match the complete SHA-256 of the file",
            details={
                "artifact_id": artifact.artifact_id,
                "expected_content_identity": expected.to_dict(),
                "observed_content_identity": observed.content_identity.to_dict(),
            },
        )


@dataclass
class VerifiedGGUFArtifact:
    """A verified GGUF file held open across native load to pin its identity."""

    source_path: Path
    fd: int
    stat_identity: tuple[int, int, int, int]
    observed: GGUFObservedMetadata
    pinned_path: str

    @classmethod
    def open(cls, artifact: ArtifactBindingV2) -> VerifiedGGUFArtifact:
        if artifact.locator.type != "filesystem":
            raise ArtifactCompatibilityError(
                "GGUF execution currently requires a filesystem artifact locator",
                details={"artifact_id": artifact.artifact_id, "locator_type": artifact.locator.type},
            )
        path = Path(artifact.locator.value)
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        except FileNotFoundError as exc:
            raise ArtifactNotFoundError(
                "GGUF artifact path does not exist",
                details={"artifact_id": artifact.artifact_id, "local_path": str(path)},
            ) from exc
        except OSError as exc:
            raise ArtifactCompatibilityError(
                "GGUF artifact could not be opened for validation",
                details={"artifact_id": artifact.artifact_id, "local_path": str(path), "reason": str(exc)},
            ) from exc
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise ArtifactCompatibilityError("GGUF artifact must be a regular file")
            stat_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            identity = _complete_identity_from_fd(fd, before.st_size)
            try:
                with os.fdopen(os.dup(fd), "rb") as handle:
                    observed = _read_header(handle, before.st_size, identity)
            except ValueError as exc:
                raise ArtifactCompatibilityError(
                    "GGUF artifact header is invalid",
                    details={"artifact_id": artifact.artifact_id, "reason": str(exc)},
                ) from exc
            after = os.fstat(fd)
            after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            if after_identity != stat_identity:
                raise ArtifactCompatibilityError("GGUF artifact changed during validation")
            _validate_claims(artifact, observed)
            # os.dup() shares the underlying open-file offset. Reset it after
            # header inspection so native loaders that open /dev/fd/<fd> begin
            # at byte zero (and descriptor consumers see the complete file).
            os.lseek(fd, 0, os.SEEK_SET)
            system = platform.system().lower()
            if system == "darwin":
                pinned_path = f"/dev/fd/{fd}"
            elif system == "linux":
                pinned_path = f"/proc/self/fd/{fd}"
            else:
                # Windows llama.cpp has no portable descriptor-path API. The
                # caller must revalidate identity after native load completes.
                pinned_path = str(path)
            return cls(path, fd, stat_identity, observed, pinned_path)
        except Exception:
            os.close(fd)
            raise

    def revalidate(self) -> None:
        before = os.fstat(self.fd)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        actual = _complete_identity_from_fd(self.fd, before.st_size)
        after = os.fstat(self.fd)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity != self.stat_identity or after_identity != self.stat_identity or actual != self.observed.content_identity:
            raise ArtifactCompatibilityError(
                "GGUF artifact changed between validation and native load",
                details={
                    "expected_content_identity": self.observed.content_identity.to_dict(),
                    "actual_content_identity": actual.to_dict(),
                    "pinned_descriptor": self.pinned_path,
                },
            )
        if platform.system().lower() not in {"darwin", "linux"}:
            try:
                path_stat = self.source_path.stat()
            except OSError as exc:
                raise ArtifactCompatibilityError("GGUF artifact path changed during native load") from exc
            path_identity = (path_stat.st_dev, path_stat.st_ino, path_stat.st_size, path_stat.st_mtime_ns)
            if path_identity != self.stat_identity:
                raise ArtifactCompatibilityError("GGUF artifact path was replaced during native load")

    def duplicate(self) -> VerifiedGGUFArtifact:
        """Return an independently owned descriptor for the same verified inode."""

        if self.fd < 0:
            raise ArtifactCompatibilityError("GGUF validation descriptor is already closed")
        try:
            fd = os.dup(self.fd)
            os.lseek(fd, 0, os.SEEK_SET)
        except OSError as exc:
            raise ArtifactCompatibilityError(
                "validated GGUF descriptor could not be duplicated",
                details={"reason": str(exc)},
            ) from exc
        system = platform.system().lower()
        if system == "darwin":
            pinned_path = f"/dev/fd/{fd}"
        elif system == "linux":
            pinned_path = f"/proc/self/fd/{fd}"
        else:
            pinned_path = str(self.source_path)
        return VerifiedGGUFArtifact(
            self.source_path,
            fd,
            self.stat_identity,
            self.observed,
            pinned_path,
        )

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> VerifiedGGUFArtifact:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
