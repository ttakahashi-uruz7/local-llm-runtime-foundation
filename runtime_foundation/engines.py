"""Engine family identity and canonical wire names.

Engine families are internal identifiers. Adapter and service boundaries use
the canonical wire identifier mapped here, independently of artifact format.
"""

from __future__ import annotations

from enum import Enum


class EngineFamily(str, Enum):
    MOCK = "mock"
    MLX = "mlx"
    LLAMA_CPP = "llama_cpp"


_WIRE_IDENTIFIERS = {
    EngineFamily.MOCK: "mock",
    EngineFamily.MLX: "mlx",
    EngineFamily.LLAMA_CPP: "llama.cpp",
}

_ALIASES = {
    "mock": EngineFamily.MOCK,
    "mlx": EngineFamily.MLX,
    "mlx-lm": EngineFamily.MLX,
    "mlx_lm": EngineFamily.MLX,
    "llama.cpp": EngineFamily.LLAMA_CPP,
    "llama_cpp": EngineFamily.LLAMA_CPP,
    "llama-cpp": EngineFamily.LLAMA_CPP,
}

_DEFAULT_ENGINE_BY_ARTIFACT_FORMAT = {
    "mlx": EngineFamily.MLX,
    "safetensors": EngineFamily.MLX,
    "gguf": EngineFamily.LLAMA_CPP,
}


def engine_family(identifier: str) -> EngineFamily | None:
    """Resolve a known alias to its internal family, without guessing unknowns."""

    if not isinstance(identifier, str):
        return None
    return _ALIASES.get(identifier.strip().lower())


def engine_wire_identifier(family: EngineFamily) -> str:
    """Return the canonical external name for an internal engine family."""

    return _WIRE_IDENTIFIERS[family]


def normalize_engine_identifier(identifier: str) -> str:
    """Normalize known aliases while preserving custom adapter identifiers."""

    normalized = identifier.strip()
    if not normalized:
        return normalized
    family = engine_family(normalized)
    return engine_wire_identifier(family) if family is not None else normalized


def default_engine_for_artifact_format(artifact_format: str) -> EngineFamily | None:
    """Select a default family from a format; this does not validate capability."""

    if not isinstance(artifact_format, str):
        return None
    return _DEFAULT_ENGINE_BY_ARTIFACT_FORMAT.get(artifact_format.strip().lower())
