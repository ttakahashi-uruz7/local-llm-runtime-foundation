from __future__ import annotations

from dataclasses import replace

import pytest

from runtime_foundation import (
    EngineBindingV2,
    EngineCapability,
    EngineIdentity,
    HostProfile,
    ModelArtifactBinding,
    RuntimeCore,
)
from runtime_foundation.adapters.mock import MockAdapter
from runtime_foundation.engines import (
    EngineFamily,
    default_engine_for_artifact_format,
    engine_family,
    engine_wire_identifier,
    normalize_engine_identifier,
)
from runtime_foundation.errors import ArtifactCompatibilityError, EngineNotFoundError


class CapabilityAdapter(MockAdapter):
    def __init__(self, name: str, artifact_formats: list[str]) -> None:
        super().__init__()
        self.name = name
        self._artifact_formats = artifact_formats

    def discover_capability(self) -> EngineCapability:
        return replace(super().discover_capability(), artifact_formats=self._artifact_formats)


@pytest.mark.parametrize(
    ("alias", "family", "wire"),
    [
        ("mlx", EngineFamily.MLX, "mlx"),
        ("mlx-lm", EngineFamily.MLX, "mlx"),
        ("mlx_lm", EngineFamily.MLX, "mlx"),
        ("llama.cpp", EngineFamily.LLAMA_CPP, "llama.cpp"),
        ("llama_cpp", EngineFamily.LLAMA_CPP, "llama.cpp"),
        ("llama-cpp", EngineFamily.LLAMA_CPP, "llama.cpp"),
    ],
)
def test_engine_aliases_resolve_to_internal_family_and_canonical_wire_name(alias, family, wire) -> None:
    assert engine_family(alias) is family
    assert engine_wire_identifier(family) == wire
    assert normalize_engine_identifier(alias) == wire


@pytest.mark.parametrize(
    ("artifact_format", "expected"),
    [
        ("mlx", EngineFamily.MLX),
        ("safetensors", EngineFamily.MLX),
        ("GGUF", EngineFamily.LLAMA_CPP),
        ("ggml", None),
        ("bin", None),
    ],
)
def test_default_engine_mapping_is_independent_from_compatibility(artifact_format, expected) -> None:
    assert default_engine_for_artifact_format(artifact_format) is expected


def test_gguf_default_selects_llama_cpp_but_capability_must_confirm_compatibility(tmp_path) -> None:
    artifact_path = tmp_path / "fixture.gguf"
    artifact_path.write_bytes(b"test fixture")
    artifact = ModelArtifactBinding("fixture", str(artifact_path), "gguf")
    adapter = CapabilityAdapter("llama.cpp", ["mlx"])
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"llama_cpp": adapter})

    with pytest.raises(ArtifactCompatibilityError) as caught:
        core._select_adapter(artifact, requested=None)

    assert caught.value.details["engine"] == "llama.cpp"
    assert caught.value.details["format"] == "gguf"
    assert caught.value.details["compatibility_status"] == "unsupported"


def test_explicit_llama_cpp_aliases_select_same_adapter_and_ggml_is_rejected(tmp_path) -> None:
    artifact_path = tmp_path / "fixture.gguf"
    artifact_path.write_bytes(b"test fixture")
    gguf = ModelArtifactBinding("fixture", str(artifact_path), "gguf")
    adapter = CapabilityAdapter("llama.cpp", ["gguf"])
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"llama-cpp": adapter})

    for alias in ("llama.cpp", "llama_cpp", "llama-cpp"):
        assert core._select_adapter(gguf, requested=alias) is adapter

    ggml = ModelArtifactBinding("legacy", str(artifact_path), "ggml")
    with pytest.raises(EngineNotFoundError):
        core._select_adapter(ggml, requested=None)
    with pytest.raises(ArtifactCompatibilityError):
        core._select_adapter(ggml, requested="llama_cpp")


def test_duplicate_adapter_aliases_are_rejected() -> None:
    with pytest.raises(ValueError, match="same engine identifier"):
        RuntimeCore(
            host_profile=HostProfile.mock_windows(),
            adapters={
                "llama.cpp": CapabilityAdapter("llama.cpp", ["gguf"]),
                "llama_cpp": CapabilityAdapter("llama.cpp", ["gguf"]),
            },
        )


def test_custom_engine_identifiers_keep_their_registered_name() -> None:
    assert normalize_engine_identifier("vendor.runtime") == "vendor.runtime"


def test_v2_engine_binding_keeps_canonical_llama_cpp_family_for_aliases() -> None:
    for alias in ("llama.cpp", "llama_cpp", "llama-cpp"):
        binding = EngineBindingV2.from_legacy(EngineIdentity(engine=alias), adapter_id="llama.cpp")
        assert binding.family == "llama.cpp"
        assert binding.implementation.id == "llama.cpp"
