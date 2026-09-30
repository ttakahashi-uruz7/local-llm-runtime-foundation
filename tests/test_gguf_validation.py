from __future__ import annotations

import os
import struct
from pathlib import Path

import pytest

from runtime_foundation.contracts_v2 import ArtifactBindingV2, ArtifactLocator, ContentIdentity
from runtime_foundation import HostProfile, RuntimeCore
from runtime_foundation.adapters.mock import MockAdapter
from runtime_foundation.errors import ArtifactCompatibilityError
from runtime_foundation.gguf import VerifiedGGUFArtifact


def _string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf(path: Path, *, architecture: str = "tiny", file_type: int = 0, context: int = 64) -> bytes:
    metadata = [
        ("general.architecture", 8, architecture),
        ("general.file_type", 4, file_type),
        (f"{architecture}.context_length", 4, context),
        ("tokenizer.chat_template", 8, "{% for message in messages %}{{ message.content }}{% endfor %}"),
        ("general.alignment", 4, 32),
    ]
    encoded = bytearray(b"GGUF")
    encoded.extend(struct.pack("<IQQ", 3, 1, len(metadata)))
    for key, value_type, value in metadata:
        encoded.extend(_string(key))
        encoded.extend(struct.pack("<I", value_type))
        if value_type == 8:
            encoded.extend(_string(value))
        else:
            encoded.extend(struct.pack("<I", value))
    encoded.extend(_string("weight"))
    encoded.extend(struct.pack("<I Q I Q", 1, 1, 0, 0))
    encoded.extend(b"\0" * ((32 - len(encoded) % 32) % 32))
    encoded.extend(struct.pack("<f", 0.25))
    path.write_bytes(encoded)
    return bytes(encoded)


def _artifact(path: Path, **kwargs: object) -> ArtifactBindingV2:
    return ArtifactBindingV2(
        artifact_id="test-gguf",
        locator=ArtifactLocator("filesystem", str(path)),
        format="gguf",
        **kwargs,
    )


def test_gguf_header_observation_and_complete_content_identity(tmp_path: Path) -> None:
    path = tmp_path / "fixture.gguf"
    contents = _gguf(path)
    with VerifiedGGUFArtifact.open(_artifact(path)) as verified:
        observed = verified.observed
        assert observed.format == "gguf"
        assert observed.version == 3
        assert observed.architecture == "tiny"
        assert observed.quantization == "F32"
        assert observed.context_length == 64
        assert observed.chat_template is not None
        assert observed.content_identity == ContentIdentity.from_file(path)
        assert observed.content_identity.scheme == "complete"
        assert verified.pinned_path != str(path) if os.name != "nt" else verified.pinned_path == str(path)
        verified.revalidate()
    assert path.read_bytes() == contents


def test_gguf_file_type_uses_llama_file_type_enum_not_tensor_type_enum(tmp_path: Path) -> None:
    path = tmp_path / "q4_k_m.gguf"
    _gguf(path, file_type=15)
    with VerifiedGGUFArtifact.open(_artifact(path, quantization="Q4_K_M")) as verified:
        assert verified.observed.quantization == "Q4_K_M"


@pytest.mark.parametrize(
    ("artifact_kwargs", "expected"),
    [
        ({"quantization": "Q4_K_M"}, "quantization"),
        ({"metadata": {"architecture": "other"}}, "architecture"),
        ({"metadata": {"context_length": 1024}}, "context length"),
    ],
)
def test_caller_claims_are_checked_against_gguf_metadata(
    tmp_path: Path, artifact_kwargs: dict[str, object], expected: str
) -> None:
    path = tmp_path / "fixture.gguf"
    _gguf(path)
    with pytest.raises(ArtifactCompatibilityError, match=expected):
        VerifiedGGUFArtifact.open(_artifact(path, **artifact_kwargs))


@pytest.mark.parametrize("contents", [b"not GGUF", b"GGUF\x03\x00", b"GGUF" + struct.pack("<IQQ", 99, 0, 0)])
def test_invalid_or_truncated_gguf_is_rejected(tmp_path: Path, contents: bytes) -> None:
    path = tmp_path / "invalid.gguf"
    path.write_bytes(contents)
    with pytest.raises((ArtifactCompatibilityError, ValueError)):
        VerifiedGGUFArtifact.open(_artifact(path))


def test_gguf_revalidation_detects_in_place_replacement(tmp_path: Path) -> None:
    path = tmp_path / "fixture.gguf"
    _gguf(path)
    verified = VerifiedGGUFArtifact.open(_artifact(path))
    try:
        with path.open("r+b") as handle:
            handle.seek(-4, os.SEEK_END)
            handle.write(struct.pack("<f", 0.5))
        with pytest.raises(ArtifactCompatibilityError, match="changed"):
            verified.revalidate()
    finally:
        verified.close()


def test_atomic_path_replacement_keeps_the_validated_descriptor_identity(tmp_path: Path) -> None:
    path = tmp_path / "fixture.gguf"
    original = _gguf(path)
    verified = VerifiedGGUFArtifact.open(_artifact(path))
    replacement = tmp_path / "replacement.gguf"
    _gguf(replacement, context=128)
    try:
        os.replace(replacement, path)
        if os.name == "nt":
            with pytest.raises(ArtifactCompatibilityError, match="replaced"):
                verified.revalidate()
        else:
            verified.revalidate()
            with Path(verified.pinned_path).open("rb") as pinned:
                assert pinned.read() == original
    finally:
        verified.close()


def test_gguf_validation_rejects_wrong_artifact_format(tmp_path: Path) -> None:
    path = tmp_path / "fixture.gguf"
    _gguf(path)
    artifact = ArtifactBindingV2(
        artifact_id="test-gguf",
        locator=ArtifactLocator("filesystem", str(path)),
        format="ggml",
    )
    with pytest.raises(ArtifactCompatibilityError, match="format gguf"):
        VerifiedGGUFArtifact.open(artifact)


def test_core_binds_observed_gguf_identity_before_load_and_records_observation(tmp_path: Path) -> None:
    path = tmp_path / "fixture.gguf"
    _gguf(path)
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"mock": MockAdapter()})

    loaded = core.load(_artifact(path), adapter="mock")

    assert loaded["artifact"]["format"] == "gguf"
    assert loaded["artifact"]["quantization"] == "F32"
    assert loaded["artifact"]["artifact_hash"].startswith("sha256:")
    assert loaded["raw"]["gguf_validation"]["architecture"] == "tiny"
    assert loaded["raw"]["gguf_validation"]["content_identity"]["scheme"] == "complete"
