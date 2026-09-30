from __future__ import annotations

import pytest

from runtime_foundation import (
    BuildIdentityV1,
    GenerationRequest,
    HostProfile,
    LoadOptions,
    ModelArtifactBinding,
    RuntimeCore,
    RuntimeOptions,
)
from runtime_foundation.adapters.mock import MockAdapter
from runtime_foundation.contracts_v3 import LoadOptionsResolutionV1
from runtime_foundation.errors import LoadConflictError, UnsupportedRuntimeOptionError


class V3LoadAdapter(MockAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.load_count = 0
        self.build_version = "1"

    def build_identity(self):
        return BuildIdentityV1.from_components(
            kind="test-load-adapter-build-v1",
            components={"adapter": {"version": self.build_version}},
        )

    def resolve_load_options(self, artifact, options):
        del artifact
        if options.acceleration.backend not in {"auto", "cpu"}:
            raise UnsupportedRuntimeOptionError("mock test adapter supports only auto or cpu load")
        return LoadOptionsResolutionV1(requested=options, resolved=options, effective=options)

    def load_with_options(self, artifact, resolution):
        self.load_count += 1
        raw = self.load(artifact)
        return {**raw, "effective_load_options": resolution.effective.to_dict()}


def _fixture(tmp_path):
    path = tmp_path / "fixture.bin"
    path.write_bytes(b"v3 load fixture")
    return ModelArtifactBinding("fixture", str(path), "bin")


def test_same_artifact_and_effective_load_options_reuse_one_load(tmp_path) -> None:
    adapter = V3LoadAdapter()
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"mock": adapter})
    artifact = _fixture(tmp_path)
    options = LoadOptions(model_context_size=4096, batch=128)

    first = core.load(artifact, adapter="mock", load_options=options, consumer_id="first")
    second = core.load(artifact, adapter="mock", load_options=options, consumer_id="second")

    assert first["reused"] is False
    assert second["reused"] is True
    assert adapter.load_count == 1
    assert first["raw"]["load_identity"]["fingerprint"] == second["raw"]["load_identity"]["fingerprint"]
    assert first["raw"]["load_options_resolution"]["requested"]["batch"] == 128


def test_load_options_mismatch_conflicts_until_explicit_unload(tmp_path) -> None:
    adapter = V3LoadAdapter()
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"mock": adapter})
    artifact = _fixture(tmp_path)
    first = core.load(
        artifact,
        adapter="mock",
        load_options=LoadOptions(model_context_size=4096),
        consumer_id="one",
    )
    second = core.load(artifact, adapter="mock", load_options=LoadOptions(model_context_size=4096), consumer_id="two")

    with pytest.raises(LoadConflictError) as caught:
        core.load(
            artifact,
            adapter="mock",
            load_options=LoadOptions(model_context_size=8192),
            consumer_id="three",
        )
    assert caught.value.details["loaded_load_identity_fingerprint"] == first["raw"]["load_identity"]["fingerprint"]
    assert caught.value.details["requested_load_options"]["model_context_size"] == 8192
    assert adapter.load_count == 1

    assert core.unload(artifact.artifact_id, consumer_id="one", lease_id=first["lease_id"])["unloaded"] is False
    assert core.unload(artifact.artifact_id, consumer_id="two", lease_id=second["lease_id"])["unloaded"] is True
    reloaded = core.load(
        artifact,
        adapter="mock",
        load_options=LoadOptions(model_context_size=8192),
        consumer_id="three",
    )
    assert reloaded["reused"] is False
    assert adapter.load_count == 2


def test_load_options_mismatch_while_generation_is_active_is_rejected(tmp_path) -> None:
    adapter = V3LoadAdapter()
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"mock": adapter})
    artifact = _fixture(tmp_path)
    loaded = core.load(
        artifact,
        adapter="mock",
        load_options=LoadOptions(model_context_size=4096),
        consumer_id="consumer",
    )
    request = GenerationRequest(
        model_artifact_id=artifact.artifact_id,
        request_id="active-v3-generation",
        consumer_id="consumer",
        lease_id=loaded["lease_id"],
        messages=[{"role": "user", "content": "wait"}],
        runtime_options=RuntimeOptions.from_payload({"engine_options": {"mock.chunk_delay_ms": 50}}),
    )
    events = core.stream(request)
    assert next(events).type == "started"

    with pytest.raises(LoadConflictError) as caught:
        core.load(
            artifact,
            adapter="mock",
            load_options=LoadOptions(model_context_size=8192),
            consumer_id="consumer",
        )
    assert caught.value.details["active_request_ids"] == [request.request_id]
    assert core.cancel(request.request_id)["cancelled"] is True
    terminal = list(events)[-1]
    assert terminal.type == "error" and terminal.error["code"] == "cancelled"
    assert core.health()["lifecycle_state"] == "LOADED"


def test_engine_build_identity_change_blocks_load_reuse(tmp_path) -> None:
    adapter = V3LoadAdapter()
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"mock": adapter})
    artifact = _fixture(tmp_path)
    first = core.load(artifact, adapter="mock", load_options=LoadOptions(), consumer_id="consumer")
    adapter.build_version = "2"

    with pytest.raises(LoadConflictError) as caught:
        core.load(artifact, adapter="mock", load_options=LoadOptions(), consumer_id="consumer")
    assert caught.value.details["loaded_load_identity_fingerprint"] != caught.value.details[
        "requested_load_identity_fingerprint"
    ]
    assert first["reused"] is False
    assert adapter.load_count == 1


def test_adapter_must_reject_unimplemented_non_default_load_options(tmp_path) -> None:
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"mock": MockAdapter()})
    with pytest.raises(UnsupportedRuntimeOptionError):
        core.load(_fixture(tmp_path), adapter="mock", load_options=LoadOptions(threads=4))
