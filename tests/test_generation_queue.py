from __future__ import annotations

import threading
import time

import pytest

from runtime_foundation import (
    GenerationOptions,
    GenerationRequestV3,
    HostProfile,
    LoadOptions,
    ModelArtifactBinding,
    RuntimeCore,
    SchedulingOptionsV3,
    ThinkingIntent,
    ThinkingMode,
)
from runtime_foundation.adapters.mock import MockAdapter
from runtime_foundation.errors import RequestCancelledError, RuntimeBusyError


class GatedMockAdapter(MockAdapter):
    """Mock whose generations block until the test opens their gate."""

    def __init__(self) -> None:
        super().__init__()
        self.gates: dict[str, threading.Event] = {}
        self.started: list[str] = []

    def generate(self, request, cancel_event):
        self.started.append(request.request_id)
        gate = self.gates.get(request.request_id)
        if gate is not None:
            assert gate.wait(5), "test gate was never opened"
        return super().generate(request, cancel_event)


@pytest.fixture
def queue_core(tmp_path):
    artifact_path = tmp_path / "queue-model.bin"
    artifact_path.write_bytes(b"test-only-runtime-fixture")
    artifact = ModelArtifactBinding("queue-model", str(artifact_path), "bin")
    adapter = GatedMockAdapter()
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"mock": adapter})
    core.load(artifact, adapter="mock", load_options=LoadOptions(), consumer_id="novel")
    core.load(artifact, adapter="mock", load_options=LoadOptions(), consumer_id="rsvp")
    return core, adapter, artifact


def _request(artifact, request_id, *, consumer_id="novel", priority="interactive", queue_timeout_ms=5_000):
    return GenerationRequestV3(
        model_artifact_id=artifact.artifact_id,
        messages=[{"role": "user", "content": "hello there"}],
        generation_options=GenerationOptions(max_tokens=2, thinking_intent=ThinkingIntent(mode=ThinkingMode.OFF)),
        request_id=request_id,
        consumer_id=consumer_id,
        scheduling=SchedulingOptionsV3(priority=priority, queue_timeout_ms=queue_timeout_ms),
    )


def _run(core, request, results):
    def target():
        try:
            results[request.request_id] = core.generate_v3(request).to_dict()
        except Exception as exc:  # noqa: BLE001 - the test inspects the raised error
            results[request.request_id] = exc

    thread = threading.Thread(target=target)
    thread.start()
    return thread


def _wait_until(predicate):
    deadline = time.monotonic() + 5
    while not predicate():
        assert time.monotonic() < deadline, "condition was not reached"
        time.sleep(0.01)


def test_second_request_waits_for_the_slot_instead_of_failing(queue_core) -> None:
    core, adapter, artifact = queue_core
    adapter.gates["first"] = threading.Event()
    results: dict[str, object] = {}
    first = _run(core, _request(artifact, "first", consumer_id="rsvp", priority="background"), results)
    _wait_until(lambda: adapter.started == ["first"])
    second = _run(core, _request(artifact, "second"), results)
    _wait_until(lambda: core.health_v3()["generation_queue"]["queued_request_ids"] == ["second"])

    assert core.health_v3()["active_request_ids"] == ["first"]
    adapter.gates["first"].set()
    first.join(5)
    second.join(5)

    assert adapter.started == ["first", "second"]
    assert isinstance(results["first"], dict) and results["first"]["text"]
    assert isinstance(results["second"], dict) and results["second"]["text"]
    assert core.health_v3()["generation_queue"]["queued_request_ids"] == []


def test_interactive_requests_are_admitted_before_waiting_background_requests(queue_core) -> None:
    core, adapter, artifact = queue_core
    adapter.gates["running"] = threading.Event()
    results: dict[str, object] = {}
    running = _run(core, _request(artifact, "running", consumer_id="rsvp", priority="background"), results)
    _wait_until(lambda: adapter.started == ["running"])
    background = _run(core, _request(artifact, "background", consumer_id="rsvp", priority="background"), results)
    _wait_until(lambda: core.health_v3()["generation_queue"]["queued_request_ids"] == ["background"])
    interactive = _run(core, _request(artifact, "interactive"), results)
    _wait_until(
        lambda: core.health_v3()["generation_queue"]["queued_request_ids"] == ["interactive", "background"]
    )

    adapter.gates["running"].set()
    for thread in (running, background, interactive):
        thread.join(5)

    assert adapter.started == ["running", "interactive", "background"]
    assert all(isinstance(results[name], dict) and results[name]["text"] for name in ("running", "background", "interactive"))


def test_queue_timeout_fails_before_generation_starts(queue_core) -> None:
    core, adapter, artifact = queue_core
    adapter.gates["running"] = threading.Event()
    results: dict[str, object] = {}
    running = _run(core, _request(artifact, "running"), results)
    _wait_until(lambda: adapter.started == ["running"])

    with pytest.raises(RuntimeBusyError) as caught:
        core.generate_v3(_request(artifact, "late", queue_timeout_ms=50))

    adapter.gates["running"].set()
    running.join(5)
    assert caught.value.details["generation_started"] is False
    assert caught.value.details["queue_timeout_ms"] == 50
    assert caught.value.retryable is True
    assert adapter.started == ["running"]


def test_zero_queue_timeout_keeps_the_immediate_busy_answer(queue_core) -> None:
    core, adapter, artifact = queue_core
    adapter.gates["running"] = threading.Event()
    results: dict[str, object] = {}
    running = _run(core, _request(artifact, "running"), results)
    _wait_until(lambda: adapter.started == ["running"])

    with pytest.raises(RuntimeBusyError):
        core.generate_v3(_request(artifact, "no-wait", queue_timeout_ms=0))

    adapter.gates["running"].set()
    running.join(5)


def test_cancel_removes_a_queued_request(queue_core) -> None:
    core, adapter, artifact = queue_core
    adapter.gates["running"] = threading.Event()
    results: dict[str, object] = {}
    running = _run(core, _request(artifact, "running"), results)
    _wait_until(lambda: adapter.started == ["running"])
    queued = _run(core, _request(artifact, "queued", consumer_id="rsvp", priority="background"), results)
    _wait_until(lambda: core.health_v3()["generation_queue"]["queued_request_ids"] == ["queued"])

    cancelled = core.cancel("queued")
    queued.join(5)
    adapter.gates["running"].set()
    running.join(5)

    assert cancelled["state"] == "removed_from_queue"
    assert isinstance(results["queued"], RequestCancelledError)
    assert results["queued"].details["generation_started"] is False
    assert adapter.started == ["running"]


def test_duplicate_request_id_is_refused_while_the_first_runs(queue_core) -> None:
    core, adapter, artifact = queue_core
    adapter.gates["same"] = threading.Event()
    results: dict[str, object] = {}
    running = _run(core, _request(artifact, "same"), results)
    _wait_until(lambda: adapter.started == ["same"])

    with pytest.raises(RuntimeBusyError, match="already running or queued"):
        core.generate_v3(_request(artifact, "same"))

    adapter.gates["same"].set()
    running.join(5)


def test_scheduling_payload_is_validated_and_optional() -> None:
    assert SchedulingOptionsV3.from_payload(None) == SchedulingOptionsV3()
    parsed = SchedulingOptionsV3.from_payload({"priority": "background", "queue_timeout_ms": 1000})
    assert parsed.to_dict()["priority"] == "background"
    with pytest.raises(ValueError):
        SchedulingOptionsV3.from_payload({"priority": "urgent"})
    with pytest.raises(ValueError):
        SchedulingOptionsV3.from_payload({"queue_timeout_ms": -1})
    with pytest.raises(ValueError):
        SchedulingOptionsV3.from_payload({"queue": "x"})
