"""A reserved stream must release its generation slot even if nobody reads it."""

from __future__ import annotations

import json
import socket
import threading
import time

import anyio
import httpx
import pytest
import uvicorn

from runtime_foundation import ModelArtifactBinding, ThinkingIntent, ThinkingMode
from runtime_foundation.adapters.mock import MockAdapter
from runtime_foundation.contracts_v3 import ExecutionConstraints, GenerationOptions, GenerationRequestV3, LoadOptions
from runtime_foundation.core import LifecycleState, RuntimeCore
from runtime_foundation.host import HostProfile
from runtime_foundation.service import create_app


def _loaded_core(tmp_path) -> RuntimeCore:
    artifact_path = tmp_path / "stream-release-model.bin"
    artifact_path.write_bytes(b"stream-release-fixture")
    artifact = ModelArtifactBinding("stream-release-model", str(artifact_path), "bin")
    core = RuntimeCore(host_profile=HostProfile.mock_windows(), adapters={"mock": MockAdapter()})
    core.load(artifact, adapter="mock", load_options=LoadOptions(), consumer_id="stream-release-test")
    return core


def _request(request_id: str, *, max_tokens: int = 4, timeout_ms: int | None = None) -> GenerationRequestV3:
    return GenerationRequestV3(
        model_artifact_id="stream-release-model",
        messages=[{"role": "user", "content": "hello " * 40}],
        generation_options=GenerationOptions(
            max_tokens=max_tokens,
            temperature=0,
            thinking_intent=ThinkingIntent(mode=ThinkingMode.OFF),
        ),
        execution_constraints=ExecutionConstraints(timeout_ms=timeout_ms),
        request_id=request_id,
        consumer_id="stream-release-test",
    )


def _assert_idle(core: RuntimeCore, within_seconds: float = 5.0) -> None:
    deadline = time.monotonic() + within_seconds
    while time.monotonic() < deadline and (core._active or core._state != LifecycleState.LOADED):
        time.sleep(0.05)
    assert core._active == {}
    assert core._state == LifecycleState.LOADED


def test_cancel_releases_a_stream_nobody_started_reading(tmp_path) -> None:
    core = _loaded_core(tmp_path)
    core.stream_v3(_request("unread"))
    assert list(core._active) == ["unread"]

    cancelled = core.cancel("unread")

    assert cancelled["state"] == "released_unread_stream"
    _assert_idle(core)


def test_timeout_releases_a_stream_nobody_started_reading(tmp_path) -> None:
    core = _loaded_core(tmp_path)
    core.stream_v3(_request("unread-timeout", timeout_ms=200))

    _assert_idle(core)


def test_closing_an_unread_or_partly_read_stream_releases_the_slot(tmp_path) -> None:
    core = _loaded_core(tmp_path)
    core.stream_v3(_request("never-read")).close()
    _assert_idle(core)

    stream = core.stream_v3(_request("partly-read", max_tokens=64))
    assert next(stream).type == "started"
    stream.close()
    _assert_idle(core)


def test_a_reading_consumer_still_gets_the_cancelled_terminal_event(tmp_path) -> None:
    core = _loaded_core(tmp_path)
    stream = core.stream_v3(_request("reading", max_tokens=64))
    assert next(stream).type == "started"

    core.cancel("reading")
    events = list(stream)

    assert events[-1].type == "error"
    assert events[-1].error["code"] == "cancelled"
    _assert_idle(core)


@pytest.fixture
def served_core(tmp_path):
    core = _loaded_core(tmp_path)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(core), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    yield core, f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)


def test_a_response_whose_client_already_left_releases_the_slot(tmp_path) -> None:
    # The state a client leaves behind when it gives up while queued: by the
    # time the slot frees and the response starts, the server only receives
    # http.disconnect, and Starlette never iterates the event stream.
    core = _loaded_core(tmp_path)
    app = create_app(core)
    body = json.dumps(_request("left-while-queued").to_dict()).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v3/generate/stream",
        "raw_path": b"/v3/generate/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        "client": ("127.0.0.1", 50000),
        "server": ("127.0.0.1", 8765),
    }
    messages = [{"type": "http.request", "body": body, "more_body": False}]

    async def receive():
        if messages:
            return messages.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):
        return None

    anyio.run(app, scope, receive, send)

    _assert_idle(core, within_seconds=1.5)


def test_a_client_that_disconnects_mid_stream_releases_the_slot(served_core) -> None:
    core, base_url = served_core
    body = _request("disconnects", max_tokens=400).to_dict()
    with (
        httpx.Client(timeout=10) as client,
        client.stream("POST", f"{base_url}/v3/generate/stream", json=body) as response,
    ):
        for index, _ in enumerate(response.iter_lines()):
            if index >= 1:
                break

    _assert_idle(core)
