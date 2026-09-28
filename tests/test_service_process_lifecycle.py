from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from runtime_foundation import (
    ADAPTER_LINEAGE_METADATA_KEY,
    AdapterLineageV1,
    AdapterTargetBaseV1,
    ArtifactBindingV2,
    ArtifactLocator,
    ContentIdentity,
    ExecutionInputV1,
    GenerationRequestV2,
    LocalRuntimeClient,
    ThinkingIntent,
    ThinkingMode,
)


_TEST_SERVICE = r"""
import os
import uvicorn
from runtime_foundation.service import create_app

app = create_app()
server_slot = {}

@app.post("/__test/shutdown")
def shutdown_test_process():
    server_slot["server"].should_exit = True
    return {"shutting_down": True}

server = uvicorn.Server(uvicorn.Config(
    app,
    host=os.environ["RUNTIME_FOUNDATION_HOST"],
    port=int(os.environ["RUNTIME_FOUNDATION_PORT"]),
    log_level="error",
    access_log=False,
))
server_slot["server"] = server
server.run()
"""


def _direct_fixture(tmp_path: Path) -> ExecutionInputV1:
    base_path = tmp_path / "base.safetensors"
    base_path.write_bytes(b"service restart base fixture")
    base = ArtifactBindingV2(
        artifact_id="service-restart-base",
        locator=ArtifactLocator("filesystem", str(base_path)),
        content_identity=ContentIdentity.from_file(base_path),
        format="mlx",
        quantization="8-bit-g64",
        revision="service-restart-base-r1",
        metadata={
            "model_identity": "fixture/service-restart-model",
            "tokenizer_identity": "fixture/service-restart-tokenizer",
            "target_module_shapes": {"self_attn.q_proj": {"in_features": 4, "out_features": 6}},
        },
    )

    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    (adapter_path / "adapter_config.json").write_text(
        '{"fine_tune_type":"lora","lora_parameters":{"rank":2,"keys":["self_attn.q_proj"]}}',
        encoding="utf-8",
    )
    (adapter_path / "adapters.safetensors").write_bytes(b"service restart adapter fixture")
    lineage = AdapterLineageV1(
        target_base=AdapterTargetBaseV1(
            artifact_id=base.artifact_id,
            content_identity=base.content_identity,
            revision=base.revision,
            format=base.format,
            quantization=base.quantization,
            model_identity=base.metadata["model_identity"],
            tokenizer_identity=base.metadata["tokenizer_identity"],
        ),
        rank=2,
        target_modules=("self_attn.q_proj",),
        tensor_shapes={"self_attn.q_proj": {"lora_a": [4, 2], "lora_b": [2, 6]}},
    )
    adapter = ArtifactBindingV2(
        artifact_id="service-restart-adapter",
        locator=ArtifactLocator("filesystem", str(adapter_path)),
        content_identity=ContentIdentity.from_file(adapter_path),
        format="mlx-lora",
        revision="service-restart-adapter-r1",
        metadata={ADAPTER_LINEAGE_METADATA_KEY: lineage.to_dict()},
    )
    return ExecutionInputV1(base=base, adapters=(adapter,))


def _request(execution_input: ExecutionInputV1, lease_id: str) -> GenerationRequestV2:
    return GenerationRequestV2(
        model_artifact_id=execution_input.base.artifact_id,
        messages=[{"role": "user", "content": "service restart fixture"}],
        consumer_id="service-restart-test",
        lease_id=lease_id,
        execution_input=execution_input,
        thinking_intent=ThinkingIntent(mode=ThinkingMode.OFF),
    )


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _start_test_service(port: int) -> subprocess.Popen[str]:
    environment = dict(os.environ)
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN"):
        environment.pop(key, None)
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["RUNTIME_FOUNDATION_HOST"] = "127.0.0.1"
    environment["RUNTIME_FOUNDATION_PORT"] = str(port)
    environment["PYTHONUNBUFFERED"] = "1"
    repository = Path(__file__).resolve().parents[1]
    return subprocess.Popen(
        [sys.executable, "-c", _TEST_SERVICE],
        cwd=repository,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_ready(process: subprocess.Popen[str], base_url: str) -> None:
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output, _ = process.communicate()
            raise AssertionError(f"Foundation test service exited before readiness:\n{output}")
        try:
            response = httpx.get(f"{base_url}/health", timeout=0.5)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.1)
    raise AssertionError("Foundation test service did not become ready")


def _stop_test_service(process: subprocess.Popen[str], base_url: str) -> str:
    shutdown_failure: str | None = None
    if process.poll() is None:
        try:
            response = httpx.post(f"{base_url}/__test/shutdown", timeout=2)
            if response.status_code != 200:
                shutdown_failure = f"shutdown endpoint returned HTTP {response.status_code}"
        except httpx.HTTPError as exc:
            shutdown_failure = f"shutdown endpoint was unavailable: {exc}"
        if shutdown_failure is not None and process.poll() is None:
            # This fallback can affect only the child process created by this test.
            process.terminate()
    try:
        output, _ = process.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        process.terminate()
        output, _ = process.communicate(timeout=5)
        raise AssertionError(f"Foundation test service did not shut down cleanly:\n{output}")
    assert process.returncode == 0, output
    assert shutdown_failure is None, shutdown_failure
    return output


def _load_generate_unload(client: LocalRuntimeClient, execution_input: ExecutionInputV1) -> str:
    loaded = client.load(execution_input=execution_input, engine="mock", consumer_id="service-restart-test")
    result = client.generate(_request(execution_input, loaded["lease_id"]))
    trace = result["generation_v2"]["trace"]
    assert trace["execution_input"]["execution_input_fingerprint"] == execution_input.fingerprint
    assert trace["execution_binding"]["execution_input"]["execution_input_fingerprint"] == execution_input.fingerprint
    assert client.unload(
        execution_input.base.artifact_id,
        consumer_id="service-restart-test",
        lease_id=loaded["lease_id"],
    )["unloaded"]
    return str(trace["execution_id"])


def test_direct_execution_recovers_across_independent_service_restart(tmp_path: Path) -> None:
    execution_input = _direct_fixture(tmp_path)
    port = _free_loopback_port()
    base_url = f"http://127.0.0.1:{port}"

    first_process = _start_test_service(port)
    try:
        _wait_ready(first_process, base_url)
        with LocalRuntimeClient(base_url, timeout=20) as client:
            health = client.health()
            assert health["status"] == "ready"
            assert health["loaded_artifact_id"] is None
            first_execution_id = _load_generate_unload(client, execution_input)
    finally:
        _stop_test_service(first_process, base_url)

    second_process = _start_test_service(port)
    try:
        _wait_ready(second_process, base_url)
        with LocalRuntimeClient(base_url, timeout=20) as client:
            health = client.health()
            assert health["status"] == "ready"
            assert health["loaded_artifact_id"] is None
            assert health.get("loaded_execution_input_fingerprint") is None
            second_execution_id = _load_generate_unload(client, execution_input)
            assert first_execution_id != second_execution_id
    finally:
        _stop_test_service(second_process, base_url)
