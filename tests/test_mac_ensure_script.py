from __future__ import annotations

import os
import re
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mac" / "ensure-foundation.sh"

pytestmark = pytest.mark.skipif(
    shutil.which("zsh") is None or not (ROOT / ".venv" / "bin" / "python").is_file(),
    reason="needs zsh and the repository-local POSIX .venv",
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_ensure_starts_once_then_reuses_running_service(tmp_path: Path) -> None:
    port = _free_port()
    lock_path = tmp_path / "service.lock"
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "RUNTIME_FOUNDATION_PORT": str(port),
        "RUNTIME_FOUNDATION_INSTANCE_LOCK": str(lock_path),
    }
    try:
        first = subprocess.run([str(SCRIPT)], env=env, capture_output=True, text=True, timeout=90)
        assert first.returncode == 0, first.stderr
        assert "Foundation started" in first.stdout
        assert httpx.get(f"http://127.0.0.1:{port}/health", timeout=5).status_code == 200

        second = subprocess.run([str(SCRIPT)], env=env, capture_output=True, text=True, timeout=30)
        assert second.returncode == 0, second.stderr
        assert "already running" in second.stdout
        assert (tmp_path / "Library" / "Logs" / "local-llm-runtime-foundation" / "service.log").is_file()
    finally:
        holder = lock_path.read_text(encoding="utf-8") if lock_path.is_file() else ""
        match = re.search(r"pid=(\d+)", holder)
        if match:
            os.kill(int(match.group(1)), signal.SIGTERM)
            time.sleep(0.5)
