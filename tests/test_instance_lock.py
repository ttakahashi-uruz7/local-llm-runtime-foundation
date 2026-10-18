from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from runtime_foundation import service
from runtime_foundation.instance_lock import (
    DEFAULT_LOCK_PATH,
    LOCK_PATH_ENV,
    InstanceAlreadyRunningError,
    acquire_instance_lock,
    lock_path_from_environment,
)

_HOLD_LOCK = r"""
import sys, time
from pathlib import Path
from runtime_foundation.instance_lock import acquire_instance_lock
handle = acquire_instance_lock(Path(sys.argv[1]), description="pid=holder url=http://127.0.0.1:8765")
print("locked", flush=True)
sys.stdin.readline()
"""


def _hold_in_subprocess(path: Path) -> subprocess.Popen[str]:
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLD_LOCK, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "locked"
    return holder


def _release(holder: subprocess.Popen[str]) -> None:
    assert holder.stdin is not None
    holder.stdin.write("\n")
    holder.stdin.flush()
    holder.wait(timeout=10)


def test_second_process_cannot_take_the_lock(tmp_path: Path) -> None:
    path = tmp_path / "locks" / "service.lock"
    holder = _hold_in_subprocess(path)
    try:
        with pytest.raises(InstanceAlreadyRunningError) as caught:
            acquire_instance_lock(path, description="pid=second")
        if os.name != "nt":
            assert "url=http://127.0.0.1:8765" in str(caught.value)
    finally:
        _release(holder)


def test_lock_is_released_when_the_holder_exits(tmp_path: Path) -> None:
    path = tmp_path / "service.lock"
    _release(_hold_in_subprocess(path))

    handle = acquire_instance_lock(path, description="pid=next")
    try:
        assert path.read_text(encoding="utf-8") == "pid=next"
    finally:
        handle.close()


def test_lock_path_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(LOCK_PATH_ENV, raising=False)
    assert lock_path_from_environment() == DEFAULT_LOCK_PATH
    monkeypatch.setenv(LOCK_PATH_ENV, str(tmp_path / "custom.lock"))
    assert lock_path_from_environment() == tmp_path / "custom.lock"
    monkeypatch.setenv(LOCK_PATH_ENV, "off")
    assert lock_path_from_environment() is None


def test_main_exits_without_starting_a_server_when_another_instance_holds_the_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "service.lock"
    monkeypatch.setenv(LOCK_PATH_ENV, str(path))
    started: list[object] = []
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: started.append(args))
    holder = _hold_in_subprocess(path)
    try:
        with pytest.raises(SystemExit, match="already holds"):
            service.main()
    finally:
        _release(holder)
    assert started == []


def test_main_holds_the_lock_while_serving(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "service.lock"
    monkeypatch.setenv(LOCK_PATH_ENV, str(path))
    monkeypatch.setenv("RUNTIME_FOUNDATION_PORT", "8799")
    observed: list[str] = []
    import uvicorn

    def fake_run(*args, **kwargs):
        with pytest.raises(InstanceAlreadyRunningError):
            acquire_instance_lock(path, description="pid=second")
        observed.append(path.read_text(encoding="utf-8"))

    monkeypatch.setattr(uvicorn, "run", fake_run)
    service.main()

    assert observed and observed[0].endswith("url=http://127.0.0.1:8799")
    acquire_instance_lock(path, description="after").close()
