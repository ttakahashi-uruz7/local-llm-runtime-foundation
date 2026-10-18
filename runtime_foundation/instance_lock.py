"""Single-instance lock for the local Foundation service.

Each Foundation process holds its own model. A second service on another port
would let two consumers load two models side by side, which is the memory
exhaustion the one-model lifecycle exists to prevent. ``main`` takes an
exclusive OS file lock for the life of the process; the OS releases it when
the process exits, so a crash leaves no stale lock behind.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import IO

LOCK_PATH_ENV = "RUNTIME_FOUNDATION_INSTANCE_LOCK"
DEFAULT_LOCK_PATH = Path.home() / ".cache" / "local-llm-runtime-foundation" / "service.lock"


class InstanceAlreadyRunningError(RuntimeError):
    def __init__(self, path: Path, holder: str) -> None:
        self.path = path
        self.holder = holder
        super().__init__(
            f"another Local LLM Runtime Foundation service already holds {path} ({holder or 'holder unknown'}); "
            "use that service instead of starting a second one"
        )


def lock_path_from_environment() -> Path | None:
    raw = os.environ.get(LOCK_PATH_ENV, "").strip()
    if raw.lower() in {"off", "disabled", "none"}:
        return None
    return Path(raw).expanduser() if raw else DEFAULT_LOCK_PATH


def _try_lock(handle: IO[str]) -> bool:
    # msvcrt locks bytes from the current position; pin it to byte 0 so every
    # process contends for the same byte.
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def acquire_instance_lock(path: Path, *, description: str) -> IO[str]:
    """Hold ``path`` exclusively and record ``description``; keep the handle open."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+", encoding="utf-8")  # noqa: SIM115 - held for the process lifetime
    if not _try_lock(handle):
        try:
            handle.seek(0)
            holder = handle.read().strip()
        except OSError:  # Windows locks are mandatory, so the holder may be unreadable.
            holder = ""
        handle.close()
        raise InstanceAlreadyRunningError(path, holder)
    handle.seek(0)
    handle.truncate()
    handle.write(description)
    handle.flush()
    return handle
