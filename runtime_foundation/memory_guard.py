"""Host memory admission check before a model load.

Two kernel watchdog panics on a 64 GB Apple Silicon Mac (2026-10-09) happened
with almost all physical memory wired: another local runtime kept a ~28 GB
model in Metal memory while Foundation loaded a 33-40 GB model. Foundation
holds one model per process, but it cannot see models held by other processes
except through the host's wired memory. This guard reads wired memory right
before a load and refuses the load when wired memory plus the artifact size
would pass a share of physical memory.

The check is macOS-only. On other hosts, or when the observation fails, it
reports ``skipped`` and never blocks.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import HostMemoryInsufficientError

LIMIT_RATIO_ENV = "RUNTIME_FOUNDATION_LOAD_MEMORY_LIMIT_RATIO"
# Both panics occurred at ~96% wired; 85% leaves room for KV cache, Metal
# scratch buffers and the rest of macOS.
DEFAULT_LIMIT_RATIO = 0.85
# The Mock engine allocates no model memory.
DEFAULT_EXEMPT_ENGINES = ("mock",)

WiredObservation = dict[str, int]


def observe_wired_memory() -> WiredObservation | None:
    """Return wired and physical memory in bytes on macOS, otherwise None."""

    if platform.system() != "Darwin":
        return None
    try:
        vm_stat = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10, check=True).stdout
        total = int(
            subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=10, check=True)
            .stdout.strip()
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    page_size = re.search(r"page size of (\d+) bytes", vm_stat)
    wired = re.search(r"Pages wired down:\s+(\d+)", vm_stat)
    if page_size is None or wired is None:
        return None
    return {"wired_bytes": int(wired.group(1)) * int(page_size.group(1)), "physical_memory_bytes": total}


def artifact_size_bytes(paths: Iterable[str | Path | None]) -> int:
    """Sum the on-disk size of artifact files or directories."""

    total = 0
    for value in paths:
        if value is None:
            continue
        path = Path(value)
        if path.is_file():
            total += path.stat().st_size
        elif path.is_dir():
            total += sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    return total


def _limit_ratio_from_environment() -> float | None:
    raw = os.environ.get(LIMIT_RATIO_ENV, "").strip()
    if not raw:
        return DEFAULT_LIMIT_RATIO
    if raw.lower() in {"off", "disabled", "none"}:
        return None
    try:
        ratio = float(raw)
    except ValueError as exc:
        raise ValueError(f"{LIMIT_RATIO_ENV} must be a number in (0, 1] or 'off'") from exc
    if not 0 < ratio <= 1:
        raise ValueError(f"{LIMIT_RATIO_ENV} must be a number in (0, 1] or 'off'")
    return ratio


@dataclass(frozen=True)
class LoadMemoryGuard:
    """Refuse a load whose artifact would push wired memory past ``limit_ratio``."""

    limit_ratio: float | None = DEFAULT_LIMIT_RATIO
    observe: Callable[[], WiredObservation | None] = observe_wired_memory
    exempt_engines: tuple[str, ...] = DEFAULT_EXEMPT_ENGINES

    @classmethod
    def from_environment(cls) -> LoadMemoryGuard:
        return cls(limit_ratio=_limit_ratio_from_environment())

    def check(self, *, engine: str, artifact_paths: Iterable[str | Path | None]) -> dict[str, Any]:
        if self.limit_ratio is None:
            return {"status": "skipped", "reason": "disabled"}
        if engine in self.exempt_engines:
            return {"status": "skipped", "reason": "engine_exempt", "engine": engine}
        observed = self.observe()
        if observed is None:
            return {"status": "skipped", "reason": "unobservable"}
        model_bytes = artifact_size_bytes(artifact_paths)
        limit = int(observed["physical_memory_bytes"] * self.limit_ratio)
        details = {
            "wired_bytes": observed["wired_bytes"],
            "model_bytes": model_bytes,
            "physical_memory_bytes": observed["physical_memory_bytes"],
            "wired_limit_bytes": limit,
            "wired_limit_ratio": self.limit_ratio,
        }
        if observed["wired_bytes"] + model_bytes > limit:
            gib = 1 << 30
            raise HostMemoryInsufficientError(
                f"loading this model ({model_bytes / gib:.1f} GiB) on top of "
                f"{observed['wired_bytes'] / gib:.1f} GiB already wired would exceed "
                f"{limit / gib:.1f} GiB; unload models held by other local runtimes "
                "(for example `~/.lmstudio/bin/lms unload --all`) and retry",
                details=details,
            )
        return {"status": "passed", **details}
