"""Read-only host observation for local runtime diagnostics."""

from __future__ import annotations

import ctypes
import importlib
import importlib.metadata
import importlib.util
import os
import platform as platform_module
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any

from .contracts import CONTRACT_VERSION, utc_now


def _command_output(*args: str, timeout: float = 2.0) -> str | None:
    """Run a fixed diagnostic command without invoking a shell."""

    try:
        completed = subprocess.run(
            list(args),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    value = completed.stdout.strip()
    return value or None


def _size_bytes(value: str) -> int | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([kmgt](?:i?b)?|b)?", value, flags=re.IGNORECASE)
    if not match:
        return None
    number = float(match.group(1))
    unit = (match.group(2) or "").lower()
    multipliers = {
        "": 1,
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "ki": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "mi": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "gi": 1024**3,
        "gib": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
        "ti": 1024**4,
        "tib": 1024**4,
    }
    return int(number * multipliers.get(unit, 1))


def _normalize_platform(value: str) -> str:
    value = value.lower()
    return {"darwin": "darwin", "windows": "windows", "linux": "linux"}.get(value, value or "unknown")


def _normalize_architecture(value: str) -> str:
    value = value.lower()
    if value in {"arm64", "aarch64"}:
        return "arm64"
    if value in {"amd64", "x86_64", "x64"}:
        return "x86_64"
    return value or "unknown"


def _darwin_sysctl(name: str) -> str | None:
    return _command_output("sysctl", "-n", name)


def _physical_memory_bytes(system: str) -> int | None:
    if system == "darwin":
        value = _darwin_sysctl("hw.memsize")
        try:
            return int(value) if value else None
        except ValueError:
            return None
    if system == "windows":
        try:
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
        except (AttributeError, OSError, TypeError):
            return None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if isinstance(pages, int) and isinstance(page_size, int):
            return pages * page_size
    except (AttributeError, OSError, ValueError):
        return None
    return None


def process_memory_bytes() -> int | None:
    """Return a best-effort process footprint without making it a policy result."""

    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if _normalize_platform(platform_module.system()) != "darwin":
            value *= 1024
        return value
    except (ImportError, OSError, ValueError):
        pass
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError, AttributeError, ValueError):
        return None


def _mlx_status(system: str, architecture: str) -> tuple[bool, bool, str | None, str | None, str | None]:
    if system != "darwin" or architecture != "arm64":
        return False, False, None, None, "MLX requires Apple Silicon macOS"
    try:
        if importlib.util.find_spec("mlx") is None or importlib.util.find_spec("mlx_lm") is None:
            return False, False, None, None, "mlx and mlx-lm are not both installed"
        mlx = importlib.import_module("mlx.core")
        importlib.import_module("mlx_lm")
        device = str(mlx.default_device()).lower()
        metal = "gpu" in device or "metal" in device
    except (ImportError, ModuleNotFoundError, AttributeError, RuntimeError, ValueError) as exc:
        return False, False, None, None, f"MLX import failed: {exc}"
    try:
        mlx_version = importlib.metadata.version("mlx")
    except importlib.metadata.PackageNotFoundError:
        mlx_version = None
    try:
        mlx_lm_version = importlib.metadata.version("mlx-lm")
    except importlib.metadata.PackageNotFoundError:
        mlx_lm_version = None
    return True, metal, mlx_version, mlx_lm_version, None


def _darwin_swap() -> tuple[int | None, int | None]:
    value = _darwin_sysctl("vm.swapusage")
    if not value:
        return None, None
    total = re.search(r"total\s*=\s*([^\s]+)", value, flags=re.IGNORECASE)
    used = re.search(r"used\s*=\s*([^\s]+)", value, flags=re.IGNORECASE)
    return (
        _size_bytes(used.group(1)) if used else None,
        _size_bytes(total.group(1)) if total else None,
    )


def _darwin_memory_pressure() -> str | None:
    value = _command_output("memory_pressure", "-Q")
    if not value:
        return None
    match = re.search(r"free\s+percentage\s*:\s*([0-9]+(?:\.[0-9]+)?)", value, flags=re.IGNORECASE)
    if not match:
        return "unknown"
    free_percent = float(match.group(1))
    if free_percent <= 5:
        return "critical"
    if free_percent <= 15:
        return "warning"
    return "normal"


@dataclass(frozen=True)
class HostProfile:
    """A point-in-time host observation.  It intentionally has no target label."""

    platform: str
    architecture: str
    os_version: str
    os_build: str | None
    hardware_model: str | None
    cpu_count: int | None
    physical_memory_bytes: int | None
    unified_memory_bytes: int | None
    memory_pressure: str | None
    swap_used_bytes: int | None
    swap_total_bytes: int | None
    mlx_available: bool
    metal_available: bool
    mlx_version: str | None
    mlx_lm_version: str | None
    captured_at: str

    @classmethod
    def detect(cls) -> "HostProfile":
        system = _normalize_platform(platform_module.system())
        architecture = _normalize_architecture(platform_module.machine())
        is_darwin = system == "darwin"
        physical = _physical_memory_bytes(system)
        swap_used, swap_total = _darwin_swap() if is_darwin else (None, None)
        mlx_available, metal_available, mlx_version, mlx_lm_version, _ = _mlx_status(system, architecture)
        return cls(
            platform=system,
            architecture=architecture,
            os_version=platform_module.release() or "unknown",
            os_build=(
                _command_output("sw_vers", "-buildVersion") if is_darwin else platform_module.version() or None
            ),
            hardware_model=_darwin_sysctl("hw.model") if is_darwin else None,
            cpu_count=os.cpu_count(),
            physical_memory_bytes=physical,
            unified_memory_bytes=physical if is_darwin else None,
            memory_pressure=_darwin_memory_pressure() if is_darwin else None,
            swap_used_bytes=swap_used,
            swap_total_bytes=swap_total,
            mlx_available=mlx_available,
            metal_available=metal_available,
            mlx_version=mlx_version,
            mlx_lm_version=mlx_lm_version,
            captured_at=utc_now(),
        )

    capture = detect

    @classmethod
    def mock_windows(cls) -> "HostProfile":
        return cls(
            platform="windows",
            architecture="x86_64",
            os_version="mock-development",
            os_build=None,
            hardware_model=None,
            cpu_count=1,
            physical_memory_bytes=None,
            unified_memory_bytes=None,
            memory_pressure=None,
            swap_used_bytes=None,
            swap_total_bytes=None,
            mlx_available=False,
            metal_available=False,
            mlx_version=None,
            mlx_lm_version=None,
            captured_at=utc_now(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"contract_version": CONTRACT_VERSION, **asdict(self)}

    def runtime_snapshot(self) -> dict[str, Any]:
        """Refresh volatile host fields while preserving this profile's identity."""

        current = type(self).detect()
        return current.to_dict()


def runtime_snapshot() -> dict[str, Any]:
    """Capture host and process observations for raw execution metrics."""

    profile = HostProfile.detect()
    payload = profile.to_dict()
    payload["process_memory_bytes"] = process_memory_bytes()
    return payload
