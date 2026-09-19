"""Deterministic runtime build-identity observation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .contracts_v2 import BuildIdentityV1

FOUNDATION_BUILD_IDENTITY_KIND = "foundation-python-package-tree-v1"


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def foundation_package_tree_components(root: str | Path) -> dict[str, Any]:
    """Return path-independent, deterministic components for a Foundation tree."""

    package_root = Path(root)
    if not package_root.is_dir():
        raise ValueError("Foundation package root must be a directory")
    files: list[dict[str, Any]] = []
    for path in sorted(
        package_root.rglob("*.py"), key=lambda candidate: candidate.relative_to(package_root).as_posix()
    ):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(package_root).as_posix()
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": _file_digest(path),
            }
        )
    return {"package": "runtime_foundation", "files": files}


def build_identity_from_package_tree(root: str | Path) -> BuildIdentityV1:
    """Build a deterministic identity for a Foundation Python package tree."""

    return BuildIdentityV1.from_components(
        kind=FOUNDATION_BUILD_IDENTITY_KIND,
        components=foundation_package_tree_components(root),
    )


def observe_foundation_build_identity() -> BuildIdentityV1 | None:
    """Observe the installed/source Foundation identity, or remain unknown."""

    try:
        return build_identity_from_package_tree(Path(__file__).resolve().parent)
    except (OSError, ValueError, TypeError):
        return None


__all__ = [
    "FOUNDATION_BUILD_IDENTITY_KIND",
    "build_identity_from_package_tree",
    "foundation_package_tree_components",
    "observe_foundation_build_identity",
]
