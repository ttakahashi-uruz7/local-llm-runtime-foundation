from __future__ import annotations

from pathlib import Path


def test_foundation_source_has_no_benchmark_or_studio_authority_imports() -> None:
    source = "\n".join(path.read_text(encoding="utf-8").lower() for path in Path("runtime_foundation").rglob("*.py"))
    for forbidden in ("app.registry", "app.storage", "production_eligible", "optimizer", "benchmark scoring", "cloud fallback"):
        assert forbidden not in source
