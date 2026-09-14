"""Network boundary helpers for the unauthenticated local service."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit


def validate_loopback_host(value: str | None) -> str:
    """Allow only loopback bind addresses for the unauthenticated v1 service."""

    host = (value or "127.0.0.1").strip()
    if host.lower() == "localhost":
        return host
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError(
            "loopback host must be localhost or an IP address whose loopback flag is true"
        ) from exc
    if not address.is_loopback:
        raise ValueError("host must be a loopback address; remote/LAN access is not supported in Foundation v1")
    return host


def validate_loopback_url(value: str) -> str:
    """Validate a Foundation service URL before a client can connect to it."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError("base_url must be a non-empty HTTP(S) loopback URL")
    base_url = value.strip().rstrip("/")
    try:
        parsed = urlsplit(base_url)
        host = parsed.hostname
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("base_url must be a valid HTTP(S) loopback URL") from exc
    if parsed.scheme.lower() not in {"http", "https"} or host is None or parsed.username or parsed.password:
        raise ValueError("base_url must be an HTTP(S) loopback URL without credentials")
    validate_loopback_host(host)
    return base_url
