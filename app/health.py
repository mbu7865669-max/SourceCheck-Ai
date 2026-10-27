"""
Shared health-probe helpers.

Both the web gateway and the agent service report bridge reachability, so the
(non-obvious) URL arithmetic and the never-raise probe live here instead of
being duplicated - getting the URL wrong (/v1/health 404s) makes a perfectly
healthy bridge look down.
"""

from __future__ import annotations

from typing import Any, Dict

import httpx

from app.config import settings


def bridge_health_url() -> str:
    """Map the OpenAI base URL (.../v1) to the bridge's root /health URL.

    Probing /v1/health 404s, so the trailing /v1 has to be stripped here rather
    than assumed away by callers.
    """
    base = settings.BRIDGE_ENDPOINT.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")].rstrip("/")
    return f"{base}/health"


def bridge_health() -> Dict[str, Any]:
    """Probe the bridge's liveness endpoint. Never raises - health must not fail."""
    url = bridge_health_url()

    try:
        with httpx.Client(timeout=3.0) as client:
            response = client.get(url)
            response.raise_for_status()
            body = response.json()
        return {
            "endpoint": settings.BRIDGE_ENDPOINT,
            "reachable": True,
            "model": body.get("model", settings.BRIDGE_MODEL),
            "opencode_version": body.get("opencode_version", "unknown"),
        }
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        return {
            "endpoint": settings.BRIDGE_ENDPOINT,
            "reachable": False,
            "hint": "Start it with: python bridge/opencode_bridge.py",
            "error": str(exc),
        }
