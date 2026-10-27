"""
Tests for the shared health helpers (app/health.py).

The bridge probe is shared by the web gateway and the agent service, so its
URL arithmetic lives in one place - probing /v1/health 404s and makes a
perfectly healthy bridge look down.
"""

from app.config import settings
from app.health import bridge_health, bridge_health_url


class TestBridgeHealthUrl:
    def test_strips_v1_suffix(self, monkeypatch):
        # The OpenAI base ends in /v1, but /health lives at the bridge root.
        monkeypatch.setattr(settings, "BRIDGE_ENDPOINT", "http://127.0.0.1:18383/v1")
        assert bridge_health_url() == "http://127.0.0.1:18383/health"

    def test_no_v1_suffix_left_alone(self, monkeypatch):
        monkeypatch.setattr(settings, "BRIDGE_ENDPOINT", "http://localhost:9000")
        assert bridge_health_url() == "http://localhost:9000/health"

    def test_trailing_slash_is_normalised(self, monkeypatch):
        monkeypatch.setattr(settings, "BRIDGE_ENDPOINT", "http://127.0.0.1:18383/v1/")
        assert bridge_health_url() == "http://127.0.0.1:18383/health"

    def test_v1_in_middle_of_host_is_not_stripped(self, monkeypatch):
        # Only a *trailing* /v1 is the OpenAI base; one inside the host must survive.
        monkeypatch.setattr(settings, "BRIDGE_ENDPOINT", "http://127.0.0.1:18383/v1beta")
        assert bridge_health_url() == "http://127.0.0.1:18383/v1beta/health"


class TestBridgeHealthProbe:
    def test_unreachable_bridge_never_raises(self, monkeypatch):
        # Nothing listens on :1 - the probe must degrade into a report, not a throw.
        monkeypatch.setattr(settings, "BRIDGE_ENDPOINT", "http://127.0.0.1:1/v1")
        state = bridge_health()
        assert state["reachable"] is False
        assert state["endpoint"] == "http://127.0.0.1:1/v1"
        assert "error" in state
        assert "hint" in state

    def test_probe_reports_configured_endpoint(self, monkeypatch):
        monkeypatch.setattr(settings, "BRIDGE_ENDPOINT", "http://bridge:18383/v1")
        state = bridge_health()
        assert state["endpoint"] == "http://bridge:18383/v1"
