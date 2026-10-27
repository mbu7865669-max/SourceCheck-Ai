"""
Tests for the web gateway (main.py): HTTP routes and CLI argument handling.

The gateway proxies POST /answer to the agent service, so the network hop is
stubbed here and the pipeline itself is covered by tests/test_agent.py; the
agent's own HTTP surface lives in tests/test_agent_service.py. Here we verify
the transport layer - routing, validation, status codes, passthrough of the
agent's errors, and the fact that `python main.py serve` starts a server rather
than web-searching the word "serve".
"""

import sys
import types

import httpx
import pytest
from fastapi.testclient import TestClient

import main
from app.config import settings
from app.schemas import AgentResponse


# ------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    # Import inside the fixture so each test sees a freshly built app.
    return TestClient(main.create_app())


def ok_response(question: str = "What year?") -> AgentResponse:
    return AgentResponse(
        question=question,
        answer="1969",
        sources=["https://example.com/a"],
        claims=[],
        insufficient_evidence=False,
        status="success",
    )


@pytest.fixture
def patch_call_agent(monkeypatch):
    """Replace main._call_agent - the HTTP hop to the agent service.

    `_install(response=...)` answers with a canned AgentResponse (or raw dict),
    `_install(error=...)` simulates the agent being unreachable, and the returned
    object records the payloads that were forwarded for assertions.
    """

    def _install(
        response: AgentResponse | dict | None = None,
        error: Exception | None = None,
        status_code: int = 200,
    ):
        sent: list = []

        def _call(payload: dict) -> httpx.Response:
            sent.append(payload)
            if error is not None:
                raise error
            if response is None:
                body: dict = ok_response().model_dump()
            elif isinstance(response, AgentResponse):
                body = response.model_dump()
            else:
                body = response
            return httpx.Response(status_code, json=body)

        monkeypatch.setattr(main, "_call_agent", _call)
        return sent

    return _install


# ------------------------------------------------------------------------------
# Agent health URL derivation
# ------------------------------------------------------------------------------


class TestAgentHealthUrl:
    def test_appends_health_path(self, monkeypatch):
        monkeypatch.setattr(settings, "AGENT_ENDPOINT", "http://agent:8001")
        assert main._agent_health_url() == "http://agent:8001/health"

    def test_trailing_slash_is_normalised(self, monkeypatch):
        monkeypatch.setattr(settings, "AGENT_ENDPOINT", "http://agent:8001/")
        assert main._agent_health_url() == "http://agent:8001/health"


# ------------------------------------------------------------------------------
# Agent reachability probe (never raises)
# ------------------------------------------------------------------------------


class TestAgentHealthProbe:
    def test_reachable_agent_reports_body(self, monkeypatch):
        # Nothing listens on :1, so this must degrade into reachable=False...
        monkeypatch.setattr(settings, "AGENT_ENDPOINT", "http://127.0.0.1:1")
        state = main._agent_health()
        assert state["reachable"] is False
        assert state["endpoint"] == "http://127.0.0.1:1"
        assert "error" in state

    def test_probe_never_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "AGENT_ENDPOINT", "not-a-url")
        assert isinstance(main._agent_health(), dict)


# ------------------------------------------------------------------------------
# Content negotiation on /
# ------------------------------------------------------------------------------


class TestWantsHtml:
    """`/` shows the UI to people and metadata to machines - this decides which."""

    @pytest.mark.parametrize(
        "accept,expected",
        [
            # Browsers: HTML first, no JSON type at all.
            ("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8", True),
            # curl / httpx / TestClient default: not a page request.
            ("*/*", False),
            ("application/json", False),
            ("", False),
            # Explicit ordering wins when both appear.
            ("text/html,application/json", True),
            ("application/json,text/html", False),
            ("text/html;q=0.9,application/json;q=1.0", False),
        ],
    )
    def test_accept_header_decision(self, accept, expected):
        assert main.wants_html(accept) is expected


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------


class TestRoutes:
    def test_root_lists_endpoints(self, client):
        body = client.get("/").json()
        assert body["service"] == "SourceCheck AI"
        assert "answer" in body["endpoints"]
        assert body["model_provider"] == settings.MODEL_PROVIDER

    def test_root_points_at_the_ui(self, client):
        assert client.get("/").json()["ui"] == "/ui"

    def test_root_serves_ui_to_browsers(self, client):
        # What Chrome/Firefox actually send to a fresh page load.
        browser = ("text/html,application/xhtml+xml,application/xml;q=0.9,"
                   "image/avif,image/webp,*/*;q=0.8")
        resp = client.get("/", headers={"accept": browser})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert 'id="question"' in resp.text  # the page (its question box), not metadata

    def test_root_serves_json_to_api_clients(self, client):
        for accept in ("application/json", "*/*", ""):
            resp = client.get("/", headers={"accept": accept})
            assert resp.headers["content-type"].startswith("application/json"), accept
            assert resp.json()["service"] == "SourceCheck AI"

    def test_ui_serves_the_browser_page(self, client):
        resp = client.get("/ui")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        # The page is the client for this very API, so it must reference both routes.
        assert "SourceCheck AI" in resp.text
        assert "/answer" in resp.text
        assert "/health" in resp.text

    def test_models_come_from_the_agent(self, client, monkeypatch):
        catalogue = {"provider": "opencode", "default": "opencode/a-free",
                     "models": ["opencode/a-free", "opencode/b-free"]}
        monkeypatch.setattr(main, "_fetch_agent_models",
                            lambda: httpx.Response(200, json=catalogue))
        assert client.get("/models").json() == catalogue

    def test_models_is_503_when_the_agent_is_down(self, client, monkeypatch):
        def down():
            raise httpx.ConnectError("refused")

        monkeypatch.setattr(main, "_fetch_agent_models", down)
        resp = client.get("/models")
        assert resp.status_code == 503
        assert "Agent service unreachable" in resp.json()["detail"]

    def test_answer_forwards_the_chosen_model(self, client, monkeypatch):
        sent = []
        monkeypatch.setattr(main, "_call_agent",
                            lambda payload: sent.append(payload) or httpx.Response(
                                200, json={"question": "q", "answer": "a"}))
        client.post("/answer", json={"question": "q", "model": "opencode/b-free"})
        assert sent[0]["model"] == "opencode/b-free"

    def test_livez_never_probes_the_agent(self, client, monkeypatch):
        # The container HEALTHCHECK hits /livez every few seconds; it must stay local.
        def boom():
            raise AssertionError("/livez must not probe the agent")

        monkeypatch.setattr(main, "_agent_health", boom)
        resp = client.get("/livez")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "service": "web"}

    def test_ui_is_compressed_and_revalidated(self, client):
        resp = client.get("/ui", headers={"accept-encoding": "gzip"})
        assert resp.headers["content-encoding"] == "gzip"
        # Revalidate every load (cheap 304) so a redeploy never leaves a stale UI.
        assert resp.headers["cache-control"] == "no-cache"

        etag = resp.headers["etag"]
        again = client.get("/ui", headers={"if-none-match": etag})
        assert again.status_code == 304

    def test_ui_is_not_exposed_in_openapi(self, client):
        # It is a page, not an API endpoint - keep the generated schema clean.
        assert "/ui" not in client.get("/openapi.json").json()["paths"]

    def test_health_reports_agent_view(self, client, monkeypatch):
        # The gateway merges whatever the agent reports (provider, bridge, ...).
        monkeypatch.setattr(
            main,
            "_agent_health",
            lambda: {
                "endpoint": "http://agent:8001",
                "reachable": True,
                "body": {"status": "ok", "provider": "mock", "search_provider": "duckduckgo"},
            },
        )
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["service"] == "web"
        assert body["agent"]["reachable"] is True
        assert body["provider"] == "mock"
        assert "bridge" not in body  # agent only probes it when provider is opencode

    def test_health_merges_bridge_details_from_agent(self, client, monkeypatch):
        monkeypatch.setattr(
            main,
            "_agent_health",
            lambda: {
                "endpoint": "http://agent:8001",
                "reachable": True,
                "body": {
                    "provider": "opencode",
                    "bridge": {"endpoint": "http://bridge:18383/v1", "reachable": True},
                },
            },
        )
        body = client.get("/health").json()
        assert body["bridge"]["reachable"] is True

    def test_health_degrades_when_agent_is_down(self, client, monkeypatch):
        # The gateway itself is alive even when the pipeline is not: report
        # `degraded` (not a 500) and synthesize the fields the UI pill reads.
        monkeypatch.setattr(settings, "AGENT_ENDPOINT", "http://agent:8001")
        monkeypatch.setattr(
            main,
            "_agent_health",
            lambda: {"endpoint": "http://agent:8001", "reachable": False, "error": "refused"},
        )
        body = client.get("/health").json()
        assert body["status"] == "degraded"
        assert body["agent"]["reachable"] is False
        assert body["provider"] == "unknown"
        assert body["bridge"]["reachable"] is False
        assert "http://agent:8001" in body["bridge"]["hint"]

    def test_health_never_500s_when_agent_is_down(self, client, monkeypatch):
        monkeypatch.setattr(
            main,
            "_agent_health",
            lambda: {"endpoint": "http://agent:8001", "reachable": False, "error": "refused"},
        )
        assert client.get("/health").status_code == 200

    def test_answer_success(self, client, patch_call_agent):
        patch_call_agent(ok_response())
        resp = client.post("/answer", json={"question": "What year?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["answer"] == "1969"
        assert body["notices"] == []  # additive field, always present

    def test_answer_rejects_missing_question(self, client):
        resp = client.post("/answer", json={})
        assert resp.status_code == 422

    def test_answer_rejects_empty_question(self, client):
        # min_length=1 keeps whitespace/empty input out of the pipeline entirely.
        resp = client.post("/answer", json={"question": ""})
        assert resp.status_code == 422

    def test_answer_rejects_out_of_range_limits(self, client):
        resp = client.post("/answer", json={"question": "q", "max_search_results": 0})
        assert resp.status_code == 422

    def test_answer_forwards_optional_limits(self, client, patch_call_agent):
        sent = patch_call_agent(ok_response())
        resp = client.post(
            "/answer",
            json={"question": "What year?", "max_search_results": 3, "max_pages_to_fetch": 2},
        )
        assert resp.status_code == 200
        assert resp.json()["answer"] == "1969"
        # The overrides must survive the hop, or the UI's tuning knobs are lies.
        assert sent[0]["max_search_results"] == 3
        assert sent[0]["max_pages_to_fetch"] == 2
        assert "max_evidence_passages" not in sent[0]  # omitted when unset

    def test_model_failure_passes_through_as_503_with_detail(self, patch_call_agent, client):
        patch_call_agent({"detail": "OpenCode bridge is not reachable"}, status_code=503)
        resp = client.post("/answer", json={"question": "What year?"})
        assert resp.status_code == 503
        assert "not reachable" in resp.json()["detail"]

    def test_model_timeout_passes_through_as_503(self, patch_call_agent, client):
        patch_call_agent({"detail": "timed out after 120s"}, status_code=503)
        resp = client.post("/answer", json={"question": "What year?"})
        assert resp.status_code == 503
        assert "120s" in resp.json()["detail"]

    def test_agent_crash_passes_through_as_500_json(self, patch_call_agent, client):
        patch_call_agent({"detail": "Internal agent error"}, status_code=500)
        resp = client.post("/answer", json={"question": "What year?"})
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Internal agent error"

    def test_unreachable_agent_maps_to_503(self, client, patch_call_agent, monkeypatch):
        monkeypatch.setattr(settings, "AGENT_ENDPOINT", "http://agent:8001")
        patch_call_agent(error=httpx.ConnectError("connection refused"))
        resp = client.post("/answer", json={"question": "What year?"})
        assert resp.status_code == 503
        detail = resp.json()["detail"]
        assert "unreachable" in detail
        assert "http://agent:8001" in detail  # names where it tried to go

    def test_agent_timeout_maps_to_503(self, client, patch_call_agent):
        patch_call_agent(error=httpx.ReadTimeout("timed out"))
        resp = client.post("/answer", json={"question": "What year?"})
        assert resp.status_code == 503
        assert "timed out" in resp.json()["detail"]

    def test_non_json_agent_response_maps_to_502(self, client, monkeypatch):
        # A broken hop is a bad gateway, not a silent success or a bare 500.
        monkeypatch.setattr(
            main, "_call_agent", lambda _payload: httpx.Response(200, text="<html>not json</html>")
        )
        resp = client.post("/answer", json={"question": "What year?"})
        assert resp.status_code == 502
        assert "agent" in resp.json()["detail"].lower()

    def test_error_bodies_are_json_not_html(self, client):
        resp = client.post("/answer", json={})
        assert resp.headers["content-type"].startswith("application/json")
        assert "detail" in resp.json()

    def test_root_names_the_agent_endpoint(self, client, monkeypatch):
        monkeypatch.setattr(settings, "AGENT_ENDPOINT", "http://agent:8001")
        assert client.get("/").json()["agent"]["endpoint"] == "http://agent:8001"


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------


@pytest.fixture
def fake_uvicorn(monkeypatch):
    """Intercept the late `import uvicorn` inside run_cli()."""
    calls = []
    module = types.ModuleType("uvicorn")
    module.run = lambda *args, **kwargs: calls.append((args, kwargs))
    monkeypatch.setitem(sys.modules, "uvicorn", module)
    return calls


class TestCli:
    def test_bare_serve_word_starts_server(self, fake_uvicorn, monkeypatch):
        # Regression: this used to fall through and web-search the word "serve".
        monkeypatch.setattr(settings, "MODEL_PROVIDER", "mock")
        assert main.run_cli(["serve"]) == 0
        assert len(fake_uvicorn) == 1

    @pytest.mark.parametrize("word", ["serve", "server", "api", "SERVE", " Serve "])
    def test_serve_spellings_are_recognised(self, word, fake_uvicorn, monkeypatch):
        monkeypatch.setattr(settings, "MODEL_PROVIDER", "mock")
        assert main.run_cli([word]) == 0
        assert len(fake_uvicorn) == 1

    def test_serve_flag_starts_server(self, fake_uvicorn):
        assert main.run_cli(["--serve"]) == 0
        assert len(fake_uvicorn) == 1

    def test_serve_configures_host_and_port(self, fake_uvicorn, monkeypatch):
        monkeypatch.setattr(settings, "HOST", "127.0.0.1")
        monkeypatch.setattr(settings, "PORT", 9999)
        main.run_cli(["serve"])
        _, kwargs = fake_uvicorn[0]
        assert kwargs["host"] == "127.0.0.1"
        assert kwargs["port"] == 9999

    def test_no_arguments_exits_with_usage(self):
        with pytest.raises(SystemExit) as exc:
            main.run_cli([])
        assert exc.value.code == 2  # argparse.usage error, not a traceback

    def test_provider_override_is_applied(self, monkeypatch, capsys):
        monkeypatch.setattr(settings, "MODEL_PROVIDER", "mock")

        captured = {}

        class FakeAgent:
            def __init__(self, **kwargs):
                captured["model"] = kwargs.get("model")

            def answer_question(self, question, **kwargs):
                return ok_response(question)

        monkeypatch.setattr(main, "SourceCheckAgent", FakeAgent)
        assert main.run_cli(["--provider", "mock", "What year?"]) == 0
        # The factory must have been consulted with the overridden provider.
        from app.model import MockLLM

        assert isinstance(captured["model"], MockLLM)

    def test_exit_code_reflects_insufficient_evidence(self, monkeypatch):
        class FakeAgent:
            def __init__(self, **kwargs):
                pass

            def answer_question(self, question, **kwargs):
                return AgentResponse(
                    question=question,
                    answer="Insufficient factual evidence found to answer the question.",
                    sources=[],
                    claims=[],
                    insufficient_evidence=True,
                    status="insufficient_evidence",
                )

        monkeypatch.setattr(main, "SourceCheckAgent", FakeAgent)
        assert main.run_cli(["What year?"]) == 1

    def test_json_mode_emits_machine_readable_output(self, monkeypatch, capsys):
        class FakeAgent:
            def __init__(self, **kwargs):
                pass

            def answer_question(self, question, **kwargs):
                return ok_response(question)

        monkeypatch.setattr(main, "SourceCheckAgent", FakeAgent)
        assert main.run_cli(["--json", "What year?"]) == 0
        out = capsys.readouterr().out
        import json as _json

        parsed = _json.loads(out)
        assert parsed["answer"] == "1969"
        assert parsed["status"] == "success"
