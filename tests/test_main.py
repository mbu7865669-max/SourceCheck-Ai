"""
Tests for the application entrypoint (main.py): HTTP routes and CLI argument handling.

The agent pipeline itself is covered by tests/test_agent.py; here we only verify the
transport layer around it - routing, validation, status codes, and the fact that
`python main.py serve` starts a server rather than web-searching the word "serve".
"""

import sys
import types

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
def patch_agent(monkeypatch):
    """Replace main.SourceCheckAgent with a factory returning a canned response."""

    def _install(response: AgentResponse, error: Exception | None = None):
        class FakeAgent:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def answer_question(self, question: str, **kwargs):
                if error is not None:
                    raise error
                return response

        monkeypatch.setattr(main, "SourceCheckAgent", FakeAgent)
        return FakeAgent

    return _install


# ------------------------------------------------------------------------------
# Bridge health URL derivation
# ------------------------------------------------------------------------------


class TestBridgeHealthUrl:
    def test_strips_v1_suffix(self, monkeypatch):
        # The OpenAI base ends in /v1, but /health lives at the bridge root.
        monkeypatch.setattr(settings, "BRIDGE_ENDPOINT", "http://127.0.0.1:18383/v1")
        assert main._bridge_health_url() == "http://127.0.0.1:18383/health"

    def test_no_v1_suffix_left_alone(self, monkeypatch):
        monkeypatch.setattr(settings, "BRIDGE_ENDPOINT", "http://localhost:9000")
        assert main._bridge_health_url() == "http://localhost:9000/health"

    def test_trailing_slash_is_normalised(self, monkeypatch):
        monkeypatch.setattr(settings, "BRIDGE_ENDPOINT", "http://127.0.0.1:18383/v1/")
        assert main._bridge_health_url() == "http://127.0.0.1:18383/health"

    def test_v1_in_middle_of_host_is_not_stripped(self, monkeypatch):
        # Only a *trailing* /v1 is the OpenAI base; one inside the host must survive.
        monkeypatch.setattr(settings, "BRIDGE_ENDPOINT", "http://127.0.0.1:18383/v1beta")
        assert main._bridge_health_url() == "http://127.0.0.1:18383/v1beta/health"


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------


class TestRoutes:
    def test_root_lists_endpoints(self, client):
        body = client.get("/").json()
        assert body["service"] == "SourceCheck AI"
        assert "answer" in body["endpoints"]
        assert body["model_provider"] == settings.MODEL_PROVIDER

    def test_health_reports_provider(self, client, monkeypatch):
        monkeypatch.setattr(settings, "MODEL_PROVIDER", "mock")
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["provider"] == "mock"
        assert "bridge" not in body  # only probed when provider is opencode

    def test_health_probes_bridge_for_opencode_provider(self, client, monkeypatch):
        monkeypatch.setattr(settings, "MODEL_PROVIDER", "opencode")
        monkeypatch.setattr(settings, "BRIDGE_ENDPOINT", "http://127.0.0.1:1/v1")
        body = client.get("/health").json()
        assert body["provider"] == "opencode"
        assert body["bridge"]["reachable"] is False  # nothing listens on :1
        assert "opencode_bridge.py" in body["bridge"]["hint"]

    def test_health_never_500s_when_bridge_is_down(self, client, monkeypatch):
        monkeypatch.setattr(settings, "MODEL_PROVIDER", "opencode")
        monkeypatch.setattr(settings, "BRIDGE_ENDPOINT", "http://127.0.0.1:1/v1")
        assert client.get("/health").status_code == 200

    def test_answer_success(self, client, patch_agent):
        patch_agent(ok_response())
        resp = client.post("/answer", json={"question": "What year?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["answer"] == "1969"

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

    def test_answer_accepts_optional_limits(self, client, patch_agent):
        patch_agent(ok_response())
        resp = client.post(
            "/answer",
            json={"question": "What year?", "max_search_results": 3, "max_pages_to_fetch": 2},
        )
        assert resp.status_code == 200
        assert resp.json()["answer"] == "1969"

    def test_model_failure_maps_to_503_with_detail(self, client, patch_agent):
        from app.model import ModelError

        patch_agent(ok_response(), error=ModelError("OpenCode bridge is not reachable"))
        resp = client.post("/answer", json={"question": "What year?"})
        assert resp.status_code == 503
        assert "not reachable" in resp.json()["detail"]

    def test_model_timeout_maps_to_503(self, client, patch_agent):
        from app.model import ModelTimeoutError

        patch_agent(ok_response(), error=ModelTimeoutError("timed out after 120s"))
        resp = client.post("/answer", json={"question": "What year?"})
        assert resp.status_code == 503
        assert "120s" in resp.json()["detail"]

    def test_unexpected_error_maps_to_500_json(self, client, patch_agent):
        patch_agent(ok_response(), error=RuntimeError("boom"))
        resp = client.post("/answer", json={"question": "What year?"})
        assert resp.status_code == 500
        assert resp.json()["detail"] == "Internal agent error"

    def test_error_bodies_are_json_not_html(self, client):
        resp = client.post("/answer", json={})
        assert resp.headers["content-type"].startswith("application/json")
        assert "detail" in resp.json()


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
