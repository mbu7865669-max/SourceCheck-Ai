"""
Tests for the agent service (agent_service.py) - the pipeline's own HTTP surface.

The web gateway proxies to this service, so everything the gateway's tests stub
out (running the pipeline, mapping model failures to 503, probing the bridge)
is exercised for real here.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

import agent_service
from app.config import settings
from app.model import MockLLM, ModelError, ModelTimeoutError
from app.schemas import AgentResponse


@pytest.fixture
def client() -> TestClient:
    return TestClient(agent_service.create_app())


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
    """Replace agent_service.SourceCheckAgent with a factory returning a canned response."""

    def _install(response: AgentResponse, error: Exception | None = None):
        answers: list = []  # one record per answer_question(...) call

        class FakeAgent:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def answer_question(self, question: str, **kwargs):
                answers.append({"question": question, **kwargs})
                if error is not None:
                    raise error
                return response

        FakeAgent.answers = answers
        monkeypatch.setattr(agent_service, "SourceCheckAgent", FakeAgent)
        return FakeAgent

    return _install


# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------


class TestRoutes:
    def test_root_describes_the_service(self, client):
        body = client.get("/").json()
        assert body["service"] == "SourceCheck AI - Agent"
        assert "answer" in body["endpoints"]
        assert body["model_provider"] == settings.MODEL_PROVIDER

    def test_health_reports_provider(self, client, monkeypatch):
        monkeypatch.setattr(settings, "MODEL_PROVIDER", "mock")
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["service"] == "agent"
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

    def test_health_uses_mock_provider_by_default(self, client):
        # Without env overrides the pipeline defaults to the offline mock model.
        assert client.get("/health").json()["provider"] == settings.MODEL_PROVIDER.lower()


# ------------------------------------------------------------------------------
# POST /answer - the real pipeline surface
# ------------------------------------------------------------------------------


class TestAnswer:
    def test_answer_success(self, client, patch_agent):
        patch_agent(ok_response())
        resp = client.post("/answer", json={"question": "What year?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["answer"] == "1969"
        assert body["notices"] == []

    def test_answer_forwards_overrides_to_the_pipeline(self, client, patch_agent):
        fake = patch_agent(ok_response())
        resp = client.post(
            "/answer",
            json={"question": "What year?", "max_search_results": 3, "max_evidence_passages": 4},
        )
        assert resp.status_code == 200
        # The gateway serializes `exclude_none`, so only overrides actually
        # supplied here reach the pipeline - both must arrive intact.
        assert len(fake.answers) == 1
        assert fake.answers[0]["question"] == "What year?"
        assert fake.answers[0]["max_search_results"] == 3
        assert fake.answers[0]["max_evidence_passages"] == 4
        assert fake.answers[0]["max_pages_to_fetch"] is None

    def test_answer_rejects_missing_question(self, client):
        assert client.post("/answer", json={}).status_code == 422

    def test_answer_rejects_empty_question(self, client):
        assert client.post("/answer", json={"question": ""}).status_code == 422

    def test_answer_rejects_out_of_range_limits(self, client):
        resp = client.post("/answer", json={"question": "q", "max_pages_to_fetch": 99})
        assert resp.status_code == 422

    def test_model_failure_maps_to_503_with_detail(self, client, patch_agent):
        patch_agent(ok_response(), error=ModelError("OpenCode bridge is not reachable"))
        resp = client.post("/answer", json={"question": "What year?"})
        assert resp.status_code == 503
        assert "not reachable" in resp.json()["detail"]

    def test_model_timeout_maps_to_503(self, client, patch_agent):
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
# The pipeline still runs in-process (mock model, no network)
# ------------------------------------------------------------------------------


class TestEndToEndMock:
    def test_real_pipeline_answers_over_http(self, client, monkeypatch):
        """The real SourceCheckAgent + offline mock model, reached through HTTP.

        SEARCH_PROVIDER=mock keeps it offline (no DuckDuckGo call), so this
        exercises routing, validation and the pipeline wiring end to end
        without touching the network.
        """
        monkeypatch.setattr(settings, "MODEL_PROVIDER", "mock")
        monkeypatch.setattr(settings, "SEARCH_PROVIDER", "mock")
        monkeypatch.setattr(agent_service, "get_model", lambda **_: MockLLM())
        resp = client.post(
            "/answer",
            json={"question": "What year did the first human walk on the Moon?"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] in ("success", "insufficient_evidence", "error")
        assert isinstance(body["notices"], list)
        assert isinstance(body["errors"], list)


def test_livez_never_probes_the_bridge(client, monkeypatch):
    # The container HEALTHCHECK hits /livez every few seconds; it must stay local.
    def boom():
        raise AssertionError("/livez must not probe the bridge")

    monkeypatch.setattr(agent_service, "bridge_health", boom)
    resp = client.get("/livez")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "agent"}


# ------------------------------------------------------------------------------
# Model picker: GET /models and the per-request `model` field
# ------------------------------------------------------------------------------


def _bridge_transport(handler):
    """Route agent_service's httpx.Client through a MockTransport."""
    real_client = httpx.Client

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    return factory


def test_models_passes_the_bridge_catalogue_through(client, monkeypatch):
    monkeypatch.setattr(settings, "MODEL_PROVIDER", "opencode")
    monkeypatch.setattr(settings, "BRIDGE_ENDPOINT", "http://bridge:18383/v1")
    monkeypatch.setattr(settings, "BRIDGE_MODEL", "opencode/a-free")
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={"object": "list", "data": [
            {"id": "opencode/a-free"}, {"id": "opencode/b-free"}, {"object": "junk"}]})

    monkeypatch.setattr(agent_service.httpx, "Client", _bridge_transport(handler))
    body = client.get("/models").json()
    assert seen == ["http://bridge:18383/v1/models"]
    assert body == {"provider": "opencode", "default": "opencode/a-free",
                    "models": ["opencode/a-free", "opencode/b-free"]}


def test_models_is_503_when_the_bridge_is_down(client, monkeypatch):
    monkeypatch.setattr(settings, "MODEL_PROVIDER", "opencode")

    def handler(request):
        raise httpx.ConnectError("refused", request=request)

    monkeypatch.setattr(agent_service.httpx, "Client", _bridge_transport(handler))
    resp = client.get("/models")
    assert resp.status_code == 503
    assert "catalogue unavailable" in resp.json()["detail"]


def test_models_is_empty_for_providers_without_a_catalogue(client, monkeypatch):
    monkeypatch.setattr(settings, "MODEL_PROVIDER", "mock")
    assert client.get("/models").json() == {"provider": "mock", "default": None, "models": []}


def test_answer_uses_the_requested_model(client, monkeypatch, patch_agent):
    patch_agent(ok_response())
    monkeypatch.setattr(settings, "MODEL_PROVIDER", "opencode")
    monkeypatch.setattr(agent_service, "_bridge_catalogue", lambda: ["opencode/b-free"])
    requested = []
    monkeypatch.setattr(
        agent_service, "get_model", lambda **kw: requested.append(kw.get("model_name")) or MockLLM()
    )
    client.post("/answer", json={"question": "What year?", "model": "opencode/b-free"})
    client.post("/answer", json={"question": "What year?"})
    assert requested == ["opencode/b-free", None]


def test_answer_rejects_a_model_the_bridge_does_not_serve(client, monkeypatch, patch_agent):
    fake = patch_agent(ok_response())
    monkeypatch.setattr(settings, "MODEL_PROVIDER", "opencode")
    monkeypatch.setattr(agent_service, "_bridge_catalogue", lambda: ["opencode/b-free"])
    resp = client.post("/answer", json={"question": "What year?", "model": "openai/gpt-5"})
    assert resp.status_code == 400
    assert "not served by the bridge" in resp.json()["detail"]
    assert fake.answers == []  # rejected before any retrieval ran
