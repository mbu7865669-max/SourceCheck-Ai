"""
Agent service - the retrieval/verification/reasoning pipeline as its own HTTP service.

One of the three containers in the stack:

    web     gateway + Web UI (:8000, published as host :80)
    agent   this service (:8001, internal to the compose network)
    bridge  OpenAI-compatible OpenCode bridge (:18383)

The web gateway proxies POST /answer here, so the pipeline can scale, restart
and fail independently of the UI, and never holds a browser request open while
the bridge thinks. Run it with:

    python agent_service.py           # uvicorn agent_service:app on :8001

Endpoints:

    GET  /health   liveness + configured model provider + bridge reachability
    GET  /         service metadata
    POST /answer   run the full retrieval -> evidence -> reasoning pipeline
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.agent import SourceCheckAgent
from app.config import settings
from app.health import bridge_health
from app.model import ModelError, ModelTimeoutError, get_model
from app.schemas import AgentResponse, AnswerRequest

logger = logging.getLogger("sourcecheck.agent")


def create_app() -> FastAPI:
    """Build the agent's FastAPI application. Kept as a factory so tests can mount it."""
    app = FastAPI(
        title="SourceCheck AI - Agent",
        description=(
            "Autonomous web information retrieval agent: retrieve, verify, reason, "
            "and return grounded, cited answers. Normally reached through the web "
            "gateway, but callable directly on the compose network."
        ),
        version="1.1.0",
    )

    @app.get("/livez", include_in_schema=False)
    def livez() -> Dict[str, str]:
        """Process liveness for the container HEALTHCHECK - no downstream probes.

        `/health` fans out to the next hop(s) on every call; polling it from
        Docker every few seconds was pure overhead and log noise, since it
        never fails on a downstream outage anyway.
        """
        return {"status": "ok", "service": "agent"}

    @app.get("/health")
    def health() -> Dict[str, Any]:
        """Liveness probe plus a cheap check that the configured model is reachable."""
        provider = settings.MODEL_PROVIDER.lower().strip()
        payload: Dict[str, Any] = {
            "status": "ok",
            "service": "agent",
            "provider": provider,
            "search_provider": settings.SEARCH_PROVIDER,
        }

        if provider == "opencode":
            payload["bridge"] = bridge_health()

        return payload

    @app.get("/models")
    def models() -> Dict[str, Any]:
        """The free models this agent can reason with, for the Web UI's model picker.

        The bridge owns the catalogue (it enforces free-only), so this passes its
        `/v1/models` through rather than keeping a second list that could drift.
        Providers other than `opencode` have nothing to choose between.
        """
        provider = settings.MODEL_PROVIDER.lower().strip()
        if not _uses_bridge(provider):
            return {"provider": provider, "default": None, "models": []}
        return {"provider": provider, "default": settings.BRIDGE_MODEL, "models": _bridge_catalogue()}

    @app.get("/")
    def root() -> Dict[str, Any]:
        """Service metadata for API clients (the UI lives on the web service)."""
        return {
            "service": "SourceCheck AI - Agent",
            "role": "retrieval + verification + reasoning pipeline",
            "endpoints": {"health": "/health", "models": "/models", "answer": "POST /answer"},
            "model_provider": settings.MODEL_PROVIDER,
        }

    @app.post("/answer", response_model=AgentResponse)
    def answer(req: AnswerRequest) -> AgentResponse:
        """Run the full pipeline. Declared `def` so FastAPI runs it in its threadpool."""
        # Check a requested model before retrieval: the bridge would refuse it anyway,
        # but only after a full search + fetch round had been spent for nothing.
        if req.model and _uses_bridge(settings.MODEL_PROVIDER.lower().strip()):
            catalogue = _bridge_catalogue()
            if req.model not in catalogue:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model {req.model!r} is not served by the bridge. See GET /models.",
                )

        try:
            agent = SourceCheckAgent(model=get_model(model_name=req.model))
            return agent.answer_question(
                question=req.question,
                max_search_results=req.max_search_results,
                max_pages_to_fetch=req.max_pages_to_fetch,
                max_evidence_passages=req.max_evidence_passages,
            )
        except (ModelError, ModelTimeoutError) as exc:
            # The model layer already produced a precise message - surface it as a 503
            # rather than a bare 500, since it almost always means "backend is down".
            logger.error("Model backend failure: %s", exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Unhandled failure in /answer")
            raise HTTPException(status_code=500, detail="Internal agent error") from exc

    @app.exception_handler(HTTPException)
    def _json_errors(_request: Request, exc: HTTPException) -> JSONResponse:
        """Keep error bodies JSON-shaped so the gateway can pass them straight through."""
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return app


def _uses_bridge(provider: str) -> bool:
    return provider in ("opencode", "opencode_bridge", "bridge")


def _bridge_catalogue() -> List[str]:
    """Model ids the bridge serves (free-only). Raises a 503 when it can't be read."""
    url = f"{settings.BRIDGE_ENDPOINT.rstrip('/')}/models"
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json().get("data", [])
    except (httpx.HTTPError, ValueError) as exc:
        logger.error("Bridge model catalogue unavailable at %s: %s", url, exc)
        raise HTTPException(
            status_code=503, detail=f"Bridge model catalogue unavailable at {url}"
        ) from exc
    return [entry["id"] for entry in data if isinstance(entry, dict) and entry.get("id")]


# ASGI entrypoint for `uvicorn agent_service:app`.
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "agent_service:app",
        host=settings.HOST,
        port=settings.AGENT_PORT,
        log_level=settings.LOG_LEVEL.lower(),
    )
